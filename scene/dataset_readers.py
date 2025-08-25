#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import glob
import sys
import re
import cv2
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
from PIL import Image
from tqdm import tqdm
from typing import NamedTuple
from colorama import Fore, init, Style
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
try:
    import laspy
except:
    print("No laspy")
from utils.graphics_utils import BasicPointCloud
import concurrent.futures

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    CX: np.array
    CY: np.array
    image: np.array
    mask: np.array
    depth: np.array
    depth_params: dict
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    
def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def read_las_file(path):
    las = laspy.read(path)
    positions = np.vstack((las.x, las.y, las.z)).transpose()
    try:
        colors = np.vstack((las.red, las.green, las.blue)).transpose()
    except:
        colors = np.random.rand(positions.shape[0], positions.shape[1])
    normals = np.random.rand(positions.shape[0], positions.shape[1])

    return positions, colors, normals

def read_multiple_las_files(paths, ply_path):
    all_positions = []
    all_colors = []
    all_normals = []

    for path in paths:
        positions, colors, normals = read_las_file(path)
        all_positions.append(positions)
        all_colors.append(colors)
        all_normals.append(normals)

    all_positions = np.vstack(all_positions)
    all_colors = np.vstack(all_colors)
    all_normals = np.vstack(all_normals)

    print("Saving point cloud to .ply file...")
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    elements = np.empty(all_positions.shape[0], dtype=dtype)
    attributes = np.concatenate((all_positions, all_normals, all_colors), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(ply_path)

    return BasicPointCloud(points=all_positions, colors=all_colors, normals=all_normals)

def fetchPly(path):
    print("read ply file from {}".format(path))
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

    try:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    except:
        colors = np.random.rand(positions.shape[0], positions.shape[1])
    try:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    except:
        normals = np.random.rand(positions.shape[0], positions.shape[1])
    try:
        label_ids = np.array(vertices['label']).T
    except:
        label_ids = np.zeros_like(vertices['x'])

    return BasicPointCloud(points=positions, colors=colors, normals=normals, label_ids=label_ids)

def storePly(path, xyz, rgb, label_ids):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('label', 'u1')]
    
    normals = np.zeros_like(xyz)
    label_ids = label_ids.reshape(-1, 1)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb, label_ids), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)
    
def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, masks_folder, depths_folder):
    cam_infos = []
    
    def process_frame(idx, key):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE" or intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            CX, CY = intr.params[1], intr.params[2]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE" or intr.model=="OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            CX, CY = intr.params[2], intr.params[3]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        
        image_path = os.path.join(images_folder, extr.name)
        if not os.path.exists(image_path):
            return None
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name.split(".")[0]]
            except:
                print("\n", key, "not found in depths_params")
        if masks_folder is not None:
            mask_path = os.path.join(masks_folder, extr.name)
            mask = Image.open(mask_path)
        else:   
            mask = None
        if depths_folder is not None:
            depth_path = os.path.join(depths_folder, extr.name.replace(".JPG", ".png")) 
            # depth_path = os.path.join(depths_folder, extr.name.replace(".jpg", ".png")) 
            depth = cv2.imread(depth_path, -1).astype(np.float32) / float(2**16)
        else:
            depth = None

        return CameraInfo(
            uid=uid, 
            R=R, 
            T=T, 
            FovY=FovY,
            FovX=FovX, 
            CX=CX,
            CY=CY,
            image=image,
            mask=mask,
            depth=depth,
            depth_params=depth_params,
            image_path=image_path, 
            image_name=image_name, 
            width=width, 
            height=height
        )

    ct = 0
    progress_bar = tqdm(cam_extrinsics, desc="Loading dataset")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_frame, idx, key) for idx, key in enumerate(cam_extrinsics)]

        for future in concurrent.futures.as_completed(futures):
            cam_info = future.result()
            if cam_info is None:
                continue
            else:
                cam_infos.append(cam_info)
            
            ct+=1
            if ct % 10 == 0:
                progress_bar.set_postfix({"num": Fore.YELLOW+f"{ct}/{len(cam_extrinsics)}"+Style.RESET_ALL})
                progress_bar.update(10)
        
        progress_bar.close()

    cam_infos = sorted(cam_infos, key = lambda x : x.image_path)
    return cam_infos

def readCamerasFromTransforms(path, transformsfile, add_mask, add_depth, center, scale, depth_type='ue', depths_params=None):

    cam_infos = []
    test_cam_infos = []
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        # fovx = contents["camera_angle_x"]
        fl_y = contents["fl_y"]
        fl_x = contents["fl_x"]
        cx = contents["cx"]
        cy = contents["cy"]
        
        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join("images", frame["file_path"])

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:, [0, 1]] = c2w[:, [1, 0]]  
            c2w[:3, 2:3] *= -1  

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            FovY = focal2fov(fl_y, image.size[1])
            FovX = focal2fov(fl_x, image.size[0])
            CX = cx
            CY = cy

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, CX=CX, CY=CY, image=image, mask=None, depth=None, depth_params=None,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))

        test_frames = contents["test_frames"]
        for idx, frame in enumerate(test_frames):
            cam_name = os.path.join("images", frame["file_path"])

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            # c2w[:3, 1:3] *= -1
            c2w[:, [0, 1]] = c2w[:, [1, 0]]  
            c2w[:3, 2:3] *= -1 

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            FovY = focal2fov(fl_y, image.size[1])
            FovX = focal2fov(fl_x, image.size[0])
            CX = cx
            CY = cy

            test_cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, CX=CX, CY=CY, image=image, mask=None, depth=None, depth_params=None,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))

    cam_infos = sorted(cam_infos, key = lambda x : x.image_path)
    test_cam_infos = sorted(test_cam_infos, key= lambda x : x.image_path)
    return cam_infos, test_cam_infos

def readColmapSceneInfo(path, eval, images, depths, masks, add_mask, add_depth, llffhold=32):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        try:
            cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
            cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
            cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
        except:
            cameras_extrinsic_file = os.path.join(path, "colmap", "images.txt")
            cameras_intrinsic_file = os.path.join(path, "colmap", "cameras_undistorted.txt")
            cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    
    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    depths_params = None
    if add_depth:
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    reading_dir = os.path.join(path, images)
    mask_dir = os.path.join(path, masks) if add_mask else None
    depth_dir = os.path.join(path, depths) if add_depth else None
    cam_infos = readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, reading_dir, mask_dir, depth_dir)
    
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]

        if "lerf" in path:
            if "waldo_kitchen" in path:
                test_frame = ["frame_00053", "frame_00066", "frame_00089", "frame_00140", "frame_00154"]
                test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in test_frame]
            elif "ramen" in path:
                test_frame = ["frame_00006", "frame_00024", "frame_00060", "frame_00065", "frame_00081", "frame_00119", "frame_00128"]
                test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in test_frame]
            elif "figurines" in path:
                test_frame = ["frame_00041", "frame_00105", "frame_00152", "frame_00195"]
                test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in test_frame]
            elif "teatime" in path:
                test_frame = ["frame_00002", "frame_00025", "frame_00043", "frame_00107", "frame_00129", "frame_00140"]
                test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in test_frame]
        if "3dovs" in path:
            if "bed" in path:
                test_frame = ["00", "04", "10", "23", "30"]
                test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in test_frame]
            elif "bench" in path:
                test_frame = ["02", "05", "25", "27", "32"]
                test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in test_frame]
            elif "blue_sofa" in path:
                test_frame = ["03", "05", "13", "24", "27"]
                test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in test_frame]
            elif "covered_desk" in path:
                test_frame = ["00", "01", "11", "26", "29"]
                test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in test_frame]
            elif "lawn" in path:
                test_frame = ["01", "03", "09", "13", "29"]
                test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in test_frame]
            elif "office_desk" in path:
                test_frame = ["03", "07", "12", "14", "20"]
                test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in test_frame]
            elif "room" in path:
                test_frame = ["00", "04", "19", "25", "30"]
                test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in test_frame]
            elif "snacks" in path:
                test_frame = ["04", "08", "18", "26", "40"]
                test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in test_frame]
            elif "sofa" in path:
                test_frame = ["02", "04", "10", "15", "22"]
                test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in test_frame]
            elif "table" in path:
                test_frame = ["00", "02", "14", "26", "30"]
                test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in test_frame]            
        else:
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        # train_cam_infos = cam_infos
        # test_cam_infos = []
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if "test" not in c.image_name]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if "test" in c.image_name]


    nerf_normalization = getNerfppNorm(train_cam_infos)

    if "3dovs" in path or "lerf_ovs" in path:
        ply_path = os.path.join(path, "sparse/0/points3D_deva.ply")
    elif "scannet" in path:
        ply_path = os.path.join(path, "points3D.ply")
    elif "mipnerf360" in path:
        ply_path = os.path.join(path, "points3D.ply")
    else:
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
    # bin_path = os.path.join(path, "sparse/0/points3D.bin")
    # txt_path = os.path.join(path, "sparse/0/points3D.txt")
    
    if not os.path.exists(ply_path):
        # print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        # try:
        #     xyz, rgb, _ = read_points3D_binary(bin_path)
        # except:
        #     xyz, rgb, _ = read_points3D_text(txt_path)
        # storePly(ply_path, xyz, rgb)
        raise FileNotFoundError
    # try:
    print(f'start fetching data from ply file')
    pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readNerfSyntheticInfo(path, eval, add_mask, add_depth, center, scale):
    # print("Reading Training Transforms")
    # train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", add_mask, add_depth, center, scale)
    # print("Reading Test Transforms")
    # test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", add_mask, add_depth, center, scale)
    print("Reading Training and Test Transforms")
    train_cam_infos, test_cam_infos = readCamerasFromTransforms(path, "nerfstudio/transforms_undistorted.json", add_mask, add_depth, center, scale)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_paths = glob.glob(os.path.join(path, "points3D.ply"))
    if len(ply_paths)==0:
        ply_path = os.path.join(path, "points3d.ply")
        # Since this data set has no colmap data, we start with random points
        num_pts = 10_000
        print(f"Generating random point cloud ({num_pts})...")
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        colors = np.random.random((num_pts, 3))
        normals=np.zeros((num_pts, 3))
        pcd = BasicPointCloud(points=xyz, colors=colors, normals=normals)

        storePly(ply_path, xyz, colors*255)
    else:
        ply_path = ply_paths[0]
        pcd = fetchPly(ply_path)
    
    pcd.points[:, :] -= center
    pcd.points[:, :] /= scale  # mainly adapt to params

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCityInfo(path, eval, add_mask, add_depth, center, scale, ds=1, llffhold=32, depth_type='ue'):
    
    # json_path = glob.glob(os.path.join(path, f"transforms.json"))[0].split('/')[-1]
    json_path = "transforms.json" if ds==1 else f"transforms_{ds}.json"
    print("Reading Training Transforms from {}".format(json_path))
    
    # load ply
    ply_path = glob.glob(os.path.join(path, "*.ply"))[0]
    if os.path.exists(ply_path):
        try:
            pcd = fetchPly(ply_path)
        except:
            raise ValueError("must have tiepoints!")
    else:
        las_paths = glob.glob(os.path.join(path, "LAS/*.las"))
        las_path = las_paths[0]
        print(f'las_path: {las_path}')
        try:
            pcd = read_multiple_las_files(las_paths, ply_path)
        except:
            raise ValueError("Load LAS failed!")
    
    # recenter poses and points clouds
    pcd.points[:,:] -= center
    pcd.points[:,:] /=scale # mainly adapt to params
    
    depth_params_file = os.path.join(path, "depth_params.json")
    depths_params = None
    if add_depth:
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)
            
    # load camera
    cam_infos = readCamerasFromTransforms(path, json_path, add_mask, add_depth, center, scale, depth_type=depth_type, depths_params=depths_params)
    
    print("Load Cameras: ", len(cam_infos))
    train_cam_infos = []
    test_cam_infos = []
    
    if not eval:
        train_cam_infos.extend(cam_infos)
        test_cam_infos = []
    else:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

    
sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "City": readCityInfo
}