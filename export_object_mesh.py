import torch
from scene import Scene
import os
import sys
import yaml
from tqdm import tqdm
from os import makedirs
import torchvision
from argparse import ArgumentParser
from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh
from utils.general_utils import parse_cfg
import open3d as o3d

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument('-m', '--model_path', type=str, required=True)
    parser.add_argument("--scene_name", default=None)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--voxel_size", default=-1.0, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument("--num_cluster", default=10, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--query_label_id", default=-1, type=int, help='Mesh: id of queried gaussians')
    parser.add_argument("--unbounded", action="store_true", help='Mesh: using unbounded mode for meshing')
    parser.add_argument("--mesh_res", default=2048, type=int, help='Mesh: resolution for unbounded mesh extraction')
    args = parser.parse_args(sys.argv[1:])
    
    with open(os.path.join(args.model_path, "config.yaml")) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    args.scene_name = args.model_path.split('/')[-2]

    if args.scene_name is not None:
        try:
            cfg["model_params"]["exp_name"] = os.path.join(cfg["model_params"]["exp_name"], args.scene_name)
            cfg["model_params"]["source_path"] = os.path.join(cfg["model_params"]["source_path"], args.scene_name)
        except:
            print("OverrideError: Cannot override 'exp_name' and 'source_path' in 'model_params'. Exiting.")
            sys.exit(1)
            
    lp, op, pp = parse_cfg(cfg)
    lp.model_path = args.model_path

    print("Rendering " + args.model_path)
    
    modules = __import__('scene')
    model_config = lp.model_config
    iteration = args.iteration
    gaussians = getattr(modules, model_config['name'])(**model_config['kwargs'])
    scene = Scene(lp, gaussians, load_iteration=iteration, shuffle=False)

    queried_object_mask = gaussians.label_ids.squeeze() == args.query_label_id

    modules = __import__('gaussian_renderer')
    gaussExtractor = GaussianExtractor(gaussians, getattr(modules, 'render'), pp, scene.background, queried_object_mask) 

    # set the active_sh to 0 to export only diffuse texture
    train_dir = os.path.join(args.model_path, 'train', "id_{}".format(args.query_label_id), "mesh")
    os.makedirs(train_dir, exist_ok=True)
    if gaussExtractor.gaussians.active_sh_degree != None: 
        gaussExtractor.gaussians.active_sh_degree = 0
    gaussExtractor.reconstruction(scene.getTrainCameras())
    # extract the mesh and save
    if args.unbounded:
        name = 'fuse_unbounded.ply'
        mesh = gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res)
    else:
        name = 'fuse.ply'
        depth_trunc = (gaussExtractor.radius * 2.0)*5 if args.depth_trunc < 0  else args.depth_trunc
        # depth_trunc = (gaussExtractor.radius * 2.0) if args.depth_trunc < 0  else args.depth_trunc
        voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
        sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
        mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)

    o3d.io.write_triangle_mesh(os.path.join(train_dir, name), mesh)
    print("mesh saved at {}".format(os.path.join(train_dir, name)))
    # post-process the mesh and save, saving the largest N clusters
    mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
    o3d.io.write_triangle_mesh(os.path.join(train_dir, name.replace('.ply', '_post.ply')), mesh_post)
    print("mesh post processed saved at {}".format(os.path.join(train_dir, name.replace('.ply', '_post.ply'))))