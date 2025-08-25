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

import cv2
import torch
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import concurrent.futures
from tqdm import tqdm
from colorama import Fore, init, Style
from PIL import Image
import re

WARNED = False

def loadCam(args, id, cam_info, resolution_scale, background):
    orig_w, orig_h = cam_info.image.size
        
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))
        
    
    
    return Camera(resolution, colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, Cx=cam_info.CX, Cy=cam_info.CY, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, resolution_scale=resolution_scale, image=cam_info.image, 
                  alpha_mask=cam_info.mask, image_name=cam_info.image_name, image_path=cam_info.image_path, 
                  uid=id, data_device=args.data_device,data_format=args.data_format, gt_depth=cam_info.depth, depth_params=cam_info.depth_params)

def cameraList_from_camInfos(cam_infos, resolution_scale, args, background):
    camera_list = []

    init = torch.inverse(torch.ones((1, 1), device="cuda"))
    ct = 0 
    progress_bar = tqdm(cam_infos, desc="Processing image")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(loadCam, args, cam_id, c, resolution_scale, background) for cam_id, c in enumerate(cam_infos)]

        for future in concurrent.futures.as_completed(futures):
            camera = future.result()
            camera_list.append(camera)
            
            ct+=1
            if ct % 10 == 0 or ct == len(cam_infos):
                progress_bar.set_postfix({"num": Fore.YELLOW+f"{ct}/{len(cam_infos)}"+Style.RESET_ALL})
                progress_bar.update(10)
        
        progress_bar.close()

    camera_list = sorted(camera_list, key = lambda x : x.image_path)
    return camera_list
    
def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


