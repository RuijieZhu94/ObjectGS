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

import numpy as np
import torch
from torch import nn
from utils.general_utils import PILtoTorch
from utils.graphics_utils import getProjectionMatrix, getWorld2View2
import cv2
import kornia
from PIL import Image

class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, Cx, Cy, FoVx, FoVy, image, alpha_mask, 
                 image_name, image_path, resolution_scale, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 data_format='matrixcity',gt_depth=None, depth_params=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.image_path = image_path
        self.resolution_scale = resolution_scale

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        resized_image_rgb = PILtoTorch(image, resolution).to(self.data_device)
        gt_image = resized_image_rgb[:3, ...]

        if alpha_mask is not None:
            self.alpha_mask = PILtoTorch(alpha_mask, resolution).to(self.data_device)
        elif resized_image_rgb.shape[0] == 4:
            self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
        else: 
            self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))
    
        self.invdepthmap = None # use invdepth to avoid the floater in near places
        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        
        if gt_depth is not None:
            if data_format == 'colmap' or data_format == 'city':
                invdepthmapScaled = gt_depth * depth_params["scale"] + depth_params["offset"]
                invdepthmapScaled = cv2.resize(invdepthmapScaled, resolution)
                invdepthmapScaled[invdepthmapScaled < 0] = 0
                if invdepthmapScaled.ndim != 2:
                    invdepthmapScaled = invdepthmapScaled[..., 0]
                self.invdepthmap = torch.from_numpy(invdepthmapScaled[None]).to(self.data_device)
                
            # elif data_format == 'blender' or data_format == 'city':
            #     gt_depth = torch.from_numpy(cv2.resize(gt_depth, resolution)[None])
            #     if alpha_mask is not None and gt_depth.max()/gt_depth.min() > 100:
            #         assert False, "blender data format does not support alpha mask"
            #         self.alpha_mask = self.alpha_mask * (gt_depth < 0.5*(gt_depth.max()+gt_depth.min())).to(self.data_device)
            #     invdepthmap = 1. / gt_depth
            #     self.invdepthmap = invdepthmap.to(self.data_device)
        else:
            self.invdepthmap = None

        if self.alpha_mask is not None:
            self.depth_mask = self.alpha_mask.clone()
        else:
            self.depth_mask = torch.ones_like(self.invdepthmap > 0)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
        #! defalt pinhole camera model, which needs to be updated

        self.cx = Cx * resolution[0] / image.size[0]
        self.cy = Cy * resolution[1] / image.size[1]
        self.fx = self.image_width / (2 * np.tan(self.FoVx * 0.5))
        self.fy = self.image_height / (2 * np.tan(self.FoVy * 0.5))
        self.c2w = self.world_view_transform.transpose(0, 1).inverse()

        if "3dovs" in image_path:
            object_mask = Image.open(image_path.replace("images", "object_mask_deva").replace(".jpg", ".png")).convert('L')
        elif "replica" in image_path:
            object_mask = Image.open(image_path.replace("images_all", "object_mask").replace(".jpg", ".png")).convert('L')
        elif "scannet" in image_path:
            object_mask = Image.open(image_path.replace("images", "object_mask").replace(".JPG", ".png")).convert('L')
        else:
            object_mask = Image.open(image_path.replace("images", "object_mask").replace(".jpg", ".png")).convert('L')
        object_mask = np.array(object_mask.resize(resolution), dtype=np.uint8)
        self.object_mask = torch.from_numpy(object_mask)

class MiniCam:
    def __init__(
        self,
        width,
        height,
        fovy,
        fovx,
        znear,
        zfar,
        world_view_transform,
        full_proj_transform,
    ):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
