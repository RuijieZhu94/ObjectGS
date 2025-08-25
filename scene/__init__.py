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
import random
import json
import torch
from scene.base_model import GaussianModel
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks, storePly
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.graphics_utils import BasicPointCloud
from utils.semantic_utils import OneHotEncoder
import numpy as np


class Scene:

    def __init__(self, args, gaussians, load_iteration=None, shuffle=True, logger=None, 
                 opti_test_oppe = False, explicit = False, weed_ratio=0.):
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.resolution_scales = args.resolution_scales
        self.loaded_iter = None
        self.gaussians = gaussians
        self.opti_test_oppe = opti_test_oppe
        self.gaussians.explicit_gs = explicit
        self.gaussians.weed_ratio = weed_ratio

        if args.random_background:
            self.background = torch.rand(3, dtype=torch.float32, device="cuda")
        elif args.white_background:
            self.background = torch.ones(3, dtype=torch.float32, device="cuda")
        else:
            self.background = torch.zeros(3, dtype=torch.float32, device="cuda")
        
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
                
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        
        if args.data_format == 'blender':
            print("Use Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path, args.eval, args.add_mask, args.add_depth, args.center, args.scale
            )
        elif args.data_format == 'colmap':
            print("Use Colmap data set!")
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path, args.eval, args.images, args.depths, args.masks, args.add_mask, args.add_depth, args.llffhold
            )
        elif args.data_format == 'city':
            print("Use City data set!")
            scene_info = sceneLoadTypeCallbacks["City"](
                args.source_path, args.eval, args.add_mask, args.add_depth, args.center, args.scale, args.ds, args.llffhold,
                args.depth_type
            )
        else:
            assert False, "Could not recognize scene type!"
            
        if not self.loaded_iter:
            logger.info("Train cameras: {}".format(len(scene_info.train_cameras)))
            logger.info("Test cameras: {}".format(len(scene_info.test_cameras)))
            pcd = self.save_ply(scene_info.point_cloud, args.ratio, os.path.join(self.model_path, "input.ply"))
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
        
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.gaussians.set_appearance(len(scene_info.train_cameras))

        for resolution_scale in self.resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, self.background)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, self.background)
        
        if weed_ratio > 0.:
            self.gaussians.cam_infos = torch.empty(0, 4).float().cuda()
            for cam in self.getTrainCameras():
                cam_info = torch.tensor([cam.camera_center[0], cam.camera_center[1], cam.camera_center[2], cam.resolution_scale]).float().cuda()
                self.gaussians.cam_infos = torch.cat((self.gaussians.cam_infos, cam_info.unsqueeze(dim=0)), dim=0)
            
        if self.loaded_iter:
            if self.opti_test_oppe:
                self.train_cameras = self.test_cameras
                scene_info = scene_info._replace(train_cameras=scene_info.test_cameras)
                self.gaussians.set_appearance(len(scene_info.train_cameras))
                self.gaussians.create_for_opti_appe(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter)), self.cameras_extent) # only optimize test appearance for correct evalution
            elif explicit:
                self.gaussians.load_explicit(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            "point_cloud_explicit.ply"))
            else:
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            "point_cloud.ply"))
                self.gaussians.load_mlp_checkpoints(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter)))
                self.gaussians.id_encoder = OneHotEncoder(self.gaussians.label_ids)
        else:
            if args.pretrained_checkpoint != "":
                self.gaussians.create_from_pretrained(pcd, self.cameras_extent, args.pretrained_checkpoint, logger)
            else:
                self.gaussians.create_from_pcd(pcd, self.cameras_extent, args.global_appearance, logger)
        
    def save_ply(self, pcd, ratio, path):
        new_points = pcd.points[::ratio]
        new_colors = pcd.colors[::ratio]
        new_normals = pcd.normals[::ratio]
        new_label_ids = pcd.label_ids[::ratio]
        new_pcd = BasicPointCloud(points=new_points, colors=new_colors, normals=new_normals, label_ids=new_label_ids)
        storePly(path, new_points, new_colors, new_label_ids)
        return new_pcd

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_mlp_checkpoints(point_cloud_path)
        if not self.gaussians.color_attr.startswith("SH"):
            print("Neural Gaussians do not have the SH property.")
        elif self.gaussians.view_dim != 0:
            print("Neural Gaussians are affected by viewpoint.")
        else:
            self.gaussians.save_explicit(os.path.join(point_cloud_path, "point_cloud_explicit.ply"))
            
    def getTrainCameras(self):
        all_cams = []   
        for scale in self.resolution_scales:
            all_cams.extend(self.train_cameras[scale])
        return all_cams

    def getTestCameras(self):
        all_cams = []   
        for scale in self.resolution_scales:
            all_cams.extend(self.test_cameras[scale])
        return all_cams