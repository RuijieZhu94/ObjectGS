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
import sys
import imageio
import yaml
from os import makedirs
import torch
import numpy as np

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')

from scene import Scene
import json
import time
import torchvision
from tqdm import tqdm
from utils.general_utils import safe_state, parse_cfg, visualize_depth, visualize_normal
from utils.image_utils import save_rgba
from argparse import ArgumentParser

def render_set(model_path, name, iteration, views, gaussians, pipe, background, query_label_id):
    vis_normal=False
    vis_depth=False
    if gaussians.gs_attr == "2D":
        vis_normal=True
        vis_depth=True
        

    render_path = os.path.join(model_path, name, "id_{}".format(query_label_id), "renders")
    error_path = os.path.join(model_path, name, "id_{}".format(query_label_id), "errors")
    gts_path = os.path.join(model_path, name, "id_{}".format(query_label_id), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    if vis_normal:
        normal_path = os.path.join(model_path, name, "id_{}".format(query_label_id), "normal")
        makedirs(normal_path, exist_ok=True)
    if vis_depth:
        depth_path = os.path.join(model_path, name, "id_{}".format(query_label_id), "depth")
        makedirs(depth_path, exist_ok=True)

    # specify label id    
    object_mask = (gaussians.label_ids.squeeze() == query_label_id)

    modules = __import__('gaussian_renderer')
    
    t_list = []
    visible_count_list = []
    per_view_dict = {}
    for idx, view in enumerate(tqdm(views, desc="rendering progress")):
        
        # if idx > 30:
        #     continue
        # visible mask
        if gaussians.explicit_gs:
            gaussians.set_gs_mask(view.camera_center, view.resolution_scale)
            visible_mask = gaussians._gs_mask
        else:
            gaussians.set_anchor_mask(view.camera_center, view.resolution_scale)
            from gaussian_renderer.render import prefilter_voxel
            visible_mask = prefilter_voxel(view, gaussians).squeeze() if pipe.add_prefilter else gaussians._anchor_mask    


        # render function
        torch.cuda.synchronize();t_start = time.time()
        render_pkg = getattr(modules, 'render')(view, gaussians, pipe, background, visible_mask=visible_mask, object_mask=object_mask)
        torch.cuda.synchronize();t_end = time.time()

        t_list.append(t_end - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = render_pkg["visibility_filter"].sum()

        # gts
        gt = view.original_image.cuda()
        alpha_mask = view.alpha_mask.cuda()
        # alpha_mask = rendering.any(dim=0, keepdim=True)
        rendering = torch.cat([rendering, alpha_mask], dim=0)
        gt = torch.cat([gt, alpha_mask], dim=0)
        
        # error maps
        if gt.device != rendering.device:
            rendering = rendering.to(gt.device)
        # errormap = (rendering - gt).abs()
        
        # if vis_normal == True:
            # normal_map = render_pkg['render_normals'][0] 
            # vis_normal_map = visualize_normal(normal_map, view)
            # vis_alpha_mask = ((alpha_mask * 255).byte()).permute(1, 2, 0).cpu().numpy()
            # vis_normal_map = np.concatenate((vis_normal_map,vis_alpha_mask),axis=2)
            # imageio.imwrite(os.path.join(normal_path, '{0:05d}'.format(idx) + ".png"), vis_normal_map)
            
        # if vis_depth == True:
        #     depth_map = render_pkg["render_depth"]
        #     vis_depth_map = visualize_depth(depth_map) 
            # vis_depth_map = torch.concat([vis_depth_map,alpha_mask],dim=0)
            # torchvision.utils.save_image(vis_depth_map, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))

        save_rgba(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # save_rgba(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + ".png"))
        # save_rgba(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        visible_count_list.append(visible_count)
        per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()


    # print((len(aerial_t_list)-5+len(street_t_list)-5)/( sum(street_t_list[5:]) + sum(aerial_t_list[5:])))

    
def render_sets(dataset, opt, pipe, iteration, skip_train, skip_test, ape_code, explicit, query_label_id):
    with torch.no_grad():
        if pipe.no_prefilter_step > 0:
            pipe.add_prefilter = False
        else:
            pipe.add_prefilter = True
        modules = __import__('scene')
        model_config = dataset.model_config
        model_config['kwargs']['ape_code'] = ape_code
        gaussians = getattr(modules, model_config['name'])(**model_config['kwargs'])
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, explicit=explicit)
        gaussians.eval()

        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)
        
        scene.background = torch.ones_like(scene.background).cuda()

        if query_label_id == -1:
            for label in gaussians.label_ids.unique():
                if not skip_train:
                    render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipe, scene.background, label)

                if not skip_test:
                    render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipe, scene.background, label)
        else:
            if not skip_train:
                render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipe, scene.background, query_label_id)

            if not skip_test:
                render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipe, scene.background, query_label_id)
            

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument('-m', '--model_path', type=str, required=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--ape", default=-1, type=int)
    parser.add_argument("--query_label_id", default=-1, type=int)
    parser.add_argument("--scene_name", default=None, type=str)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--explicit", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    args.scene_name = args.model_path.split('/')[-2]

    with open(os.path.join(args.model_path, "config.yaml")) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        try:
            cfg["model_params"]["exp_name"] = os.path.join(cfg["model_params"]["exp_name"], args.scene_name)
            cfg["model_params"]["source_path"] = os.path.join(cfg["model_params"]["source_path"], args.scene_name)
        except:
            print("OverrideError: Cannot override 'exp_name' and 'source_path' in 'model_params'. Exiting.")
            sys.exit(1)
            
    lp, op, pp = parse_cfg(cfg)
    lp.model_path = args.model_path
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(lp, op, pp, args.iteration, args.skip_train, args.skip_test, args.ape, args.explicit, args.query_label_id)
