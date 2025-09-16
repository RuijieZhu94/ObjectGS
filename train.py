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
import shutil
import numpy as np

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')

import torch
import torchvision
import json
import wandb
import time
from datetime import datetime
from os import makedirs
import shutil
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
import lpips
from random import randint
from utils.loss_utils import l1_loss, ssim
import sys
from gaussian_renderer import network_gui
from scene import Scene
from utils.general_utils import get_expon_lr_func, safe_state, parse_cfg, visualize_depth
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, save_rgba
from argparse import ArgumentParser, Namespace
import yaml
import torch.nn.functional as F
import warnings
from render import render_sets
warnings.filterwarnings('ignore')

lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")

# Class to convert object IDs to RGB colors
class ID2RGBConverter:
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.color_map = self._generate_color_map()

    def _generate_color_map(self):
        # Generate 256 random RGB colors, each color is a tuple (0-255 range)
        return np.random.randint(0, 256, size=(256, 3), dtype=np.uint8)

    def convert(self, obj: int):
        if obj == 0:
            return 0, np.array([0, 0, 0], dtype=np.uint8)  # Predefine class 0 as black
        if 0 <= obj <= 255:
            return obj, self.color_map[obj]  # Get color from the fixed color map
        else:
            raise ValueError("ID out of range, should be between 0 and 255")

def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    assert os.path.exists(os.path.join(ROOT, '.gitignore'))
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = Path(__file__).resolve().parent

    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))
    
    print('Backup Finished!')

def training(dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, wandb=None, logger=None, ply_path=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    
    modules = __import__('scene')
    model_config = dataset.model_config
    gaussians = getattr(modules, model_config['name'])(**model_config['kwargs'])
    scene = Scene(dataset, gaussians, shuffle=pipe.shuffle, logger=logger, weed_ratio=pipe.weed_ratio)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)
    
    viewpoint_stack = None
    
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    densify_cnt = 0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    modules = __import__('gaussian_renderer')
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.add_prefilter, keep_alive = network_gui.receive()
                if custom_cam != None:
                    net_image = getattr(modules, 'render')(custom_cam, gaussians, pipe, scene.background)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        
        # Pick a random Camera
        
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        if gaussians.explicit_gs:
            gaussians.set_gs_mask(viewpoint_cam.camera_center, viewpoint_cam.resolution_scale)
            visible_mask = gaussians._gs_mask
        else:
            gaussians.set_anchor_mask(viewpoint_cam.camera_center, viewpoint_cam.resolution_scale)
            from gaussian_renderer.render import prefilter_voxel
            visible_mask = prefilter_voxel(viewpoint_cam, gaussians).squeeze() if pipe.add_prefilter else gaussians._anchor_mask    

        render_pkg = getattr(modules, 'render')(viewpoint_cam, gaussians, pipe, scene.background, visible_mask)
        image, scaling, alpha, semantics = render_pkg["render"], render_pkg["scaling"], render_pkg["render_alphas"], render_pkg["render_semantics"]

        gt_image = viewpoint_cam.original_image.cuda()
        alpha_mask = viewpoint_cam.alpha_mask.cuda()
        image = image * alpha_mask
        gt_image = gt_image * alpha_mask

        losses = dict()

        # Photometric loss 
        Ll1 = l1_loss(image, gt_image)
        ssim_loss = (1.0 - ssim(image, gt_image))
        losses["image_loss"] = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss

        # Scaling loss
        if opt.lambda_dreg > 0:
            if scaling.shape[0] > 0:
                scaling_reg = scaling.prod(dim=1).mean()
            else:
                scaling_reg = torch.tensor(0.0, device="cuda")
            losses["scaling_loss"] = opt.lambda_dreg * scaling_reg

        # Object loss
        if opt.lambda_object_loss > 0:
            # object_loss = 0.0
            # for label in gaussians.label_ids.unique():
            #     if label > 0:
            #         object_mask = (gaussians.label_ids.squeeze() == label) & visible_mask
            #         object_render_pkg = getattr(modules, 'render')(viewpoint_cam, gaussians, pipe, scene.background, visible_mask=object_mask)
            #         render_image, render_alpha = object_render_pkg["render"], object_render_pkg["render_alphas"]
            #         gt_object_mask = (viewpoint_cam.object_mask.cuda() == label).expand_as(gt_image)
            #         if gt_object_mask.sum() > 0:
            #             object_loss += l1_loss(gt_image[gt_object_mask], render_image[gt_object_mask])
            #             object_loss += 1.0 * l1_loss(gt_object_mask.float(), render_alpha)
            # losses["object_loss"] = opt.lambda_object_loss * object_loss
            gt_object_ids = gaussians.id_encoder.label_to_index(viewpoint_cam.object_mask.cuda()).long()
            object_loss = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='mean')(semantics.permute(0,3,1,2), gt_object_ids.unsqueeze(0))
            # object_loss = torch.nn.CrossEntropyLoss(reduction='mean')(semantics.permute(0,3,1,2), gt_object_ids.unsqueeze(0))
            losses["object_loss"] = opt.lambda_object_loss * object_loss

            prob_zero_class = semantics[..., 0]  
            losses["zero_penalty"] = opt.lambda_zero_penalty * prob_zero_class.mean()

        # Sky opacity loss
        if opt.lambda_sky_opa > 0:
            o = alpha.clamp(1e-6, 1-1e-6)
            sky = alpha_mask.float()
            loss_sky_opa = (-(1-sky) * torch.log(1 - o)).mean()
            losses["sky_opa_loss"] =  opt.lambda_sky_opa * loss_sky_opa

        # Opacity entropy loss
        if opt.lambda_opacity_entropy > 0:
            o = alpha.clamp(1e-6, 1 - 1e-6)
            loss_opacity_entropy = -(o*torch.log(o)).mean()
            losses["opacity_entropy_loss"] = opt.lambda_opacity_entropy * loss_opacity_entropy

        # Normal loss
        if opt.lambda_normal > 0 and iteration > opt.normal_start_iter:
            assert gaussians.render_mode=="RGB+ED" or gaussians.render_mode=="RGB+D"
            normals = render_pkg["render_normals"].squeeze(0).permute((2, 0, 1))
            normals_from_depth = render_pkg["render_normals_from_depth"] * render_pkg["render_alphas"].permute((1, 2, 0)).detach()
            if len(normals_from_depth.shape) == 4:
                normals_from_depth = normals_from_depth.squeeze(0)
            normals_from_depth = normals_from_depth.permute((2, 0, 1))
            normal_error = (1 - (normals * normals_from_depth).sum(dim=0))[None]
            losses["normal_loss"] = opt.lambda_normal * (normal_error * alpha_mask).mean()

        # Distortion loss
        if opt.lambda_dist and iteration > opt.dist_start_iter:
            losses["distort_loss"] = opt.lambda_dist * (render_pkg["render_distort"].squeeze(3) * alpha_mask).mean()
        
        # Depth loss
        if iteration > opt.start_depth and depth_l1_weight(iteration) > 0 and viewpoint_cam.invdepthmap is not None:
            assert gaussians.render_mode=="RGB+ED" or gaussians.render_mode=="RGB+D"
            render_depth = render_pkg["render_depth"]
            invDepth = torch.where(render_depth > 0.0, 1.0 / render_depth, torch.zeros_like(render_depth))            
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()
            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            losses["depth_loss"] = Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0
    
        total_loss = sum(losses.values())
        total_loss.backward()
        
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * total_loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                psnr_log = psnr(image, gt_image).mean().double()
                anchor_prim = len(gaussians.get_anchor)
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}","Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}","psnr":f"{psnr_log:.{3}f}","GS_num":f"{anchor_prim}","prefilter":f"{pipe.add_prefilter}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, dataset_name, iteration, losses, total_loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, getattr(modules, 'render'), (pipe, scene.background), wandb, logger)
            
            if (iteration in saving_iterations):
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration % pipe.vis_step == 0 or iteration == 1 or (iteration % 100 == 0 and iteration < 1000):
                viewpoint_cam = scene.getTrainCameras().copy()[10]

                if gaussians.explicit_gs:
                    gaussians.set_gs_mask(viewpoint_cam.camera_center, viewpoint_cam.resolution_scale)
                    visible_mask = gaussians._gs_mask
                else:
                    gaussians.set_anchor_mask(viewpoint_cam.camera_center, viewpoint_cam.resolution_scale)
                    from gaussian_renderer.render import prefilter_voxel
                    visible_mask = prefilter_voxel(viewpoint_cam, gaussians).squeeze() if pipe.add_prefilter else gaussians._anchor_mask    

                vis_render_pkg = getattr(modules, 'render')(viewpoint_cam, gaussians, pipe, scene.background, visible_mask)
                vis_image, alpha = vis_render_pkg["render"], vis_render_pkg["render_alphas"]
                gt_image = viewpoint_cam.original_image.cuda()
                alpha_mask = viewpoint_cam.alpha_mask.cuda()
                vis_image = vis_image * alpha_mask
                gt_image = gt_image * alpha_mask         

                other_img = []
                resolution = (int(viewpoint_cam.image_width/1.0), int(viewpoint_cam.image_height/1.0))
                vis_img = F.interpolate(vis_image.unsqueeze(0), size=(resolution[1], resolution[0]), mode='bilinear', align_corners=False)[0]
                # vis_gt_img = F.interpolate(gt_image.unsqueeze(0), size=(resolution[1], resolution[0]), mode='bilinear', align_corners=False)[0]
                # vis_alpha = F.interpolate(alpha.repeat(3, 1, 1).unsqueeze(0), size=(resolution[1], resolution[0]), mode='bilinear', align_corners=False)[0]


                if iteration > opt.start_depth and viewpoint_cam.invdepthmap is not None:
                    vis_depth = visualize_depth(invDepth) 
                    gt_depth = visualize_depth(mono_invdepth)
                    vis_depth = F.interpolate(vis_depth.unsqueeze(0), size=(resolution[1], resolution[0]), mode='bilinear', align_corners=False)[0]
                    vis_gt_depth = F.interpolate(gt_depth.unsqueeze(0), size=(resolution[1], resolution[0]), mode='bilinear', align_corners=False)[0]
                    other_img.append(vis_depth)
                    other_img.append(vis_gt_depth)
                
                grid = torchvision.utils.make_grid([
                    vis_img, 
                    # vis_gt_img, 
                    # vis_alpha,
                ] + other_img, nrow=1)

                vis_path = os.path.join(scene.model_path, "vis")
                os.makedirs(vis_path, exist_ok=True)
                torchvision.utils.save_image(grid, os.path.join(vis_path, f"{iteration:05d}_{viewpoint_cam.colmap_id:03d}.png"))
            
            # densification
            if iteration < opt.update_until and iteration > opt.start_stat:
                # add statis
                gaussians.training_statis(opt, render_pkg, image.shape[2], image.shape[1])
                densify_cnt += 1 

                # densification
                if opt.densification and iteration > opt.update_from and densify_cnt > 0 and densify_cnt % opt.update_interval == 0:
                    if dataset.pretrained_checkpoint != "":
                        gaussians.roll_back()
                    gaussians.run_densify(opt, iteration)
            
            elif iteration == opt.update_until:
                if dataset.pretrained_checkpoint != "":
                    gaussians.roll_back()
                gaussians.clean()
                    
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if iteration >= opt.iterations - pipe.no_prefilter_step:
                pipe.add_prefilter = False

            if (iteration in checkpoint_iterations):
                logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, dataset_name, iteration, losses, total_loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, wandb=None, logger=None):
    if tb_writer:
        for key, value in losses.items():
            tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/{key}', value, iteration)
        tb_writer.add_scalar(f'{dataset_name}/total_loss', total_loss.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)

    if wandb is not None:
        wandb.log({f"{dataset_name}_loss_patches_{key}":value.item() for key, value in losses.items()})
        wandb.log({f"{dataset_name}_total_loss":total_loss.item()})
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        scene.gaussians.eval()
        torch.cuda.empty_cache()
        
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                            {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx] for idx in range(0, len(scene.getTrainCameras()), 100)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                cnt = 0
                
                if wandb is not None:
                    gt_image_list = []
                    render_image_list = []
                    errormap_list = []

                for idx, viewpoint in enumerate(config['cameras']):
                    
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    alpha_mask = viewpoint.alpha_mask.cuda()
                    image = image * alpha_mask
                    gt_image = gt_image * alpha_mask
                    
                    if tb_writer and (idx < 30):
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/errormap".format(viewpoint.image_name), (gt_image[None]-image[None]).abs(), global_step=iteration)

                        if wandb:
                            render_image_list.append(image[None])
                            errormap_list.append((gt_image[None]-image[None]).abs())
                            
                    if iteration == testing_iterations[0]:
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                        if wandb:
                            gt_image_list.append(gt_image[None])

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    cnt += 1 
                
                l1_test /= cnt
                psnr_test /= cnt    
                logger.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
         
        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/'+'total_points', len(scene.gaussians.get_anchor), iteration)
        torch.cuda.empty_cache()

        scene.gaussians.train()

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    masks = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        render_image = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()
        render_mask = tf.to_tensor(render).unsqueeze(0)[:, 3:4, :, :].cuda()
        render_image = render_image * render_mask
        gt_image = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()
        gt_mask = tf.to_tensor(gt).unsqueeze(0)[:, 3:4, :, :].cuda()
        gt_image = gt_image * gt_mask
        renders.append(render_image)
        gts.append(gt_image)
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths, eval_name, visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    
    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / eval_name

    for method in os.listdir(test_dir):

        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        base_method_dir = test_dir / method
        method_dir = base_method_dir 
        if os.path.exists(method_dir):
            gt_dir = method_dir/ "gt"
            renders_dir = method_dir / "renders"
            renders, gts, image_names = readImages(renders_dir, gt_dir)

            ssims = []
            psnrs = []
            lpipss = []

            for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                ssims.append(ssim(renders[idx], gts[idx]))
                psnrs.append(psnr(renders[idx], gts[idx]))
                lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())

            logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
            logger.info("  PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
            logger.info("  SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
            logger.info("  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
            logger.info("  GS_NUMS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(visible_count).float().mean(), ".5"))
            print("")
            
            full_dict[scene_dir][method].update({
                "PSNR": torch.tensor(psnrs).mean().item(),
                "SSIM": torch.tensor(ssims).mean().item(),
                "LPIPS": torch.tensor(lpipss).mean().item(),
                "GS_NUMS": torch.tensor(visible_count).float().mean().item(),
                })

            per_view_dict[scene_dir][method].update({
                "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                "SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                "GS_NUMS": {name: vc for vc, name in zip(torch.tensor(visible_count).tolist(), image_names)}
                })

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)
    
def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO) 
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--config', type=str, help='train config file path')
    parser.add_argument('--scene_name', type=str, help='Override scene name in config', default=None)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[80000,90000,100000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[80000,90000,100000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[-1])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[-1])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gpu", type=str, default = '-1')
    args = parser.parse_args(sys.argv[1:])
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    if args.scene_name is not None:
        try:
            cfg["model_params"]["exp_name"] = os.path.join(cfg["model_params"]["exp_name"], args.scene_name)
            cfg["model_params"]["source_path"] = os.path.join(cfg["model_params"]["source_path"], args.scene_name)
        except:
            print("OverrideError: Cannot override 'exp_name' and 'source_path' in 'model_params'. Exiting.")
            sys.exit(1)

    lp, op, pp = parse_cfg(cfg)
    args.save_iterations.append(op.iterations)

    # enable logging
    cur_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    lp.model_path = os.path.join("outputs", lp.dataset_name, lp.exp_name, cur_time)
    os.makedirs(lp.model_path, exist_ok=True)
    shutil.copy(args.config, os.path.join(lp.model_path, "config.yaml"))

    logger = get_logger(lp.model_path)

    if args.test_iterations[0] == -1:
        args.test_iterations = [i for i in range(10000, op.iterations + 1, 10000)]
        # args.test_iterations = [i for i in range(5000, op.iterations + 1, 5000)]
    if len(args.test_iterations) == 0 or args.test_iterations[-1] != op.iterations:
        args.test_iterations.append(op.iterations)

    if args.save_iterations[0] == -1:
        args.save_iterations = [i for i in range(10000, op.iterations + 1, 10000)]
        # args.save_iterations = [i for i in range(5000, op.iterations + 1, 5000)]
    if len(args.save_iterations) == 0 or args.save_iterations[-1] != op.iterations:
        args.save_iterations.append(op.iterations)

    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.system("echo $CUDA_VISIBLE_DEVICES")
        logger.info(f'using GPU {args.gpu}')

    # try:
    #     saveRuntimeCode(os.path.join(lp.model_path, 'backup'))
    # except:
    #     logger.info(f'save code failed~')
    
    exp_name = lp.exp_name if lp.dataset_name=="" else lp.dataset_name+"_"+lp.exp_name
    if args.use_wandb:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"Horizon-GS",
            name=exp_name,
            # Track hyperparameters and run metadata
            settings=wandb.Settings(start_method="fork"),
            config=vars(args)
        )
    else:
        wandb = None
    
    logger.info("Optimizing " + lp.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training(lp, op, pp, exp_name, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, wandb, logger)

    # All done
    logger.info("\nTraining complete.")

    # rendering
    logger.info(f'\nStarting Rendering~')
    if lp.eval:
        visible_count = render_sets(lp, op, pp, -1, skip_train=True, skip_test=False, logger=logger)
    else:
        visible_count = render_sets(lp, op, pp, -1, skip_train=False, skip_test=True, logger=logger)
    logger.info("\nRendering complete.")

    # calc metrics
    logger.info("\n Starting evaluation...")
    eval_name = 'test' if lp.eval else 'train'
    evaluate(lp.model_path, eval_name, visible_count=visible_count, wandb=wandb, logger=logger)
    logger.info("\nEvaluating complete.")
