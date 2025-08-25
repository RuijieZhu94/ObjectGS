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

from pathlib import Path
import os
import numpy as np

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')

from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir, render_object_masks_dir, gt_object_masks_dir):
    renders = []
    gts = []
    render_object_masks = []
    gt_object_masks = []

    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        render_object_mask = Image.open(render_object_masks_dir / fname)
        gt_object_mask = Image.open(gt_object_masks_dir / fname)
        render_image = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()
        render_mask = tf.to_tensor(render).unsqueeze(0)[:, 3:4, :, :].cuda()
        render_image = render_image * render_mask
        gt_image = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()
        gt_mask = tf.to_tensor(gt).unsqueeze(0)[:, 3:4, :, :].cuda()
        gt_image = gt_image * gt_mask
        render_object_mask = torch.from_numpy(np.array(render_object_mask, dtype=np.uint8)).unsqueeze(0).cuda()
        gt_object_mask = torch.from_numpy(np.array(gt_object_mask, dtype=np.uint8)).unsqueeze(0).cuda()
        renders.append(render_image)
        gts.append(gt_image)
        image_names.append(fname)
        render_object_masks.append(render_object_mask)
        gt_object_masks.append(gt_object_mask)

    return renders, gts, image_names, render_object_masks, gt_object_masks

# IoU, Dice, Pixel Accuracy
def calculate_segmentation_metrics(pred, gt, num_classes):
    ious = []
    dices = []
    pixel_accuracies = []

    # Flatten the prediction and ground truth
    assert pred.shape == gt.shape, "pred and gt should have the same shape"
    pred_flat = pred.view(-1)
    gt_flat = gt.view(-1)

    pred_flat[gt_flat == 0] = 0  # Ignore background class

    # Calculate Pixel Accuracy
    pixel_accuracy = (pred_flat == gt_flat).float().mean()
    pixel_accuracies.append(pixel_accuracy)

    class_ids = gt_flat.unique()
    # Calculate IoU and Dice for each class
    for class_id in class_ids:
        # For the current class, get the predicted and ground truth regions
        pred_class = (pred_flat == class_id).float()
        gt_class = (gt_flat == class_id).float()

        if gt_class.sum() < 100:
            ious.append(torch.tensor(float('nan')))
            dices.append(torch.tensor(float('nan')))
        else:
            # Calculate IoU
            intersection = (pred_class * gt_class).sum()
            union = pred_class.sum() + gt_class.sum() - intersection
            iou = intersection / (union + 1e-6)  # Prevent division by zero
            ious.append(iou)

            # Calculate Dice coefficient
            dice = 2.0 * intersection / (pred_class.sum() + gt_class.sum() + 1e-6)
            dices.append(dice)

    mean_iou = torch.tensor(ious).nanmean()  
    mean_dice = torch.tensor(dices).nanmean()  
    return mean_iou, mean_dice, pixel_accuracies



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
            render_object_masks_dir = method_dir / "semantic"
            gt_object_masks_dir = method_dir / "semantic_gt"
            renders, gts, image_names, render_object_masks, gt_object_masks = readImages(renders_dir, gt_dir, render_object_masks_dir, gt_object_masks_dir)

            ssims = []
            psnrs = []
            lpipss = []
            all_ious = []
            all_dices = []
            all_pixel_accuracies = []

            num_classes = 256  # Assume class IDs range from 1 to num_classes, 0 is background

            for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                ssims.append(ssim(renders[idx], gts[idx]))
                psnrs.append(psnr(renders[idx], gts[idx]))
                lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())

                ious, dices, pixel_accuracies = calculate_segmentation_metrics(render_object_masks[idx], gt_object_masks[idx], num_classes)
                all_ious.append(ious)
                all_dices.append(dices)
                all_pixel_accuracies.append(pixel_accuracies)

            # Calculate the average of segmentation metrics
            mean_ious = torch.tensor(all_ious)  
            mean_dices = torch.tensor(all_dices)
            mean_pixel_accuracies = torch.tensor(all_pixel_accuracies)

            # Log the results
            if logger:
                logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
                logger.info("  PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
                logger.info("  SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
                logger.info("  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))

                # Print segmentation metrics
                logger.info("  IoU : \033[1;35m{:>12.7f}\033[0m".format(mean_ious, ".5"))
                logger.info("  Dice : \033[1;35m{:>12.7f}\033[0m".format(mean_dices, ".5"))
                logger.info("  Pixel Accuracy: \033[1;35m{:>12.7f}\033[0m".format(mean_pixel_accuracies, ".5"))

                # Visible count (if available)
                logger.info("  GS_NUMS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(visible_count).float().mean(), ".5"))
            print("")

            # Update the dictionary
            full_dict[scene_dir][method].update({
                "PSNR": torch.tensor(psnrs).mean().item(),
                "SSIM": torch.tensor(ssims).mean().item(),
                "LPIPS": torch.tensor(lpipss).mean().item(),
                "IoU": mean_ious.nanmean().item(),  
                "Dice": mean_dices.nanmean().item(),  
                "Pixel Accuracy": mean_pixel_accuracies.nanmean().item(),
            })

            per_view_dict[scene_dir][method].update({
                "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                "SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                "IoU": {name: iou for iou, name in zip(mean_ious.tolist(), image_names)},  # IoU for each image
                "Dice": {name: dice for dice, name in zip(mean_dices.tolist(), image_names)},  # Dice for each image
                "Pixel Accuracy": {name: pixel_accuracy for pixel_accuracy, name in zip(mean_pixel_accuracies.tolist(), image_names)},  # Pixel Accuracy for each image
            })

    # Save the updated results
    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)

    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)


if __name__ == "__main__":
    lpips_fn = lpips.LPIPS(net='vgg').cuda()
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, type=str, default="")
    args = parser.parse_args()
    evaluate(args.model_paths, "test")
