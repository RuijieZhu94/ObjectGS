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

import torch
from PIL import Image

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def save_rgba(img, save_path):
    img = (img * 255).byte()
    img = img.permute(1, 2, 0).detach().cpu().numpy()
    img = Image.fromarray(img, 'RGBA')
    img.save(save_path)

def save_mask(mask, save_path):
    mask = mask.detach().cpu().numpy() 
    mask_image = Image.fromarray(mask, 'L')  
    mask_image.save(save_path)
