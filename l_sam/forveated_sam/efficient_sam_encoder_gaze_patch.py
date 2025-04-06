# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List, Optional, Tuple, Type
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utility.watch import watch_time, Watch
from DynamicFocus_foveal_seg_classification.models.models_gaze_patch import build_net_compress, build_net_saliency, DeformSegmentationModule
from config import cfg
import cv2

def fillMissingValues_tensor(target_for_interp, copy=False, interp_mode='tri'):
    """
    fill missing values in a tenor

    input shape: [num_classes, h, w]
    output shape: [num_classes, h, w]
    """

    if copy:
        target_for_interp = target_for_interp.clone()

    def getPixelsForInterp(img):
        """
        Calculates a mask of pixels neighboring invalid values -
           to use for interpolation.

        input shape: [num_classes, h, w]
        output shape: [num_classes, h, w]
        """

        invalid_mask = torch.isnan(img)
        kernel = torch.tensor(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), device=invalid_mask.device).unsqueeze(0).unsqueeze(0).expand(1,invalid_mask.shape[0],3,3).float()

        #dilate to mark borders around invalid regions
        if max(invalid_mask.shape) > 512:
            dr = max(invalid_mask.shape)/512
            input = invalid_mask.float().unsqueeze(0)
            shape_ori = (invalid_mask.shape[-2], int(invalid_mask.shape[-1]))
            shape_scaled = (int(invalid_mask.shape[-2]/dr), int(invalid_mask.shape[-1]/dr))
            input_scaled = F.interpolate(input, shape_scaled, mode='nearest').squeeze(0)
            invalid_mask_scaled = input_scaled.unsqueeze(0) # b,c,w,h

            dilated_mask_scaled = torch.clamp(F.conv2d(invalid_mask_scaled, kernel, padding=(1, 1)), 0, 1)
            dilated_mask_scaled_t = dilated_mask_scaled.float()
            dilated_mask = F.interpolate(dilated_mask_scaled_t, shape_ori, mode='nearest').squeeze(0)
        else:

            dilated_mask = torch.clamp(F.conv2d(invalid_mask.float().unsqueeze(0),
                                                kernel, padding=(1, 1)), 0, 1).squeeze(0)

        # pixelwise "and" with valid pixel mask (~invalid_mask)
        masked_for_interp = dilated_mask *  (~invalid_mask).float()
        # Add 4 zeros corner points required for interp2d
        masked_for_interp[:,0,0] *= 0
        masked_for_interp[:,0,-1] *= 0
        masked_for_interp[:,-1,0] *= 0
        masked_for_interp[:,-1,-1] *= 0
        masked_for_interp[:,0,0] += 1
        masked_for_interp[:,0,-1] += 1
        masked_for_interp[:,-1,0] += 1
        masked_for_interp[:,-1,-1] += 1

        return masked_for_interp.bool(), invalid_mask


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
            self,
            img_size,
            patch_size,
            in_chans,
            embed_dim,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            bias=True,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        return x


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            qkv_bias,
            qk_scale=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class Mlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            act_layer=nn.GELU,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


@torch.jit.export
def get_abs_pos(
        abs_pos: torch.Tensor, has_cls_token: bool, hw: List[int]
) -> torch.Tensor:
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.

    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    h = hw[0]
    w = hw[1]
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )
        return new_abs_pos.permute(0, 2, 3, 1)
    else:
        return abs_pos.reshape(1, h, w, -1)


# Image encoder for efficient SAM.
class ImageEncoderViT(nn.Module):
    def __init__(
            self,
            img_size: int,
            patch_size: int,
            in_chans: int,
            patch_embed_dim: int,
            normalization_type: str,
            depth: int,
            num_heads: int,
            mlp_ratio: float,
            neck_dims: List[int],
            act_layer: Type[nn.Module],
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            patch_embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            act_layer (nn.Module): Activation layer.
        """
        super().__init__()

        self.img_size = img_size
        self.image_embedding_size = img_size // ((patch_size if patch_size > 0 else 1))
        self.transformer_output_dim = ([patch_embed_dim] + neck_dims)[-1]
        self.pretrain_use_cls_token = True
        pretrain_img_size = 224
        self.segmentation_module = DeformSegmentationModule(cfg)

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, patch_embed_dim)


        # Initialize absolute positional embedding with pretrain image size.
        num_patches = (pretrain_img_size // patch_size) * (
                pretrain_img_size // patch_size
        )
        num_positions = num_patches + 1
        self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, patch_embed_dim))
        self.blocks = nn.ModuleList()
        for i in range(depth):
            vit_block = Block(patch_embed_dim, num_heads, mlp_ratio, True)
            self.blocks.append(vit_block)
        self.neck = nn.Sequential(
            nn.Conv2d(
                patch_embed_dim,
                neck_dims[0],
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(neck_dims[0]),
            nn.Conv2d(
                neck_dims[0],
                neck_dims[0],
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(neck_dims[0]),
        )

    def forward(self, x: torch.Tensor, batched_points, s_bin_selected_BxHMxWMx1) -> torch.Tensor:
        assert (
                x.shape[2] == self.img_size and x.shape[3] == self.img_size
        ), "input image size must match self.img_size"
        # w = Watch()
        img_original = x
        x = self.patch_embed(x)
        # print(f"self.patch_embed = {w.see_seconds()}")

        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        x = x + get_abs_pos(
            self.pos_embed, self.pretrain_use_cls_token, [x.shape[1], x.shape[2]]
        )

        #####################################################################################################################################
        ################ learning to down sample and pruning ############################
        ## pdb.set_trace()
        #x = x.permute(0, 3, 1, 2)  # B H W C -> B C H W

        ## 新的代码：直接以注视点为中心裁剪出20x20的区域
        #batch_size, channels, height, width = x.shape
        #crop_size = 20  # 裁剪大小
        #
        ## 获取每个batch中注视点的坐标
        ## batched_points的维度为[B, 1, 1, 2]，最后一维分别是宽度坐标和高度坐标
        ## 注意：原始坐标是在原图(640x640)上的像素坐标，需要缩放到当前feature map尺寸
        ## 原图中坐标[x, y]表示第y行，第x列，原点在左上角
        #
        ## 将原始图像坐标(640x640)缩放到feature map尺寸(40x40)
        #img_size = 640  # 原始图像尺寸
        #gaze_x = (batched_points[:, 0, 0, 0].float() / img_size) * width
        #gaze_y = (batched_points[:, 0, 0, 1].float() / img_size) * height
        #
        ## 将坐标转为整数
        #gaze_y = gaze_y.round().long()
        #gaze_x = gaze_x.round().long()
        #
        ## 存储裁剪后的feature map
        #cropped_x = torch.zeros(batch_size, channels, crop_size, crop_size, device=x.device)
        #
        ## 对每个batch进行裁剪
        #for b in range(batch_size):
        #    # 计算裁剪区域的起始位置，优先尝试以注视点为中心
        #    # 但如果靠近边界，则调整位置确保获得完整的20x20区域
        #    # 并且确保注视点在裁剪区域内
        #    start_y = min(max(0, gaze_y[b] - crop_size // 2), height - crop_size)
        #    start_x = min(max(0, gaze_x[b] - crop_size // 2), width - crop_size)
        #    
        #    # 确保注视点在裁剪区域内
        #    if gaze_y[b] < start_y:
        #        start_y = max(0, gaze_y[b])
        #    if gaze_y[b] >= start_y + crop_size:
        #        start_y = max(0, min(gaze_y[b] - crop_size + 1, height - crop_size))
        #        
        #    if gaze_x[b] < start_x:
        #        start_x = max(0, gaze_x[b])
        #    if gaze_x[b] >= start_x + crop_size:
        #        start_x = max(0, min(gaze_x[b] - crop_size + 1, width - crop_size))
        #    
        #    # 进行裁剪 - 直接取20x20区域，不做填充或部分裁剪
        #    cropped_x[b] = x[b, :, start_y:start_y+crop_size, start_x:start_x+crop_size]
        #    
        ## 将裁剪后的feature map用于后续处理
        #x = cropped_x
        #
        #x = x.permute(0, 2, 3, 1)  # B C H W -> B H W C

        ####################################################################################################################################

        ####################################################################################################################################
        ############### learning to down sample and pruning ############################

        #pdb.set_trace()
        #import matplotlib.pyplot as plt
        #arr = s_bin_selected_BxHMxWMx1[0,0,:,:].cpu().numpy()
        #plt.imsave('../l_sam_experiment/selectmask.png', arr, cmap='gray')

        x = x.permute(0, 3, 1, 2)

        segSize = 20
        #x, grid = self.segmentation_module(x, batched_points, s_bin_selected_BxHMxWMx1, segSize)
        x, indices = self.segmentation_module(x, img_original, batched_points, s_bin_selected_BxHMxWMx1, segSize)

        x = x.permute(0, 2, 3, 1)

        ####################################################################################################################################

        num_patches = x.shape[1]
        assert x.shape[2] == num_patches
        x = x.reshape(x.shape[0], num_patches * num_patches, x.shape[3])

        # x = x.view(1, 1, 4096, 192)
        x = x[:,:,:]

        for blk in self.blocks:
            x = blk(x)
        x = x.reshape(x.shape[0], num_patches, num_patches, x.shape[2])

        # wang topk_reconstruct
        x = x.permute(0, 3, 1, 2)
        x_flat = x.view(x.shape[0], x.shape[1], -1)
        full_x = torch.zeros(x.shape[0], x.shape[1], 40*40).to(x.device)
        full_x.scatter_(2, indices.unsqueeze(1).expand(-1, x.shape[1], -1), x_flat)
        full_x = full_x.view(full_x.shape[0], -1, 40, 40)
        x = x.permute(0, 2, 3, 1)
        # wang topk_reconstruct

        #x = x.permute(0, 3, 1, 2)
        #full_x = self.segmentation_module.inverse_grid_sample(output=x, grid=grid, mode='nearest', kernel_size=1, sigma=1.0)
        #x = x.permute(0, 2, 3, 1)

        ## 新的代码：将20*20的feature map放回到40*40的tensor中，其余部分填充为0
        #x = x.permute(0, 3, 1, 2)  # B H W C -> B C H W
        #batch_size, channels, small_h, small_w = x.shape
        #full_h, full_w = 40, 40  # 恢复到原始大小
        #
        ## 创建填充为0的完整feature map
        #full_x = torch.zeros(batch_size, channels, full_h, full_w, device=x.device)
        #
        ## 对每个batch进行处理
        #for b in range(batch_size):
        #    # 计算在full_x中的位置
        #    # 尽量以注视点为中心放置，但优先确保完整放置20x20区域
        #    # 使用与裁剪相同的坐标转换逻辑
        #    img_size = 640  # 原始图像尺寸
        #    center_x = (batched_points[b, 0, 0, 0].float() / img_size) * full_w
        #    center_y = (batched_points[b, 0, 0, 1].float() / img_size) * full_h
        #    
        #    center_y = center_y.round().long()
        #    center_x = center_x.round().long()
        #    
        #    # 计算粘贴区域的起始位置，尽量以注视点为中心
        #    # 但如果靠近边界，则调整位置确保完整放置
        #    start_y = min(max(0, center_y - small_h // 2), full_h - small_h)
        #    start_x = min(max(0, center_x - small_w // 2), full_w - small_w)
        #    
        #    # 确保注视点在放置区域内
        #    if center_y < start_y:
        #        start_y = max(0, center_y)
        #    if center_y >= start_y + small_h:
        #        start_y = max(0, min(center_y - small_h + 1, full_h - small_h))
        #        
        #    if center_x < start_x:
        #        start_x = max(0, center_x)
        #    if center_x >= start_x + small_w:
        #        start_x = max(0, min(center_x - small_w + 1, full_w - small_w))
        #    
        #    # 将小feature map粘贴到完整feature map中
        #    full_x[b, :, start_y:start_y+small_h, start_x:start_x+small_w] = x[b]
        
        # 应用neck处理
        x = self.neck(full_x)
        
        return x
