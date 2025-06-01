import math
from typing import List, Optional, Tuple, Type
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utility.watch import watch_time, Watch
from l_sam.forveated_sam.models_saliency_encoder import DeformSegmentationModule
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

    def forward(self, x, padding_mask=None):
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

        if padding_mask is not None:
            padding_mask = padding_mask.view(B, 1, 1, N)
            attention_mask = padding_mask * padding_mask.transpose(-2, -1)
            attention_mask = attention_mask.expand(-1, self.num_heads, -1, -1)
            attn = attn.masked_fill(attention_mask == 0, -1e9)
        
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

    def forward(self, x, padding_mask=None):
        x = x + self.attn(self.norm1(x), padding_mask)
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
            cut_ratio: float,
            min_tokens: int,
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
        self.cut_ratio = cut_ratio
        self.min_tokens = min_tokens
        self.img_size = img_size
        self.image_embedding_size = img_size // ((patch_size if patch_size > 0 else 1))
        self.transformer_output_dim = ([patch_embed_dim] + neck_dims)[-1]
        self.pretrain_use_cls_token = True
        pretrain_img_size = 224
        self.segmentation_module = DeformSegmentationModule(cfg)

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, patch_embed_dim)

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

    def forward(self, x: torch.Tensor, batched_points, Y_bx1xHxW, Y_cls_bx1) -> torch.Tensor:
        assert (
                x.shape[2] == self.img_size and x.shape[3] == self.img_size
        ), "input image size must match self.img_size"
        img_original = x
        x = self.patch_embed(x)
        _, _, ts, ts = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x + get_abs_pos(
            self.pos_embed, self.pretrain_use_cls_token, [x.shape[1], x.shape[2]]
        )

        B, C, H, W = img_original.shape
        avg_ds_size = 160
        kernel_size = (H // avg_ds_size, W // avg_ds_size)
        img_avg_ds = F.avg_pool2d(img_original, kernel_size=kernel_size)
        img_avg_ds = F.interpolate(img_avg_ds, size=(avg_ds_size, avg_ds_size), mode='bilinear', align_corners=False)
        x_ds = self.patch_embed(img_avg_ds)
        x_ds = x_ds.permute(0, 2, 3, 1)
        x_ds = x_ds + get_abs_pos(
            self.pos_embed, self.pretrain_use_cls_token, [x_ds.shape[1], x_ds.shape[2]]
        )
        x_ds = x_ds.permute(0, 3, 1, 2)

        x = x.permute(0, 3, 1, 2)

        x, indices, loss, loss_nll = self.segmentation_module(x, 
                                                        x_ds, 
                                                        batched_points, 
                                                        Y_bx1xHxW,
                                                        Y_cls_bx1,
                                                        cut_ratio=self.cut_ratio, 
                                                        min_tokens=self.min_tokens)

        x = x.permute(0, 2, 1)

        for blk in self.blocks:
            x = blk(x, None)

        x = x.permute(0, 2, 1)
        x_flat = x.view(x.shape[0], x.shape[1], -1)
        full_x = torch.zeros(x.shape[0], x.shape[1], ts*ts).to(x.device)
        
        valid_indices_mask = indices >= 0
        
        for b in range(indices.shape[0]):
            valid_mask = valid_indices_mask[b]
            valid_indices = indices[b, valid_mask]
            
            if valid_indices.numel() > 0:
                valid_features = x_flat[b, :, valid_mask]
                valid_indices_exp = valid_indices.unsqueeze(0).expand(x.shape[1], -1)
                full_x[b].scatter_(1, valid_indices_exp, valid_features)
        
        full_x = full_x.view(full_x.shape[0], -1, ts, ts)
        x = self.neck(full_x)
        
        return x, loss, loss_nll
