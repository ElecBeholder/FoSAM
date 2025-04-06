# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


def average_pool(input_tensor, kernel_size, stride=None, padding=0):
    """
    Perform average pooling on a given tensor.

    Args:
        input_tensor (torch.Tensor): The input tensor of shape (N, C, H, W).
        kernel_size (int or tuple): The size of the pooling window.
        stride (int or tuple, optional): The stride of the pooling window. Defaults to kernel_size.
        padding (int or tuple, optional): Implicit zero padding to be added on both sides. Defaults to 0.

    Returns:
        torch.Tensor: The output tensor after average pooling.
    """
    if stride is None:
        stride = kernel_size  # Default stride equals kernel_size

    # Perform average pooling using PyTorch's functional API
    output_tensor = nn.functional.avg_pool2d(input_tensor, kernel_size, stride, padding)
    return output_tensor

def cross_unfold(x_BxHTxWTxD: torch.Tensor):
    x_BxHMxWMxDxKxK = x_BxHTxWTxD.unfold(1, 2, 2).unfold(2, 2, 2)

    return x_BxHMxWMxDxKxK


def cross_fold(x_BxHMxWMxDxKxK: torch.Tensor):
    x_BxHMxKxWMxKxD = x_BxHMxWMxDxKxK.permute(0, 1, -2, 2, -1, 3)
    x_BxHTxWTxD = x_BxHMxKxWMxKxD.flatten(start_dim=3, end_dim=4).flatten(start_dim=1, end_dim=2)

    return x_BxHTxWTxD


def get_merge_map_edge(s_BxHTxWTx1: torch.Tensor):
    s_BxHMxWMx1xKxK = cross_unfold(s_BxHTxWTx1)
    s_P1_BxHMxWMx1x1x1 = torch.sum(s_BxHMxWMx1xKxK, dim=[-2, -1]) / 4
    s_P0_BxHMxWMx1x1x1 = 1 - s_P1_BxHMxWMx1x1x1

    s_entropy_BxHMxWMx1 = - (s_P1_BxHMxWMx1x1x1 * torch.log2(s_P1_BxHMxWMx1x1x1) + s_P0_BxHMxWMx1x1x1 * torch.log2(s_P0_BxHMxWMx1x1x1))

    s_entropy_BxHMxWMx1 = torch.nan_to_num(s_entropy_BxHMxWMx1, 0)
    return s_entropy_BxHMxWMx1


def get_merge_map_object(s_BxHTxWTx1: torch.Tensor):
    s_BxHMxWMx1xKxK = cross_unfold(s_BxHTxWTx1)
    s_mean_BxHMxWMx1 = torch.amax(s_BxHMxWMx1xKxK, dim=[-2, -1])
    return s_mean_BxHMxWMx1


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

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, patch_embed_dim)

        self.padded_embed_1xD = nn.Parameter(torch.zeros(1, patch_embed_dim))

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

    def forward(self, x_BxCxHxW: torch.Tensor, s_bin_selected_BxHMxWMx1: torch.Tensor = None) -> torch.Tensor:
        assert (x_BxCxHxW.shape[2] == self.img_size and x_BxCxHxW.shape[3] == self.img_size), "input image size must match self.img_size"

        x_BxDxHTxWT = self.patch_embed(x_BxCxHxW)

        # B C H W -> B H W C
        x_BxHTxWTxD = x_BxDxHTxWT.permute(0, 2, 3, 1)
        x_BxHTxWTxD = x_BxHTxWTxD + get_abs_pos(self.pos_embed, self.pretrain_use_cls_token, [x_BxHTxWTxD.shape[1], x_BxHTxWTxD.shape[2]])

        B, HT, WT, D = x_BxHTxWTxD.shape
        HM, WM = HT // 2, WT // 2
        s_bin_selected_BxHMxWMx1 = torch.randint(0, 2, (B, HM, WM, 1)).to(dtype=torch.bool,device=x_BxCxHxW.device) if s_bin_selected_BxHMxWMx1 is None else s_bin_selected_BxHMxWMx1

        # ===== prune part : merge [BEG] =====
        padded_embed_1xD = torch.zeros(1, D).to(dtype=torch.float32, device=x_BxHTxWTxD.device)

        # calc saliency
        s_bin_selected_BxHMxWMx1xKxK = s_bin_selected_BxHMxWMx1.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 1, 2, 2)
        s_bin_selected_BxHTxWTx1 = cross_fold(s_bin_selected_BxHMxWMx1xKxK)

        # calculate avg x_avg_BxHMxWMxD
        x_avg_BxHMxWMxD = cross_unfold(x_BxHTxWTxD).mean(dim=[-2, -1])
        x_ori_BxHTxWTxD = x_BxHTxWTxD

        # flatten
        s_bin_selected_BxHTWT = s_bin_selected_BxHTxWTx1.flatten(start_dim=1)
        s_bin_selected_BxHMWM = s_bin_selected_BxHMxWMx1.flatten(start_dim=1)

        x_ori_BxHTWTxD = x_ori_BxHTxWTxD.flatten(start_dim=1, end_dim=2)
        x_avg_BxHMWMxD = x_avg_BxHMxWMxD.flatten(start_dim=1, end_dim=2)

        merged_avg_UxD_s = []
        splited_ori_UxD_s = []

        merged_count_B = []
        splited_count_B = []
        for b in range(B):
            # merged
            x_avg_HMWMxD = x_avg_BxHMWMxD[b]
            s_bin_entropy_HMWM = s_bin_selected_BxHMWM[b]
            merged_tokens_avg_UxD = x_avg_HMWMxD[~s_bin_entropy_HMWM]
            merged_avg_UxD_s.append(merged_tokens_avg_UxD)

            # split
            x_ori_HTWTxD = x_ori_BxHTWTxD[b]
            s_bin_entropy_HTWT = s_bin_selected_BxHTWT[b]
            splited_tokens_ori_UxD = x_ori_HTWTxD[s_bin_entropy_HTWT]
            splited_ori_UxD_s.append(splited_tokens_ori_UxD)

            count_tokens_merged = merged_tokens_avg_UxD.shape[0]
            count_tokens_splited = splited_tokens_ori_UxD.shape[0]
            merged_count_B.append(count_tokens_merged)
            splited_count_B.append(count_tokens_splited)

        max_token_count = max([m + s for m, s in zip(merged_count_B, splited_count_B)])

        merged_splited_padded_tokens_AxD_s = []
        for b in range(B):
            padding = max_token_count - merged_count_B[b] - splited_count_B[b]
            merged_splited_padded = [merged_avg_UxD_s[b], splited_ori_UxD_s[b], padded_embed_1xD.repeat(padding, 1)]
            padded_bind_tokens_AxD = torch.cat(merged_splited_padded, dim=0)
            merged_splited_padded_tokens_AxD_s.append(padded_bind_tokens_AxD)

        merged_splited_padded_tokens_BxAxD = torch.stack(merged_splited_padded_tokens_AxD_s, dim=0)

        # ===== prune part : merge [END] =====

        for blk in self.blocks:
            merged_splited_padded_tokens_BxAxD = blk(merged_splited_padded_tokens_BxAxD)

        # ===== prune part : restore [BEG] =====
        restore_x_merged_BxHMxWMxD = torch.zeros_like(x_avg_BxHMxWMxD).to(device=x_avg_BxHMxWMxD.device, dtype=torch.float32)
        restore_x_splited_BxHTxWTxD = torch.zeros_like(x_ori_BxHTxWTxD).to(device=x_avg_BxHMxWMxD.device, dtype=torch.float32)

        restore_x_merged_BxHMWMxD = restore_x_merged_BxHMxWMxD.flatten(start_dim=1, end_dim=2)
        restore_x_splited_BxHTWTxD = restore_x_splited_BxHTxWTxD.flatten(start_dim=1, end_dim=2)

        for b in range(B):
            merged_sidx = 0
            merged_eidx = merged_count_B[b]
            splited_sidx = merged_count_B[b]
            splited_eidx = merged_count_B[b] + splited_count_B[b]

            merged_splited_padded_tokens_AxD = merged_splited_padded_tokens_BxAxD[b]
            merged_tokens_UxD = merged_splited_padded_tokens_AxD[merged_sidx:merged_eidx]
            splited_tokens_UxD = merged_splited_padded_tokens_AxD[splited_sidx:splited_eidx]

            s_bin_entropy_HMWM = s_bin_selected_BxHMWM[b]
            s_bin_entropy_HTWT = s_bin_selected_BxHTWT[b]
            restore_x_merged_BxHMWMxD[b][~s_bin_entropy_HMWM] = merged_tokens_UxD
            restore_x_splited_BxHTWTxD[b][s_bin_entropy_HTWT] = splited_tokens_UxD

        restore_x_merged_BxHMxWMxD = restore_x_merged_BxHMWMxD.view(B, HM, WM, D)
        restore_x_splited_BxHTxWTxD = restore_x_splited_BxHTWTxD.view(B, HT, WT, D)

        restore_x_merged_BxHTxWTxD = cross_fold(restore_x_merged_BxHMxWMxD.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 1, 2, 2))

        s_bin_selected_BxHTxWTxD = s_bin_selected_BxHTxWTx1.repeat(1, 1, 1, D)

        restore_x_final_BxHTxWTxD = torch.where(
            s_bin_selected_BxHTxWTxD,
            restore_x_splited_BxHTxWTxD,
            restore_x_merged_BxHTxWTxD,
        )
        # ===== prune part : restore [END] =====

        x_BxDxHTxWT = self.neck(restore_x_final_BxHTxWTxD.permute(0, 3, 1, 2))
        return x_BxDxHTxWT