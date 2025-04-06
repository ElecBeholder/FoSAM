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


class DifferentiableTokenSelect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokens, final_mask):
        """
        tokens: Tensor, shape (B, N, D) —— 待选择的 token 特征
        final_mask: Tensor, shape (B, N)，前向中为 hard mask（0/1），表示哪些 token 被保留
        """
        ctx.save_for_backward(final_mask)
        B, N, D = tokens.shape
        token_list = []
        lengths = []
        # 对每个样本，根据 final_mask 选择 token（真正删除未选中的 token）
        for b in range(B):
            # 得到第 b 个样本中被选中的 token 索引
            indices = torch.nonzero(final_mask[b]).squeeze(-1)
            token_list.append(tokens[b, indices, :])
            lengths.append(indices.numel())
        max_len = max(lengths) if lengths else 0
        # 将各样本 token 序列 padding 至最大长度
        out_tokens = torch.zeros(B, max_len, D, device=tokens.device, dtype=tokens.dtype)
        out_mask = torch.zeros(B, max_len, device=tokens.device, dtype=torch.bool)
        for b in range(B):
            n = token_list[b].shape[0]
            if n > 0:
                out_tokens[b, :n, :] = token_list[b]
                out_mask[b, :n] = True
        # 保存必要信息，用于 backward 中将梯度散射回原始 token 位置
        ctx.max_len = max_len
        ctx.original_N = N
        return out_tokens, out_mask

    @staticmethod
    def backward(ctx, grad_out_tokens, grad_out_mask):
        # grad_out_mask 不传递梯度（mask 本身为离散变量）
        final_mask, = ctx.saved_tensors  # shape (B, N)
        B, N = final_mask.shape
        D = grad_out_tokens.shape[2]
        grad_tokens = torch.zeros(B, N, D, device=grad_out_tokens.device, dtype=grad_out_tokens.dtype)
        # 将每个样本中 grad_out_tokens 的梯度按照原先选中 token 的顺序散射回 (B, N, D)
        for b in range(B):
            indices = torch.nonzero(final_mask[b]).squeeze(-1)
            n = indices.numel()
            if n > 0:
                grad_tokens[b, indices, :] = grad_out_tokens[b, :n, :]
        # 对 final_mask 不传梯度
        return grad_tokens, None

def TokenSelect(tokens, final_mask):
    """
    tokens: Tensor, shape (B, N, D) —— 待选择的 token 特征
    final_mask: Tensor, shape (B, N)，前向中为 hard mask（0/1），表示哪些 token 被保留
    """
    B, N, D = tokens.shape
    token_list = []
    lengths = []
    # 对每个样本，根据 final_mask 选择 token（真正删除未选中的 token）
    for b in range(B):
        # 得到第 b 个样本中被选中的 token 索引
        indices = torch.nonzero(final_mask[b]).squeeze(-1)
        token_list.append(tokens[b, indices, :])
        lengths.append(indices.numel())
    max_len = max(lengths) if lengths else 0
    # 将各样本 token 序列 padding 至最大长度
    out_tokens = torch.zeros(B, max_len, D, device=tokens.device, dtype=tokens.dtype)
    out_mask = torch.zeros(B, max_len, device=tokens.device, dtype=torch.bool)
    for b in range(B):
        n = token_list[b].shape[0]
        out_tokens[b, :n, :] = token_list[b]
        out_mask[b, :n] = True

    return out_tokens, out_mask

def differentiable_topk_select(tokens, scores, k, tau=1.0, hard=True):
    """
    利用迭代的 Gumbel-Softmax 实现 Differentiable Top-k TokenSelect，
    返回真正删除未选 token 后的结果，输出形状为 (B, k, D)。
    
    Args:
        tokens: Tensor, shape (B, N, D)，待选择的 token 特征。
        scores: Tensor, shape (B, N)，每个 token 的得分（越高表示越重要）。
        k: int，需要选择的 token 数量。
        tau: float，温度参数，控制 Gumbel-Softmax 的平滑程度。
        hard: bool，如果为 True，则前向输出近似 one-hot，反向利用 soft 梯度（STE）。
        
    返回:
        selected_tokens: Tensor, shape (B, k, D)，选中的 token（真正删除了未选 token）。
    """
    B, N, D = tokens.shape
    # 拷贝一份 scores 用于迭代更新，避免改变原始 scores
    remaining_scores = scores.clone()  # shape: (B, N)
    selected_tokens_list = []
    
    for i in range(k):
        # 为每个样本生成 Gumbel 噪声
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(remaining_scores) + 1e-10) + 1e-10)
        # 加入噪声后除以温度，得到 logits
        logits = (remaining_scores + gumbel_noise) / tau  # shape: (B, N)
        # 利用 Gumbel-Softmax 采样得到 one-hot 选择向量
        one_hot = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)  # shape: (B, N)
        # 计算选中的 token：利用 one_hot 与 tokens 做加权求和（若 hard=True 则近似 one-hot）
        token_selected = torch.sum(tokens * one_hot.unsqueeze(-1), dim=1)  # shape: (B, D)
        selected_tokens_list.append(token_selected.unsqueeze(1))  # shape: (B, 1, D)
        
        # 更新 remaining_scores：将已选位置屏蔽（减去一个足够大的数）
        remaining_scores = remaining_scores - one_hot * 1e9
    
    # 将 k 轮选择得到的 token 拼接起来，形状为 (B, k, D)
    selected_tokens = torch.cat(selected_tokens_list, dim=1)
    return selected_tokens

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
        assert x_BxCxHxW.shape[2] == self.img_size and x_BxCxHxW.shape[3] == self.img_size, \
            "Input image size must match self.img_size"

        # 1. Patch Embedding 与位置编码
        x_BxDxHTxWT = self.patch_embed(x_BxCxHxW)  # (B, D, HT, WT)
        x_BxHTxWTxD = x_BxDxHTxWT.permute(0, 2, 3, 1)  # (B, HT, WT, D)
        x_BxHTxWTxD = x_BxHTxWTxD + get_abs_pos(self.pos_embed, self.pretrain_use_cls_token,
                                                  [x_BxHTxWTxD.shape[1], x_BxHTxWTxD.shape[2]])
        B, HT, WT, D = x_BxHTxWTxD.shape
        N = HT * WT  # token 数


        # # 2. 计算 saliency
        # if s_bin_selected_BxHMxWMx1 is None:
        #     # 若未传入，则随机生成（仅用于测试），形状 (B, HT, WT)
        #     saliency = torch.randn(B, HT, WT, device=x_BxHTxWTxD.device)
        # else:
        # s_bin_selected_BxHMxWMx1: (B, HM, WM, 1)
        # 利用上采样将 saliency 转换为 (B, HT, WT, 1)，再 squeeze 成 (B, HT, WT)
        #s_bin = s_bin_selected_BxHMxWMx1.float().permute(0, 3, 1, 2)  # (B, 1, HM, WM)
        # s_bin_expanded = F.interpolate(s_bin, scale_factor=2, mode='nearest')
        # saliency  = s_bin_expanded.permute(0, 2, 3, 1)  # (B, 2*HM, 2*WM, 1)
        s_bin = s_bin_selected_BxHMxWMx1.permute(0, 3, 1, 2)  # (B, 1, HM, WM)
        # s_bin_expanded = F.interpolate(s_bin, scale_factor=2, mode='nearest')
        # saliency  = s_bin_expanded.permute(0, 2, 3, 1)  # (B, 2*HM, 2*WM, 1)
        saliency  = s_bin.permute(0, 2, 3, 1)  # (B, 2*HM, 2*WM, 1)

        # 将 token 特征展平为 (B, N, D)；将 saliency 展平为 (B, N)
        tokens = x_BxHTxWTxD.view(B, N, D)
        saliency_flat = saliency.view(B, N)

        # 3. 计算连续 soft mask 与 hard mask
        # 这里使用阈值 0.5；temperature 越小，soft mask 越接近硬判断
        # temperature = 0.01
        temperature = 0.1
        soft_mask = torch.sigmoid((saliency_flat - 0.5) / temperature)
        hard_mask = (saliency_flat > 0.5).float()
        # 利用 STE 构造最终 mask：前向用 hard mask，反向传递 soft mask 的梯度
        final_mask = hard_mask + (soft_mask - soft_mask.detach())
        # final_mask 形状为 (B, N)，前向数值为 0 或 1，但梯度由 soft_mask 提供

        # 4. 利用自定义函数根据 final_mask 选择 token（真正删除未选中 token）
        # DifferentiableTokenSelect.forward 会返回 (selected_tokens, selected_mask)
        selected_tokens, selected_mask = DifferentiableTokenSelect.apply(tokens, final_mask)
        # selected_tokens = differentiable_topk_select(tokens, final_mask, 400)
        # selected_tokens: (B, k, D)，其中 k 为每个样本保留的 token 数（可能不同，但已 padding 成相同长度）
        # selected_mask: (B, k) 为布尔 mask，指示哪些位置为有效 token

        # 5. 将选中的 token 序列送入 Transformer Blocks 处理
        x_processed = selected_tokens  # (B, k, D)
        for blk in self.blocks:
            # 如果 Transformer Block 支持传入 padding mask，请传入 selected_mask
            x_processed = blk(x_processed)
        
        # 6. 恢复为原始 token 数量：创建一个 (B, N, D) 全 0 张量，然后将处理后的 token 根据原始顺序散射回来
        restored_tokens = torch.zeros(B, N, D, device=tokens.device, dtype=tokens.dtype)
        for b in range(B):
            # 得到当前样本中被保留 token 的索引（按照原始 token 顺序）
            indices = torch.nonzero(hard_mask[b]).squeeze(-1)  # 长度 n_b
            n = indices.numel()
            if n > 0:
                # 将处理后的 token（来自 Transformer Blocks）散射回原来的位置
                restored_tokens[b, indices, :] = x_processed[b, :n, :]
        # 这样，原始 token 序列中未被选中的位置仍为 0

        # 7. 恢复为图像空间排列：reshape 成 (B, HT, WT, D)，再转换为 (B, D, HT, WT) 送入 neck 层
        restored_tokens = restored_tokens.view(B, HT, WT, D)
        x_BxDxHTxWT = self.neck(restored_tokens.permute(0, 3, 1, 2))

        return x_BxDxHTxWT