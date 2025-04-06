import torch
import pdb
import random
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import torchvision.utils as vutils
from . import resnet, resnext, mobilenet, hrnetv2_nodownsp
from DynamicFocus_foveal_seg_classification.lib.nn import SynchronizedBatchNorm2d
from DynamicFocus_foveal_seg_classification.dataset import imresize, b_imresize
from builtins import any as b_any

from DynamicFocus_foveal_seg_classification.lib.utils import as_numpy
from DynamicFocus_foveal_seg_classification.utils import colorEncode
from scipy.io import loadmat
import numpy as np
from PIL import Image
from PIL import ImageFilter
import time
import os
import shutil
from scipy import ndimage
import scipy.interpolate
import cv2
import torchvision.models as models
from DynamicFocus_foveal_seg_classification.saliency_network import saliency_network_resnet18, fov_simple, saliency_network_resnet18_stride1, FovSimModule
from DynamicFocus_foveal_seg_classification.models.model_utils import Resnet, ResnetDilated, MobileNetV2Dilated, C1DeepSup, C1, PPM, PPMDeepsup, UPerNet
from DynamicFocus_foveal_seg_classification.DynamicFocus.utility.torch_tools import gen_grid_mtx_2xHxW
BatchNorm2d = SynchronizedBatchNorm2d
from pytorch_toolbelt.losses.dice import DiceLoss

from torch.autograd import Variable
import matplotlib.pyplot as plt


class GaussianPredictor_old(nn.Module):
    def __init__(self, input_size=20, input_channels=386):
        super(GaussianPredictor_old, self).__init__()
        BN_MOMENTUM = 0.1
        self.input_size = input_size
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=5*20, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=5*20, out_channels=5*20, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=5*20, out_channels=16, kernel_size=3, stride=1, padding=1)
        #self.fc1 = nn.Linear(16*(input_size//4)*(input_size//4), 64)
        self.fc1 = nn.Linear(16*(input_size//4)*(input_size//4), 7)
        #self.fc2 = nn.Linear(64 + 2, 7)  # 修改为7个输出参数，增加了第二个高斯分布的sigma_x和sigma_y
        self.norm1 = BatchNorm2d(5*20, momentum=BN_MOMENTUM)
        self.norm2 = BatchNorm2d(5*20, momentum=BN_MOMENTUM)
        self.norm3 = BatchNorm2d(16, momentum=BN_MOMENTUM)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, gaze_coords=None):
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.norm2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.norm3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        #x = F.relu(self.fc1(x))
        x = self.fc1(self.dropout(x))
        
        # 如果提供了注视点坐标，将其连接到fc1的输出
        #if gaze_coords is not None:
        #    x = torch.cat([x, gaze_coords], dim=1)
            
        #x = self.fc2(x)
        return x

class GaussianPredictor_multiGaussian(nn.Module):
    def __init__(self, input_size=20, input_channels=386):
        super(GaussianPredictor_multiGaussian, self).__init__()
        BN_MOMENTUM = 0.1
        self.input_size = input_size
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=10*20, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=10*20, out_channels=10*20, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=10*20, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32*(input_size//4)*(input_size//4), 45)  # 输出9个高斯分布，每个5个参数
        self.norm1 = BatchNorm2d(10*20, momentum=BN_MOMENTUM)
        self.norm2 = BatchNorm2d(10*20, momentum=BN_MOMENTUM)
        self.norm3 = BatchNorm2d(32, momentum=BN_MOMENTUM)
        self.dropout = nn.Dropout(p=0.1)
        
        # 生成9个基准点的坐标网格 (3x3 grid)
        self.grid_size = 3
        x_points = torch.linspace(0.16, 0.84, self.grid_size)
        y_points = torch.linspace(0.16, 0.84, self.grid_size)
        grid_y, grid_x = torch.meshgrid(y_points, x_points, indexing='ij')
        anchor_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)  # [9, 2]
        self.register_buffer('anchor_points', anchor_points)

    def forward(self, x, gaze_coords=None):
        batch_size = x.size(0)
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.norm2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.norm3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        # 网络输出的是每个高斯分布的5个参数
        x = self.fc1(self.dropout(x))  # [B, 45]
        
        # 重塑为 [B, 9, 5] 形状，代表9个高斯分布，每个有5个参数
        params = x.view(batch_size, 9, 5)
        
        # 使用基准点作为锚点，将预测的值解释为相对于锚点的偏移
        # 前两个参数（均值）使用偏移值
        mu_offsets = params[:, :, :2]  # [B, 9, 2] 均值的偏移量
        
        # 扩展锚点到批次大小 [B, 9, 2]
        anchors = self.anchor_points.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 最终的参数是：均值 = 锚点 + tanh(偏移) * 0.3，用于限制偏移范围
        # 其他参数（sigma_x, sigma_y, rho）保持不变
        mu_values = anchors + torch.tanh(mu_offsets) * 0.3
        
        # 创建包含所有参数的张量，每个高斯分布的5个参数依次排列
        final_params = torch.zeros_like(params)
        final_params[:, :, :2] = mu_values      # 替换均值参数
        final_params[:, :, 2:] = params[:, :, 2:]  # 保持其他参数不变
        
        # 将参数重组为 [B, 45]，按照每个高斯分布的5个参数依次排列
        result = final_params.reshape(batch_size, -1)
        
        return result

class GaussianPredictor(nn.Module):
    def __init__(self, input_size=20, input_channels=386):
        super(GaussianPredictor, self).__init__()
        self.input_size = input_size
        dim = 128  # 隐藏层维度
        num_heads = 4  # 多头注意力的头数
        num_layers = 4  # Transformer层数
        mlp_dim = 256  # MLP中间层维度
        dropout = 0.1  # dropout率
        
        # 映射到Transformer维度
        self.embedding = nn.Conv2d(input_channels, dim, kernel_size=1)
        
        # 添加特殊的[CLS]令牌用于全局表示
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        
        # Transformer编码器层
        encoder_layers = []
        for _ in range(num_layers):
            encoder_layers.append(
                nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=num_heads,
                    dim_feedforward=mlp_dim,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True
                )
            )
        self.transformer_layers = nn.ModuleList(encoder_layers)
        
        # 输出层 - 预测7个高斯参数
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 7)
        self.dropout = nn.Dropout(dropout)
        
        # 注意力池化 - 学习如何汇总token特征
        self.attn_pool = nn.Linear(dim, 1)
        
        # 如果提供gaze_coords，则使用投影层
        self.gaze_proj = nn.Linear(2, dim)

    def forward(self, x, gaze_coords=None):
        B, C, H, W = x.shape
        
        # 映射到Transformer维度
        x = self.embedding(x)  # [B, dim, H, W]
        
        # 重塑为序列
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, dim]
        
        # 添加[CLS]令牌
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 1+H*W, dim]
        
        # 如果有gaze_coords，将其融合到序列中
        if gaze_coords is not None:
            gaze_emb = self.gaze_proj(gaze_coords)  # [B, dim]
            gaze_emb = gaze_emb.unsqueeze(1)  # [B, 1, dim]
            x = torch.cat([gaze_emb, x], dim=1)  # [B, 2+H*W, dim]
        
        # 通过Transformer层
        for layer in self.transformer_layers:
            x = layer(x)
        
        # 混合使用[CLS]令牌和注意力池化
        cls_feature = x[:, 0]  # 使用[CLS]令牌作为特征
        
        # 对其余token进行注意力池化
        if gaze_coords is not None:
            tokens = x[:, 2:]  # 跳过[CLS]和gaze tokens
        else:
            tokens = x[:, 1:]  # 仅跳过[CLS] token
            
        attn_weights = F.softmax(self.attn_pool(tokens).squeeze(-1), dim=1)  # [B, H*W]
        attn_feature = (tokens * attn_weights.unsqueeze(-1)).sum(dim=1)  # [B, dim]
        
        # 组合特征
        x = cls_feature + attn_feature  # 简单相加特征
        
        # 规范化和预测
        x = self.norm(x)
        x = self.head(self.dropout(x))  # [B, 7]
        
        return x

"""
GaussianPredictor_multiGaussian_ViT - Vision Transformer版本的多高斯分布预测器

这个预测器使用Vision Transformer架构来预测9个高斯分布的参数，用于更精确地拟合分割掩码。
每个高斯分布由5个参数描述：mu_x, mu_y, sigma_x, sigma_y, rho。

特点：
- 使用ViT架构替代传统的CNN网络，提高特征提取和关系建模能力
- 支持9个高斯分布，每个分布有5个参数，共输出45个参数
- 使用3x3网格中的锚点作为基准，预测相对偏移，提高稳定性
- 支持注意力池化，更好地聚焦于重要区域
- 提供可选的gaze_coords输入，允许将注视点信息融入预测

该预测器已设置为DeformSegmentationModule的默认预测器，无需额外配置。

参数：
- input_size: 输入特征图的大小，默认为20
- input_channels: 输入特征图的通道数，默认为386

返回：
- 大小为[B, 45]的张量，包含9个高斯分布的参数
"""
class GaussianPredictor_multiGaussian_ViT(nn.Module):
    def __init__(self, input_size=20, input_channels=386):
        super(GaussianPredictor_multiGaussian_ViT, self).__init__()
        self.input_size = input_size
        dim = 192  # 增加隐藏层维度以处理更多参数
        num_heads = 6  # 增加注意力头数
        num_layers = 4  # 增加Transformer层数
        mlp_dim = 384  # 增加MLP中间层维度
        dropout = 0.1  # dropout率
        
        # 映射到Transformer维度
        self.embedding = nn.Conv2d(input_channels, dim, kernel_size=1)
        
        # 添加特殊的[CLS]令牌用于全局表示
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        
        
        # Transformer编码器层
        encoder_layers = []
        for _ in range(num_layers):
            encoder_layers.append(
                nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=num_heads,
                    dim_feedforward=mlp_dim,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True
                )
            )
        self.transformer_layers = nn.ModuleList(encoder_layers)
        
        # 输出层 - 预测9个高斯分布，每个5个参数
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 45)  # 9个高斯分布 * 5个参数
        self.dropout = nn.Dropout(dropout)
        
        # 注意力池化
        self.attn_pool = nn.Linear(dim, 1)
        
        # 如果提供gaze_coords，则使用投影层
        self.gaze_proj = nn.Linear(2, dim)
        
        # 生成9个基准点的坐标网格 (3x3 grid)
        self.grid_size = 3
        x_points = torch.linspace(0.16, 0.84, self.grid_size)
        y_points = torch.linspace(0.16, 0.84, self.grid_size)
        grid_y, grid_x = torch.meshgrid(y_points, x_points, indexing='ij')
        anchor_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)  # [9, 2]
        self.register_buffer('anchor_points', anchor_points)
        
        # 初始化权重，确保mu_values初始为0
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重，确保mu_values初始为0，标准差合理化"""
        # 使用torch.no_grad()上下文管理器临时禁用梯度计算
        with torch.no_grad():
            # 首先对所有权重进行标准初始化
            nn.init.normal_(self.head.weight, std=0.02)
            nn.init.zeros_(self.head.bias)
            
            # 对于每个高斯分布，专门初始化对应的偏置项
            for i in range(9):
                # mu_x和mu_y的偏置设为0，使得初始的mu_values就是锚点位置
                # 每个高斯有5个参数，前两个是mu_x和mu_y
                self.head.bias[i*5] = 0.0  # mu_x 偏置为0
                self.head.bias[i*5 + 1] = 0.0  # mu_y 偏置为0
                
                # 初始化sigma_x和sigma_y为合理的值（通过softplus激活后）
                # softplus(0) ≈ 0.693，适合初始的标准差
                self.head.bias[i*5 + 2] = 0.0  # sigma_x
                self.head.bias[i*5 + 3] = 0.0  # sigma_y
                
                # 初始化rho为0（通过tanh后为0，意味着无相关性）
                self.head.bias[i*5 + 4] = 0.0  # rho
            
            # 初始化CLS token
            nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x, gaze_coords=None):
        batch_size = x.size(0)
        
        # 映射到Transformer维度
        x = self.embedding(x)  # [B, dim, H, W]
        
        # 重塑为序列
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, dim]
        
        # 添加[CLS]令牌
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 1+H*W, dim]
        
        # 如果有gaze_coords，将其融合到序列中
        if gaze_coords is not None:
            gaze_emb = self.gaze_proj(gaze_coords)  # [B, dim]
            gaze_emb = gaze_emb.unsqueeze(1)  # [B, 1, dim]
            x = torch.cat([gaze_emb, x], dim=1)  # [B, 2+H*W, dim]
        
        # 通过Transformer层
        for layer in self.transformer_layers:
            x = layer(x)
        
        # 混合使用[CLS]令牌和注意力池化
        cls_feature = x[:, 0]  # 使用[CLS]令牌作为特征
        
        # 对其余token进行注意力池化
        if gaze_coords is not None:
            tokens = x[:, 2:]  # 跳过[CLS]和gaze tokens
        else:
            tokens = x[:, 1:]  # 仅跳过[CLS] token
            
        attn_weights = F.softmax(self.attn_pool(tokens).squeeze(-1), dim=1)  # [B, H*W]
        attn_feature = (tokens * attn_weights.unsqueeze(-1)).sum(dim=1)  # [B, dim]
        
        # 组合特征
        x = cls_feature + attn_feature  # 简单相加特征
        
        # 规范化和预测
        x = self.norm(x)
        x = self.head(self.dropout(x))  # [B, 45]
        
        # 重塑为 [B, 9, 5] 形状，代表9个高斯分布，每个有5个参数
        params = x.view(batch_size, 9, 5)
        
        # 使用基准点作为锚点，将预测的值解释为相对于锚点的偏移
        # 前两个参数（均值）使用偏移值
        mu_offsets = params[:, :, :2]  # [B, 9, 2] 均值的偏移量
        
        # 扩展锚点到批次大小 [B, 9, 2]
        anchors = self.anchor_points.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 最终的参数是：均值 = 锚点 + tanh(偏移) * 0.3，用于限制偏移范围
        # 其他参数（sigma_x, sigma_y, rho）保持不变
        mu_values = 0.5 + torch.tanh(mu_offsets) * 0.5
        
        # 创建包含所有参数的张量，每个高斯分布的5个参数依次排列
        final_params = torch.zeros_like(params)
        final_params[:, :, :2] = mu_values  # 替换均值参数
        
        # 限制sigma参数，防止形成过于细长的高斯分布
        # 1. 使用sigmoid确保sigma值在合理范围内
        sigma_x = F.sigmoid(params[:, :, 2]) * 0.3 + 0.05  # 范围限制在[0.05, 0.35]
        sigma_y = F.sigmoid(params[:, :, 3]) * 0.3 + 0.05  # 范围限制在[0.05, 0.35]
        
        # 2. 限制sigma_x和sigma_y的比例，防止过于细长的高斯
        ratio = torch.max(sigma_x/sigma_y, sigma_y/sigma_x)
        max_ratio = 2.0  # 最大允许的纵横比
        scale_factor = torch.min(max_ratio / ratio, torch.ones_like(ratio))
        
        # 应用比例限制：如果比例大于阈值，同时缩放两个sigma值保持面积不变
        # 对于比例过大的高斯，增大较小的sigma并减小较大的sigma
        is_x_larger = sigma_x > sigma_y
        adjusted_sigma_x = torch.where(
            is_x_larger,
            sigma_x * torch.sqrt(scale_factor),
            sigma_x / torch.sqrt(scale_factor)
        )
        adjusted_sigma_y = torch.where(
            is_x_larger,
            sigma_y / torch.sqrt(scale_factor),
            sigma_y * torch.sqrt(scale_factor)
        )
        
        # 用调整后的sigma值替换原来的参数
        final_params[:, :, 2] = torch.log(torch.exp(adjusted_sigma_x) - 1)  # 反向计算softplus的输入
        final_params[:, :, 3] = torch.log(torch.exp(adjusted_sigma_y) - 1)  # 反向计算softplus的输入
        
        # 3. 对相关系数rho施加更严格的约束，防止极端的倾斜
        final_params[:, :, 4] = params[:, :, 4] * 0.5  # 将rho的范围缩小到[-0.5, 0.5]
        
        # 将参数重组为 [B, 45]，按照每个高斯分布的5个参数依次排列
        result = final_params.reshape(batch_size, -1)
        
        return result

def generate_saliency_map(param, H=40, W=40, gaze_coords=None):
    B = param.size(0)
    
    # 第一个高斯分布参数
    mu_x = torch.sigmoid(param[:, 0]).view(B, 1, 1)
    mu_y = torch.sigmoid(param[:, 1]).view(B, 1, 1)
    sigma_x1 = (F.softplus(param[:, 2]) + 1e-6).view(B, 1, 1)
    sigma_y1 = (F.softplus(param[:, 3]) + 1e-6).view(B, 1, 1)
    rho = (torch.tanh(param[:, 4])).view(B, 1, 1)
    
    # 第二个高斯分布参数
    sigma_x2 = (F.softplus(param[:, 5]) + 1e-6).view(B, 1, 1)
    sigma_y2 = (F.softplus(param[:, 6]) + 1e-6).view(B, 1, 1)
    
    # 第二个高斯分布使用注视点作为均值
    if gaze_coords is not None:
        gaze_mu_x = gaze_coords[:, 1].view(B, 1, 1)  # 注视点x坐标
        gaze_mu_y = gaze_coords[:, 0].view(B, 1, 1)  # 注视点y坐标
    else:
        # 如果没有注视点信息，使用第一个高斯的均值
        gaze_mu_x = mu_x
        gaze_mu_y = mu_y

    x = torch.linspace(0, 1, W, device=param.device)
    y = torch.linspace(0, 1, H, device=param.device)

    xx, yy = torch.meshgrid(x, y, indexing='xy')
    xx = xx.unsqueeze(0).expand(B, H, W)
    yy = yy.unsqueeze(0).expand(B, H, W)

    # 计算第一个高斯分布
    norm_const1 = 1 / (2 * torch.pi * sigma_x1 * sigma_y1 * torch.sqrt(1 - rho**2))
    z_x1 = (xx - mu_x) / sigma_x1
    z_y1 = (yy - mu_y) / sigma_y1
    exponent1 = (z_x1**2 - 2 * rho * z_x1 * z_y1 + z_y1**2) / (2 * (1 - rho**2))
    saliency_map1 = norm_const1 * torch.exp(-exponent1)

    # 计算第二个高斯分布（无相关系数，即rho=0）
    norm_const2 = 1 / (2 * torch.pi * sigma_x2 * sigma_y2)
    z_x2 = (xx - gaze_mu_x) / sigma_x2
    z_y2 = (yy - gaze_mu_y) / sigma_y2
    exponent2 = (z_x2**2 + z_y2**2) / 2
    saliency_map2 = norm_const2 * torch.exp(-exponent2)

    # 将两个高斯分布相加
    saliency_map = saliency_map1 + saliency_map2
    
    return saliency_map

def generate_saliency_map_multiGaussian(param, H=40, W=40):
    """
    根据9个高斯分布参数生成显著性图。每个高斯分布有5个参数。
    
    Args:
        param: 形状为 [B, 45] 的高斯参数张量，每个样本有9个高斯分布，每个有5个参数
        H: 输出高度
        W: 输出宽度
        
    Returns:
        形状为 [B, H, W] 的显著性图
    """
    B = param.size(0)
    num_gaussians = 9
    params_per_gaussian = 5
    
    # 将参数重塑为 [B, 9, 5]
    params_reshaped = param.view(B, num_gaussians, params_per_gaussian)
    
    # 生成坐标网格
    x = torch.linspace(0, 1, W, device=param.device)
    y = torch.linspace(0, 1, H, device=param.device)
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    xx = xx.unsqueeze(0).expand(B, H, W)
    yy = yy.unsqueeze(0).expand(B, H, W)
    
    # 初始化显著性图
    saliency_map = torch.zeros(B, H, W, device=param.device)
    
    # 计算每个高斯分布并累加
    for i in range(num_gaussians):
        # 提取第i个高斯分布的参数
        mu_x = params_reshaped[:, i, 0].view(B, 1, 1)
        mu_y = params_reshaped[:, i, 1].view(B, 1, 1)
        sigma_x = (F.softplus(params_reshaped[:, i, 2]) + 1e-6).view(B, 1, 1)
        sigma_y = (F.softplus(params_reshaped[:, i, 3]) + 1e-6).view(B, 1, 1)
        rho = torch.tanh(params_reshaped[:, i, 4]).view(B, 1, 1)
        
        # 计算高斯分布
        norm_const = 1 / (2 * torch.pi * sigma_x * sigma_y * torch.sqrt(1 - rho**2))
        z_x = (xx - mu_x) / sigma_x
        z_y = (yy - mu_y) / sigma_y
        exponent = (z_x**2 - 2 * rho * z_x * z_y + z_y**2) / (2 * (1 - rho**2))
        gaussian_map = norm_const * torch.exp(-exponent)
        
        # 累加到显著性图

        #norm_gaussian_map = gaussian_map / (gaussian_map.max() + 1e-6)
        H = gaussian_map.shape[1]
        W = gaussian_map.shape[2]
        norm_gaussian_map = (gaussian_map.view(B,-1) / gaussian_map.view(B,-1).max(axis=1)[0].unsqueeze(1)).view(B, H, W)
        saliency_map += norm_gaussian_map
        #saliency_map += gaussian_map
    
    return saliency_map

def differentiable_topk_mask(scores_BxHW, K, temperature):
    #gumbel_noise = sample_gumbel(scores_BxHW.shape, device='cuda')
    logits_BxHW = scores_BxHW# + 0.001*gumbel_noise
    soft_mask_BxHW = F.softmax(logits_BxHW / temperature, dim=1)
    _, indices = torch.topk(logits_BxHW, K, dim=1)
    hard_mask_BxHW = torch.zeros_like(logits_BxHW)
    hard_mask_BxHW.scatter_(1, indices, 1.0)
    # STE trick：forward 用硬 mask，但 backward 用 soft_mask 的梯度

    Lambda = logits_BxHW.shape[1]
    mask_BxHW = (hard_mask_BxHW - Lambda * soft_mask_BxHW).detach() + Lambda * soft_mask_BxHW
    #mask_BxHW = soft_mask_BxHW

    #mask_BxHW = (hard_mask_BxHW - soft_mask_BxHW).detach() + soft_mask_BxHW
    #if random.randint(0, 100) == 1:
    #    plt.imsave('../l_sam_experiment/selected_mask_{}.png'.format(random.randint(1,100)), mask_BxHW.detach().clone().view(8,40,40)[0,:,:].cpu().numpy(), cmap='gray')
    return mask_BxHW, indices

def differentiable_topk(feature_map_BxCxHxW, saliency_map_BxHxW, H_s=20, W_s=20, temperature=0.1):
    B, C, H, W = feature_map_BxCxHxW.shape
    K = H_s * W_s
    saliency_flat_BxHW = saliency_map_BxHxW.view(B, -1)
    mask_BxHW, indices = differentiable_topk_mask(saliency_flat_BxHW, K, temperature)
    feature_flat_BxCxHW = feature_map_BxCxHxW.view(B, C, -1)
    output_feature_map_BxCxHW = feature_flat_BxCxHW * mask_BxHW.unsqueeze(1)
    indices_exp = indices.unsqueeze(1).expand(B, C, K)
    selected_feature_map_BxCxK = torch.gather(output_feature_map_BxCxHW, 2, indices_exp)
    _, sort_order = torch.sort(indices, dim=1)
    sort_order_exp = sort_order.unsqueeze(1).expand(B, C, K)
    selected_feature_map_sorted_BxCxK = torch.gather(selected_feature_map_BxCxK, 2, sort_order_exp)
    indices = torch.gather(indices, 1, sort_order)
    return selected_feature_map_sorted_BxCxK.view(B, C, H_s, W_s), indices

class SoftDiceLossV1(nn.Module):
    '''
    soft-dice loss, useful in binary segmentation
    '''
    def __init__(self,
                 p=2,
                 smooth=0):
        super(SoftDiceLossV1, self).__init__()
        self.p = p
        self.smooth = smooth

    def forward(self, logits, labels):
        '''
        inputs:
            logits: tensor of shape (N, H, W, ...)
            label: tensor of shape(N, H, W, ...)
        output:
            loss: tensor of shape(1, )
        '''
        logits = torch.permute(logits, (0,2,3,1)).cuda()

        probs = torch.sigmoid(logits)
        #print(logits.shape, labels.shape)
        #print(probs.max(), probs.min())
        #print(labels.max())
        numer = (probs * labels).sum()
        denor = (probs.pow(self.p) + labels.pow(self.p)).sum()
        loss = 1. - (2 * numer + self.smooth) / (denor + self.smooth)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()
    
    def forward(self, y_pred_ds_Bx1xHSxWS):
        batch_size = y_pred_ds_Bx1xHSxWS.size(0)
        h = y_pred_ds_Bx1xHSxWS.size(2)
        w = y_pred_ds_Bx1xHSxWS.size(3)

        h_tv = torch.abs(y_pred_ds_Bx1xHSxWS[:, :, 1:, :] - y_pred_ds_Bx1xHSxWS[:, :, :-1, :]).sum()
        w_tv = torch.abs(y_pred_ds_Bx1xHSxWS[:, :, :, 1:] - y_pred_ds_Bx1xHSxWS[:, :, :, :-1]).sum()

        count_h = (y_pred_ds_Bx1xHSxWS.size(2) - 1) * y_pred_ds_Bx1xHSxWS.size(3)
        count_w = y_pred_ds_Bx1xHSxWS.size(2) * (y_pred_ds_Bx1xHSxWS.size(3) - 1)
        total_variation_loss = (h_tv / count_h + w_tv / count_w) / batch_size

        return total_variation_loss   

# class DiceLoss(nn.Module):
#     """Dice Loss PyTorch
#         Created by: Zhang Shuai
#         Email: shuaizzz666@gmail.com
#         dice_loss = 1 - 2*p*t / (p^2 + t^2). p and t represent predict and target.
#     Args:
#         weight: An array of shape [C,]
#         predict: A float32 tensor of shape [N, C, *], for Semantic segmentation task is [N, C, H, W]
#         target: A int64 tensor of shape [N, *], for Semantic segmentation task is [N, H, W]
#     Return:
#         diceloss
#     """
#     def __init__(self, weight=None):
#         super(DiceLoss, self).__init__()
#         if weight is not None:
#             weight = torch.Tensor(weight)
#             self.weight = weight / torch.sum(weight) # Normalized weight
#         self.smooth = 1e-5

#     def forward(self, predict, target):
#         N, C = predict.size()[:2]
#         predict = predict.view(N, C, -1) # (N, C, *)
#         target = target.view(N, 1, -1) # (N, 1, *)

#         predict = F.softmax(predict, dim=1) # (N, C, *) ==> (N, C, *)
#         ## convert target(N, 1, *) into one hot vector (N, C, *)
#         target_onehot = torch.zeros(predict.size()).cuda()  # (N, 1, *) ==> (N, C, *)
#         target_onehot.scatter_(1, target, 1)  # (N, C, *)

#         intersection = torch.sum(predict * target_onehot, dim=2)  # (N, C)
#         union = torch.sum(predict.pow(2), dim=2) + torch.sum(target_onehot, dim=2)  # (N, C)
#         ## p^2 + t^2 >= 2*p*t, target_onehot^2 == target_onehot
#         dice_coef = (2 * intersection + self.smooth) / (union + self.smooth)  # (N, C)

#         if hasattr(self, 'weight'):
#             if self.weight.type() != predict.type():
#                 self.weight = self.weight.type_as(predict)
#                 dice_coef = dice_coef * self.weight * C  # (N, C)
#         dice_loss = 1 - torch.mean(dice_coef)  # 1

#         return dice_loss
        
def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

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

    def getPixelsForInterp_NB(img):
        """
        Calculates a mask of pixels neighboring invalid values -
           to use for interpolation.
        """
        # mask invalid pixels
        img = img.cpu().numpy()
        invalid_mask = np.isnan(img)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        #dilate to mark borders around invalid regions
        if max(invalid_mask.shape) > 512:
            dr = max(invalid_mask.shape)/512
            input = torch.tensor(invalid_mask.astype('float')).unsqueeze(0)
            shape_ori = (invalid_mask.shape[-2], int(invalid_mask.shape[-1]))
            shape_scaled = (int(invalid_mask.shape[-2]/dr), int(invalid_mask.shape[-1]/dr))
            input_scaled = F.interpolate(input, shape_scaled, mode='nearest').squeeze(0)
            invalid_mask_scaled = np.array(input_scaled).astype('bool')
            dilated_mask_scaled = cv2.dilate(invalid_mask_scaled.astype('uint8'), kernel,
                              borderType=cv2.BORDER_CONSTANT, borderValue=int(0))
            dilated_mask_scaled_t = torch.tensor(dilated_mask_scaled.astype('float')).unsqueeze(0)
            dilated_mask = F.interpolate(dilated_mask_scaled_t, shape_ori, mode='nearest').squeeze(0)
            dilated_mask = np.array(dilated_mask).astype('uint8')
        else:
            dilated_mask = cv2.dilate(invalid_mask.astype('uint8'), kernel,
                              borderType=cv2.BORDER_CONSTANT, borderValue=int(0))

        # pixelwise "and" with valid pixel mask (~invalid_mask)
        masked_for_interp = dilated_mask *  ~invalid_mask
        return masked_for_interp.astype('bool'), invalid_mask

    # Mask pixels for interpolation
    if interp_mode == 'nearest':
        interpolator=scipy.interpolate.NearestNDInterpolator
        mask_for_interp, invalid_mask = getPixelsForInterp_NB(target_for_interp)
    elif interp_mode == 'BI':
        interpolator=scipy.interpolate.LinearNDInterpolator
        mask_for_interp, invalid_mask = getPixelsForInterp_NB(target_for_interp)
    else:
        # interpolator=Interp2D(target_for_interp.shape[-2], target_for_interp.shape[-1])
        mask_for_interp, invalid_mask = getPixelsForInterp(target_for_interp)
        if invalid_mask.float().sum() == 0:
            return target_for_interp

    if interp_mode == 'nearest' or interp_mode == 'BI':
        #print('1111111111111111')
        points = np.argwhere(mask_for_interp)  #np array
        #values = target_for_interp[mask_for_interp] #tensor
        values = target_for_interp[mask_for_interp].cpu().numpy() #np
        #print(type(points), type(values))
    else:
        print('222222222222222222')
        points = torch.where(mask_for_interp[0]) # tuple of 2 for (h, w) indices
        points = torch.cat([t.unsqueeze(0) for t in points]) # [2, number_of_points]
        points = points.permute(1,0) # shape: [number_of_points, 2]
        values = target_for_interp.clone()[mask_for_interp].view(mask_for_interp.shape[0],-1).permute(1,0) # shape: [number_of_points, num_classes]
    interp = interpolator(points, values) # return [num_classes, h, w]
    if interp_mode == 'nearest' or interp_mode == 'BI':
        #print('227')
        target_for_interp[invalid_mask] = torch.tensor(interp(np.argwhere(np.array(invalid_mask)))).float().cuda()
        #print('227 done')
    else:
        if not (interp.shape == target_for_interp.shape == invalid_mask.shape and interp.device == target_for_interp.device == invalid_mask.device):
            print('SHAPE: interp={}; target_for_interp={}; invalid_mask={}\n'.format(interp.shape, target_for_interp.shape, invalid_mask.shape))
            print('DEVICE: interp={}; target_for_interp={}; invalid_mask={}\n'.format(interp.device, target_for_interp.device, invalid_mask.device))
        try:
            target_for_interp[invalid_mask] = interp[torch.where(invalid_mask)].clone()
        except:
            print('interp: {}\n'.format(interp))
            print('invalid_mask: {}\n'.format(invalid_mask))
            print('target_for_interp: {}\n'.format(target_for_interp))
        else:
            pass
    return target_for_interp

def create_map(B, hidx_B, widx_B, height=80, width=80, radius=25, max_value=0.5, min_value=0.05):
    """
    Creates a map with shape (B, 1, height, width). Each batch has a point at (hidx, widx)
    with value max_value, decreasing in a cosine style to min_value within a radius.

    Parameters:
        B (int): Batch size.
        hidx_B (torch.Tensor): Tensor of shape (B, 1), y-coordinates.
        widx_B (torch.Tensor): Tensor of shape (B, 1), x-coordinates.
        height (int): Height of the map (default=80).
        width (int): Width of the map (default=80).
        radius (int): Radius within which the value decreases (default=25).
        max_value (float): Maximum value at the center (default=0.0025).
        min_value (float): Minimum value at the edge of the radius (default=2.5e-6).

    Returns:
        torch.Tensor: A map of shape (B, 1, height, width).
    """
    # Initialize the map with zeros
    map_tensor = torch.zeros((B, 1, height, width))+2.5e-6

    # Create a grid for distance calculation
    y_grid = torch.arange(height).view(1, height, 1).repeat(1, 1, width)
    x_grid = torch.arange(width).view(1, 1, width).repeat(1, height, 1)

    # Iterate over each batch
    for b in range(B):
        # Extract coordinates for the current batch
        h, w = hidx_B[b].item(), widx_B[b].item()

        # Compute the distance from the (h, w) point
        distance = torch.sqrt((y_grid - h)**2 + (x_grid - w)**2)

        # Cosine decay formula
        decay = 0.5 * (1 + torch.cos(torch.clamp(distance / radius * torch.pi, max=torch.pi)))
        #decay = torch.exp(-distance / radius)

        # Scale decay to the specified range [min_value, max_value]
        scaled_decay = min_value + (max_value - min_value) * decay
        map_tensor[b, 0] = torch.where(distance <= radius, scaled_decay, min_value)

    return map_tensor

def smooth_map_with_gaussian(map_tensor, sigma=3):
    """
    Smooths the input map using a Gaussian filter.

    Parameters:
        map_tensor (torch.Tensor): The map tensor of shape (B, 1, H, W).
        sigma (int): Standard deviation for the Gaussian filter.

    Returns:
        torch.Tensor: Smoothed map.
    """
    # Define the size of the Gaussian kernel based on sigma
    kernel_size = 2 * int(3 * sigma) + 1  # Covers ±3 standard deviations
    # Create 1D Gaussian kernel
    x = torch.arange(kernel_size) - kernel_size // 2
    gauss_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    gauss_1d = gauss_1d / gauss_1d.sum()  # Normalize

    # Create 2D Gaussian kernel
    gauss_2d = torch.outer(gauss_1d, gauss_1d).unsqueeze(0).unsqueeze(0)

    B = map_tensor.shape[0]
    gauss_2d = gauss_2d.expand(1, 1, kernel_size, kernel_size)
    gauss_2d = gauss_2d.to(map_tensor.device)

    # Apply Gaussian filter to each map
    smoothed_map = F.conv2d(map_tensor, gauss_2d, padding=kernel_size // 2, groups=1)
    return smoothed_map

class CompressNet(nn.Module):
    def __init__(self):
        super(CompressNet, self).__init__()
        # if cfg.MODEL.saliency_net == 'fovsimple':
        self.conv_last = nn.Conv2d(24,1,kernel_size=1,padding=0,stride=1)
        # else:
        # self.conv_last = nn.Conv2d(256,1,kernel_size=1,padding=0,stride=1)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.act(x)
        out = self.conv_last(x)
        return out

class SegmentationModuleBase(nn.Module):
    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

    def pixel_acc(self, pred_all, label_all):
        # accuracy with forground class
        torch.set_printoptions(threshold=10000)
        bs = pred_all.shape[0]
        acc_accu = 0.
        for i in range(bs):
            pred, label= pred_all[i:i+1, :,:], label_all[i:i+1, :, :] #pred: BxCxHxW  label: BxHxW
            #print('pred shape', pred.shape)
            _, preds = torch.max(pred, dim=1) #BxHxW
            #valid = (label > 0).long()     #bg is class 0
            #valid1 = (preds > 0).long()
            valid = (label < 50).long()
            valid1 = (preds < 50).long()
            acc_sum = torch.sum(valid * (preds == label).long())   # this is the intersectory pixels based on class
            #acc_sum = torch.sum(valid * (valid == valid1).long()) #binary intersection
            pixel_sum = torch.sum(valid)   # this is the summation of number of ground truth pixel
            pixel_sum1 = torch.sum(valid1)  # this is the summation of number of predicted true pixel
            pixel_sum_final = ((valid + valid1) > 0).sum().long()
            acc = acc_sum.float() / (pixel_sum_final.float() + 1e-10)
            acc_accu += acc
            #print(acc_sum.float(), pixel_sum.float(), pixel_sum_final.float(), pixel_sum1.float(), acc)   #tensor(6241., device='cuda:0') tensor(13009., device='cuda:0') tensor(21388., device='cuda:0') tensor(16672., device='cuda:0') tensor(0.2918, device='cuda:0')
        return acc_accu/bs
    
    def fg_bin_pixel_acc(self, pred_all, label_all):
        # accuracy with foreground binary
        torch.set_printoptions(threshold=10000)
        bs = pred_all.shape[0]
        acc_accu = 0.
        for i in range(bs):
            pred, label= pred_all[i:i+1, :,:], label_all[i:i+1, :, :] #pred: BxCxHxW  label: BxHxW
            #print('pred shape', pred.shape)
            _, preds = torch.max(pred, dim=1) #BxHxW
            #valid = (label > 0).long()
            #valid1 = (preds > 0).long()
            valid = (label < 50).long()
            valid1 = (preds < 50).long()
            acc_sum = torch.sum(valid * (valid == valid1).long())   # this is the intersectory pixels based on binary
            #acc_sum = torch.sum(valid * (valid == valid1).long()) #binary intersection
            pixel_sum = torch.sum(valid)   # this is the summation of number of ground truth pixel
            pixel_sum1 = torch.sum(valid1)  # this is the summation of number of predicted true pixel
            pixel_sum_final = ((valid + valid1)> 0).sum().long()
            acc = acc_sum.float() / (pixel_sum_final.float() + 1e-10)
            acc_accu += acc
            #print(acc_sum.float(), pixel_sum.float(), pixel_sum_final.float(), pixel_sum1.float(), acc)   #tensor(6241., device='cuda:0') tensor(13009., device='cuda:0') tensor(21388., device='cuda:0') tensor(16672., device='cuda:0') tensor(0.2918, device='cuda:0')
        return acc_accu/bs
    
    def fbg_cls_pixel_acc(self, pred_all, label_all):
        # accuracy with foreground binary
        torch.set_printoptions(threshold=10000)
        bs = pred_all.shape[0]
        acc_accu = 0.
        for i in range(bs):
            pred, label= pred_all[i:i+1, :,:], label_all[i:i+1, :, :] #pred: BxCxHxW  label: BxHxW
            _, preds = torch.max(pred, dim=1) #BxHxW
            #print(f'unique number in pred {torch.unique(preds)}')
            #print(f'unique number in label {torch.unique(label)}')
            pred_unique = torch.unique(preds)
            # if pred_unique.numel() > 1:
            #     pred_fg = pred_unique[pred_unique != 50].item()
            # else:
            #     pred_fg = pred_unique.item()
            #pred_fg = pred_unique[pred_unique != 50].item()
            label_unique = torch.unique(label)
            # if label_unique.numel() > 1:
            #     label_fg = label_unique[label_unique != 50].item()
            # else:
            #     label_fg = label_unique.item()
            #label_fg = label_unique[label_unique!= 50].item()
            #valid_fg = (label >= 0).long()
            #valid1_fg = (preds >= 0).long()
            valid_fg = (label < 50).long()
            valid1_fg = (preds < 50).long()
            acc_sum_fg = torch.sum(valid_fg * (label == preds).long())   # this is the intersectory pixels based on binary
            #acc_sum = torch.sum(valid * (valid == valid1).long()) #binary intersection
            pixel_sum_final_fg = ((valid_fg + valid1_fg)> 0).sum().long()
            acc_fg = acc_sum_fg.float() / (pixel_sum_final_fg.float() + 1e-10)

            valid_bg = (label == 50).long()
            valid1_bg = (preds== 50).long()
            acc_sum_bg = torch.sum(valid_bg * (label == preds).long())   # this is the intersectory pixels based on binary
            #acc_sum = torch.sum(valid * (valid == valid1).long()) #binary intersection
            pixel_sum_final_bg = ((valid_bg + valid1_bg)> 0).sum().long()
            acc_bg = acc_sum_bg.float() / (pixel_sum_final_bg.float() + 1e-10)

            acc_accu += acc_fg*0.5+acc_bg*0.5
            #acc_accu += torch.tensor(pred_fg == label_fg).float() 
            #print(acc_sum.float(), pixel_sum.float(), pixel_sum_final.float(), pixel_sum1.float(), acc)   #tensor(6241., device='cuda:0') tensor(13009., device='cuda:0') tensor(21388., device='cuda:0') tensor(16672., device='cuda:0') tensor(0.2918, device='cuda:0')
        return acc_accu/bs
    
    def fbg_bin_pixel_acc(self, pred_all, label_all):
        # accuracy with foreground binary
        torch.set_printoptions(threshold=10000)
        bs = pred_all.shape[0]
        acc_accu = 0.
        for i in range(bs):
            pred, label= pred_all[i:i+1, :,:], label_all[i:i+1, :, :] #pred: BxCxHxW  label: BxHxW
            #print('pred shape', pred.shape)
            _, preds = torch.max(pred, dim=1) #BxHxW
            valid_fg = (label < 50).long()
            valid1_fg = (preds < 50).long()
            acc_sum_fg = torch.sum(valid_fg * (valid_fg == valid1_fg).long())   # this is the intersectory pixels based on binary
            #acc_sum = torch.sum(valid * (valid == valid1).long()) #binary intersection
            pixel_sum_final_fg = ((valid_fg + valid1_fg)> 0).sum().long()
            acc_fg = acc_sum_fg.float() / (pixel_sum_final_fg.float() + 1e-10)
            
            valid_bg = (label == 50).long()
            valid1_bg = (preds == 50).long()
            acc_sum_bg = torch.sum(valid_bg * (valid_bg == valid1_bg).long())   # this is the intersectory pixels based on binary
            #acc_sum = torch.sum(valid * (valid == valid1).long()) #binary intersection
            pixel_sum_final_bg = ((valid_bg + valid1_bg)> 0).sum().long()
            acc_bg = acc_sum_bg.float() / (pixel_sum_final_bg.float() + 1e-10)
            acc_accu += acc_fg*0.5+acc_bg*0.5
            #print(acc_sum.float(), pixel_sum.float(), pixel_sum_final.float(), pixel_sum1.float(), acc)   #tensor(6241., device='cuda:0') tensor(13009., device='cuda:0') tensor(21388., device='cuda:0') tensor(16672., device='cuda:0') tensor(0.2918, device='cuda:0')
        return acc_accu/bs
    
class DeformSegmentationModule(nn.Module):
    def __init__(self, cfg):
        super(DeformSegmentationModule, self).__init__()
        # 使用轻量级 ViT 作为 backbone
        self.backbone = LightweightViT(dim=384, depth=10, heads=6)
        print("使用轻量级 ViT 作为 backbone")
        
        # 保留原来的网格大小设置
        self.grid_size_x = 20
        self.grid_size_y = 20
        self.padding_size_y = 10
        self.padding_size_x = 10
        
        # 分类器用于51类分类
        self.classifier = nn.Linear(384, 51)
        
        # 保存中间结果的变量
        self.xs = None  # 用于存储生成的 saliency map
        self.prediction = None  # 用于存储分类预测结果
        self.embeddings = None  # 用于存储所有 embeddings
        self.focus_embedding = None  # 用于存储注视点处的 embedding
    
    def forward(self, img_data, img_original, focus_point, s_bin_selected_BxHMxWMx1=None, segSize=None):
        """
        参数:
            img_data: 已经被tokenize的图像数据 [B, C, 40, 40]
            img_original: 原始图像
            focus_point: 注视点坐标 [B, 2]，范围在[0,1]，其中 focus_point[:, 0] 是 x 坐标，focus_point[:, 1] 是 y 坐标
            s_bin_selected_BxHMxWMx1: 训练时使用的掩码(当前代码不需要)
            segSize: 分割大小(当前代码不需要)
        返回:
            feature_map: 通过 differentiable_topk 选择的特征
        """
        batch_size = img_data.shape[0]
        
        # 1. 使用 backbone 提取特征
        # 注意：img_data 已经添加了位置编码，直接传入
        embeddings = self.backbone(img_data)  # [B, 40, 40, 384]
        #embeddings = embeddings / (embeddings.norm(dim=3, keepdim=True) + 1e-6)
        self.embeddings = embeddings  # 保存 embeddings 供后续使用
        
        # 2. 计算 saliency map
        # 将注视点坐标转换为网格索引，正确映射 x 和 y 坐标
        H, W = 40, 40
        focus_x = torch.clamp((focus_point[:, 0, 0, 0] / 640 * W).long(), 0, W-1)  # x 坐标
        focus_y = torch.clamp((focus_point[:, 0, 0, 1] / 640 * H).long(), 0, H-1)  # y 坐标
        
        # 初始化 saliency map
        saliency_map = torch.zeros(batch_size, H, W, device=img_data.device)
        
        # 对每个样本计算 saliency map
        for b in range(batch_size):
            # 获取注视点处的 embedding
            focus_embedding = embeddings[b, focus_y[b], focus_x[b], :]  # [384]
            
            # 计算所有位置的 embedding 与注视点 embedding 的余弦相似度
            # 重塑 embeddings 为 [H*W, 384]
            flat_embeddings = embeddings[b].reshape(-1, embeddings.shape[-1])
            
            # 计算相似度 (归一化后的点积)
            focus_embedding_norm = focus_embedding / (focus_embedding.norm() + 1e-6)
            flat_embeddings_norm = flat_embeddings / (flat_embeddings.norm(dim=1, keepdim=True) + 1e-6)
            similarity = torch.matmul(flat_embeddings_norm, focus_embedding_norm)
            
            # 重塑相似度为 [H, W]
            saliency_map[b] = similarity.reshape(H, W)
        
        # 保存 saliency map 供后续使用
        self.xs = saliency_map
        self.focus_embedding = focus_embedding
        self.focus_y = focus_y
        self.focus_x = focus_x
        
        # 3. 将 saliency map 转换格式并直接调用 differentiable_topk
        saliency_map_Bx1xHxW = saliency_map.unsqueeze(1)  # [B, 1, H, W]
        
        # 调用 differentiable_topk 直接选择重要区域
        selected_feature_map = differentiable_topk(
            img_data, 
            saliency_map_Bx1xHxW.detach(), 
            H_s=self.grid_size_y, 
            W_s=self.grid_size_x, 
            temperature=0.1
        )
        
        return selected_feature_map
    
    def make_prediction(self, mask=None):
        """
        使用选中的 embeddings 进行分类预测
        参数:
            mask: 用于选择 embeddings 的掩码 [B, H, W, 1]
        返回:
            predictions: 分类预测结果 [B, 51]
        """
        if mask is None:
            # 如果没有提供掩码，使用 saliency map 作为掩码
            mask = (self.xs > self.xs.mean(dim=[1, 2], keepdim=True)).float().unsqueeze(-1)
        
        batch_size = self.embeddings.shape[0]
        predictions = []
        # 存储每个样本中所有embedding的分类结果
        all_fg_embeddings_preds = []
        #all_bg_embeddings_preds = []
        all_embeddings_preds = []
        # 存储每个样本中选定的embedding数量
        num_embeddings_per_sample = []
        similarity_list = []
        
        for b in range(batch_size):
            # 使用掩码选择 embeddings
            current_mask = mask[b, :, :, 0]  # [H, W]
            selected_indices = torch.nonzero(current_mask > 0.2)  # [N, 2]
            un_selected_indices = torch.nonzero(current_mask <= 0.2)  # [N, 2]
            
            pred_all = self.classifier(self.embeddings[b].view(-1 ,self.embeddings.shape[-1]))
            all_embeddings_preds.append(pred_all)
            if len(selected_indices) == 0:
                selected_embedding = self.focus_embedding  # [384]
                # 对单个embedding进行分类
                pred = self.classifier(selected_embedding)  # [51]
                predictions.append(pred)
                # 添加单个embedding的预测结果
                all_fg_embeddings_preds.append(pred.unsqueeze(0))
                num_embeddings_per_sample.append(1)
                fg_embeddings = selected_embedding
                mask_unselect = torch.ones_like(current_mask).type(torch.bool)
                mask_unselect[self.focus_y[b], self.focus_x[b]] = 0
                bg_embeddings = self.embeddings[b].view(-1 ,self.embeddings.shape[-1])[mask_unselect.view(-1)]
                #pred_bg = self.classifier(bg_embeddings)
                #all_bg_embeddings_preds.append(pred_bg.unsqueeze(0))
            else:
                # 收集所有选定的embeddings
                selected_embeddings = torch.stack([
                    self.embeddings[b, idx[0], idx[1], :] 
                    for idx in selected_indices
                ])  # [N, 384]
                fg_embeddings = selected_embeddings.mean(dim=0)
                bg_embeddings = self.embeddings[b, un_selected_indices[:, 0], un_selected_indices[:, 1], :]
                #pred_bg = self.classifier(bg_embeddings)
                #all_bg_embeddings_preds.append(pred_bg)
                
                # 对每个embedding单独分类（矩阵形式，不使用循环）
                individual_preds = self.classifier(selected_embeddings)  # [N, 51]
                
                # 存储该样本中所有embedding的预测结果 (转为列表以保持接口一致)
                all_fg_embeddings_preds.append(individual_preds)
                num_embeddings_per_sample.append(len(selected_embeddings))
                
                # 计算平均预测结果
                pred = individual_preds.mean(dim=0)  # [51]
                predictions.append(pred)

            focus_embedding_norm = fg_embeddings / (fg_embeddings.norm() + 1e-6)
            flat_embeddings_norm = bg_embeddings / (bg_embeddings.norm(dim=1, keepdim=True) + 1e-6)
            similarity_list.append(torch.matmul(flat_embeddings_norm, focus_embedding_norm))
        
        # 将预测结果保存为 [B, 51] 格式
        self.prediction = torch.stack(predictions)
        # 保存每个样本中所有embedding的预测结果和数量信息
        self.all_fg_embeddings_predictions = all_fg_embeddings_preds
        self.all_embeddings_predictions = all_embeddings_preds
        #self.all_bg_embeddings_predictions = all_bg_embeddings_preds
        self.num_embeddings_per_sample = num_embeddings_per_sample
        
        return self.prediction, torch.cat(similarity_list)


class LightweightViT(nn.Module):
    """
    轻量级 Vision Transformer，使用 PyTorch 预定义组件
    """
    def __init__(self, dim=384, depth=3, heads=6, mlp_dim=None):
        super(LightweightViT, self).__init__()
        from torch.nn import TransformerEncoderLayer, TransformerEncoder
        
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim or dim * 4
        
        # 使用 PyTorch 的标准 TransformerEncoderLayer
        encoder_layer = TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=self.mlp_dim,
            batch_first=True,
            activation='gelu'
        )
        
        # 创建 TransformerEncoder
        self.transformer = TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 添加 Layer Norm
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        """
        参数:
            x: 输入张量 [B, C, H, W]
        返回:
            output: 输出特征 [B, H, W, C]
        """
        B, C, H, W = x.shape
        
        # 转置为 [B, H, W, C] 格式
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        
        # 保存原始形状
        orig_shape = x.shape
        
        # 重塑为序列格式进行处理 [B, H*W, C]
        x = x.reshape(B, H * W, C)
        
        # 通过 Transformer 层
        x = self.transformer(x)  # [B, H*W, C]
        
        # 应用最终的 Layer Norm
        x = self.norm(x)
        
        # 重塑回原始空间格式 [B, H, W, C]
        x = x.reshape(orig_shape)
        
        return x

class SegmentationModule(SegmentationModuleBase):
    def __init__(self, net_enc, net_dec, crit, cfg, deep_sup_scale=None, net_fov_res=None):
        super(SegmentationModule, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.crit = crit
        self.cfg = cfg
        self.deep_sup_scale = deep_sup_scale
        self.net_fov_res = net_fov_res

    # @torchsnooper.snoop()
    def forward(self, feed_dict, *, segSize=None, F_Xlr_acc_map=False, writer=None, count=None, feed_dict_info=None, feed_batch_count=None):
        # training
        if segSize is None:
            if self.deep_sup_scale is not None: # use deep supervision technique
                (pred, pred_deepsup) = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True))
            elif self.net_fov_res is not None:
                pred = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True), res=self.net_fov_res(feed_dict['img_data']))
            else:
                pred = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True))

            loss = self.crit(pred, feed_dict['seg_label'])
            if self.deep_sup_scale is not None:
                loss_deepsup = self.crit(pred_deepsup, feed_dict['seg_label'])
                loss = loss + loss_deepsup * self.deep_sup_scale

            acc = self.pixel_acc(pred, feed_dict['seg_label'])
            return loss, acc
        # inference
        else:
            if self.net_fov_res is not None:
                pred = self.decoder(self.encoder(feed_dict['img_data'].contiguous(), return_feature_maps=True), segSize=segSize, res=self.net_fov_res(feed_dict['img_data']))
            else:
                pred = self.decoder(self.encoder(feed_dict['img_data'].contiguous(), return_feature_maps=True), segSize=segSize)
            if self.cfg.VAL.write_pred:
                _, pred_print = torch.max(pred, dim=1)
                colors = loadmat('data/color150.mat')['colors']
                pred_color = colorEncode(as_numpy(pred_print.squeeze(0)), colors)
                pred_print = torch.from_numpy(pred_color.astype(np.uint8)).unsqueeze(0).permute(0,3,1,2)
                print('train/pred size: {}'.format(pred_print.shape))
                pred_print = vutils.make_grid(pred_print, normalize=False, scale_each=True)
                writer.add_image('train/pred', pred_print, count)

            if F_Xlr_acc_map:
                loss = self.crit(pred, feed_dict['seg_label'])
                return pred, loss
            else:
                return pred

class ModelBuilder:
    # custom weights initialization
    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
    @staticmethod
    def build_encoder(arch='resnet50', fc_dim=2048, weights='', dilate_rate=4):
        pretrained = True if len(weights) == 0 else False
        arch = arch.lower()
        #print(arch)  # hrnetv2_nodownsp
        #exit()
        if arch == 'mobilenetv2dilated':
            orig_mobilenet = mobilenet.__dict__['mobilenetv2'](pretrained=pretrained)
            net_encoder = MobileNetV2Dilated(orig_mobilenet, dilate_scale=dilate_rate)
        elif arch == 'resnet18':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet18dilated':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=dilate_rate)
        elif arch == 'resnet34':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet34dilated':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=dilate_rate)
        elif arch == 'resnet50':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet50dilated':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=dilate_rate)
        elif arch == 'resnet101':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet101dilated':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=dilate_rate)
        elif arch == 'resnext101':
            orig_resnext = resnext.__dict__['resnext101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnext) # we can still use class Resnet
        elif arch == 'hrnetv2_nodownsp':
            print('use hrnet!!!!!!!!!!!!!')
            net_encoder = hrnetv2_nodownsp.__dict__['hrnetv2_nodownsp'](pretrained=False)
        # elif arch == 'segformer':
        #     print('use segformer!!!!!!!!!!!!!')
        #     net_encoder = segformer.__dict__['segformer'](pretrained=False)
        else:
            raise Exception('Architecture undefined!')

        # encoders are usually pretrained
        # net_encoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print(f'here!!!!!!!!!!!!!!!!!!  Loading weights for net_encoder {weights}')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_encoder

    @staticmethod
    def build_decoder(arch='upernet',
                      fc_dim=2048, num_class=150,
                      weights='', use_softmax=False):
        arch = arch.lower()
        if arch == 'c1_deepsup':
            net_decoder = C1DeepSup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'c1':
            net_decoder = C1(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'ppm':
            net_decoder = PPM(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'ppm_deepsup':
            net_decoder = PPMDeepsup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'upernet_lite':
            net_decoder = UPerNet(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=256)
        elif arch == 'upernet':
            net_decoder = UPerNet(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=512)
        else:
            raise Exception('Architecture undefined!')

        net_decoder.apply(ModelBuilder.weights_init)
        # net_decoder.load_state_dict(
        #     torch.load('/root/autodl-tmp/lvis_Tin_80_80_ours_1_8_9_58pm_mse_dicefocal_kernel45_12w_largerhr/decoder_epoch_120.pth', map_location=lambda storage, loc: storage), strict=False)
        if len(weights) > 0:
            print('Loading weights for net_decoder')
            net_decoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_decoder

    # @staticmethod
def build_net_saliency():
    # define saliency network
    # Spatial transformer localization-network
    # if cfg.MODEL.track_running_stats:
    #     if cfg.MODEL.saliency_net == 'resnet18':
    #         net_saliency = saliency_network_resnet18()
    #     elif cfg.MODEL.saliency_net == 'resnet18_stride1':
    #         net_saliency = saliency_network_resnet18_stride1()
    #     elif cfg.MODEL.saliency_net == 'fovsimple':
    #         net_saliency = fov_simple(cfg)
    net_saliency = saliency_network_resnet18_stride1()
    # if len(weights) == 0:
    #     net_saliency.apply(ModelBuilder.weights_init)
    #     # net_saliency.load_state_dict(
    #     #     torch.load('/root/autodl-tmp/lvis_Tin_80_80_ours_1_8_9_58pm_mse_dicefocal_kernel45_12w_largerhr/saliency_epoch_120.pth', map_location=lambda storage, loc: storage), strict=False)
    # else:
    #     print('Loading weights for net_saliency')
    #     net_saliency.load_state_dict(
    #         torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
    return net_saliency

# @staticmethod
def build_net_compress():
    net_compress = CompressNet()

    return net_compress
