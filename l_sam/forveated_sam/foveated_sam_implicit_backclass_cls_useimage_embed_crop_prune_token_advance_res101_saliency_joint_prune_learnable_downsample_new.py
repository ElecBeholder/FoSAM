# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import traceback
from typing import Any, List, Tuple, Type

import torch
import torch.nn.functional as F
from utility.torch_tools import gen_grid_mtx_2xHxW
import numpy as np

from torch import nn, Tensor
from torchvision import transforms

import preset
from d_model.nn_A0_utils import init_weights_random
from d_model.nn_C1_mobilenetv2 import MobileNetV2
from utility.fctn import save_image

from utility.watch import watch_time
from l_sam.forveated_sam.efficient_sam_decoder import MaskDecoder, PromptEncoder
from l_sam.forveated_sam.efficient_sam_encoder_saliency_joint_prune_learnable_downsample_new import ImageEncoderViT
from l_sam.forveated_sam.two_way_transformer import TwoWayAttentionBlock, TwoWayTransformer
from DynamicFocus_foveal_seg_classification.models.models import build_net_compress, build_net_saliency, DeformSegmentationModule
from config import cfg
import torchvision.models as models


import torchvision.models as models

def get_custom_resnet(num_classes=51, in_channels=3):

    resnet = models.resnet18(pretrained=False)
    resnet.conv1 = nn.Conv2d(
        in_channels, 
        64, 
        kernel_size=7, 
        stride=2, 
        padding=3, 
        bias=False
    )

    resnet.fc = nn.Linear(512, num_classes)
    
    return resnet


class DualResNet(nn.Module):
    def __init__(self, num_classes=51, 
                 base_model='resnet18', 
                 feat_dim=512): 
        super().__init__()

        if base_model == 'resnet18':
            self.resnet1 = models.resnet18(pretrained=False)
        elif base_model == 'resnet34':
            self.resnet1 = models.resnet34(pretrained=False)
        elif base_model == 'resnet50':
            self.resnet1 = models.resnet50(pretrained=False)
            feat_dim = 2048
        elif base_model == 'resnet101':
            self.resnet1 = models.resnet101(pretrained=False)
            feat_dim = 2048
        else:
            raise NotImplementedError(f"Unsupported base model: {base_model}")
        
        self.resnet1.conv1 = nn.Conv2d(
            in_channels=258,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=True
        )
        self.resnet1.fc = nn.Identity()

        # ============ 分支2：输入 (5, 224, 224) ============
        if base_model == 'resnet18':
            self.resnet2 = models.resnet18(pretrained=False)
        elif base_model == 'resnet34':
            self.resnet2 = models.resnet34(pretrained=False)
        elif base_model == 'resnet50':
            self.resnet2 = models.resnet50(pretrained=False)
        elif base_model == 'resnet101':
            self.resnet2 = models.resnet101(pretrained=False)

        self.resnet2.conv1 = nn.Conv2d(
            in_channels=5,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=True
        )
        self.resnet2.fc = nn.Identity()


        self.classifier = nn.Linear(feat_dim * 2, num_classes)

    def forward(self, x1, x2):
        """
        x1.shape = [B, 256, 40, 40]
        x2.shape = [B,   5, 224, 224]
        """
        feat1 = self.resnet1(x1)  
        feat2 = self.resnet2(x2)  
        feats = torch.cat([feat1, feat2], dim=1)  
        out = self.classifier(feats)  # [B, num_classes=51]
        return out


def crop_and_resize_padding(batched_images, masks, size=80):
    B, C, H, W = batched_images.shape  
    cropped_images = []

    for i in range(B):

        mask = masks[i].squeeze(0)  # (H_mask, W_mask)
        image = batched_images[i]  # (C, H, W)

        mask = mask.unsqueeze(0).unsqueeze(0).float()  # (1, 1, H_mask, W_mask)
        mask_resized = F.interpolate(mask, size=(H, W), mode='bilinear').squeeze(0).squeeze(0)  # (H, W)

        coords = torch.nonzero(mask_resized > 0)  

        if coords.shape[0] == 0: 
            cropped_images.append(F.interpolate(image.unsqueeze(0), size=(size, size)))
            continue

        y_min = coords[:, 0].min()-50  
        x_min = coords[:, 1].min()-50  
        y_max = coords[:, 0].max()+50 
        x_max = coords[:, 1].max()+50  

        cropped = (image)[:, y_min:y_max + 1, x_min:x_max + 1]  # (C, h, w)

        h, w = cropped.shape[1], cropped.shape[2]
        if h > w:
            new_h = size
            new_w = int(w * (size / h))
        else:
            new_w = size
            new_h = int(h * (size / w))

        resized = F.interpolate(cropped.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)

        pad_h = (size - new_h) // 2
        pad_w = (size - new_w) // 2
        padding = (pad_w, size - new_w - pad_w, pad_h, size - new_h - pad_h)

        padded = F.pad(resized, padding, mode='constant', value=0)

        cropped_images.append(padded)

    return torch.cat(cropped_images, dim=0)  # (B, C, size, size)





class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
        
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        """
        in_channels：拼接后通道数（上采样后的 + encoder 对应层）
        """
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            # 使用反卷积
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(UNet, self).__init__()

        self.inc = DoubleConv(n_channels, 64)           # 640×640 -> 640×640
        self.down1 = Down(64, 128)                        # 640 -> 320
        self.down2 = Down(128, 256)                       # 320 -> 160
        self.down3 = Down(256, 512)                       # 160 -> 80
        self.down4 = Down(512, 1024)                      # 80 -> 40
        self.down5 = Down(1024, 2048)                     # 40 -> 20
        
        self.up1 = Up(2048 + 1024, 1024, bilinear)        # 20->40
        self.up2 = Up(1024 + 512, 512, bilinear)          # 40->80
        self.up3 = Up(512 + 256, 256, bilinear)           # 80->160
        self.up4 = Up(256 + 128, 128, bilinear)           # 160->320
        self.up5 = Up(128 + 64, 64, bilinear)             # 320->640
        
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        self.final_pool = nn.AdaptiveAvgPool2d((40, 40))
        
    def forward(self, x):
        x1 = self.inc(x)       # B x 64 x 640 x 640
        x2 = self.down1(x1)    # B x 128 x 320 x 320
        x3 = self.down2(x2)    # B x 256 x 160 x 160
        x4 = self.down3(x3)    # B x 512 x 80 x 80
        x5 = self.down4(x4)    # B x 1024 x 40 x 40
        x6 = self.down5(x5)    # B x 2048 x 20 x 20  
        
        x = self.up1(x6, x5)   # B x 1024 x 40 x 40
        x = self.up2(x, x4)    # B x 512 x 80 x 80
        x = self.up3(x, x3)    # B x 256 x 160 x 160
        x = self.up4(x, x2)    # B x 128 x 320 x 320
        x = self.up5(x, x1)    # B x 64 x 640 x 640
        
        x = self.outc(x)       # B x 1 x 640 x 640
        x = self.final_pool(x) # B x 1 x 20 x 20
        return x




def crop_and_resize_with_gaze(batched_images, masks, gaze_coords, size=80):
    B, C, H, W = batched_images.shape
    cropped_images = []
    new_gaze_coords = []

    for i in range(B):
        mask_i = masks[i]
        if mask_i.dim() == 3 and mask_i.size(0) == 1:
            mask_i = mask_i.squeeze(0)
        if mask_i.shape[0] != H or mask_i.shape[1] != W:
            mask_i = mask_i.unsqueeze(0).unsqueeze(0).float()  # [1, 1, H_m, W_m]
            mask_i = F.interpolate(mask_i, size=(H, W), mode='bilinear', align_corners=False)
            mask_i = mask_i.squeeze(0).squeeze(0)  # [H, W]

        coords = torch.nonzero(mask_i > 0)  # [N, 2]
        if coords.shape[0] == 0:
            fallback = F.interpolate(batched_images[i].unsqueeze(0), size=(size, size), mode='bilinear')
            cropped_images.append(fallback)
            gx, gy = gaze_coords[i]
            gx_new = gx * (size / W)
            gy_new = gy * (size / H)
            new_gaze_coords.append(torch.tensor([gx_new, gy_new]))
            continue

        y_min, x_min = coords[:, 0].min() - int(0.2 * (coords[:, 0].max() - coords[:, 0].min())), \
                       coords[:, 1].min() - int(0.2 * (coords[:, 1].max() - coords[:, 1].min()))
        y_max, x_max = coords[:, 0].max() + int(0.2 * (coords[:, 0].max() - coords[:, 0].min())), \
                       coords[:, 1].max() + int(0.2 * (coords[:, 1].max() - coords[:, 1].min()))

        bbox_h = (y_max - y_min + 1).item()
        bbox_w = (x_max - x_min + 1).item()
        center_y = (y_min + y_max).float() / 2.0
        center_x = (x_min + x_max).float() / 2.0

        side = max(bbox_h, bbox_w)
        half_side = side // 2

        new_y_min = int(center_y - half_side)
        new_x_min = int(center_x - half_side)
        new_y_max = new_y_min + side - 1
        new_x_max = new_x_min + side - 1

        if new_y_min < 0:
            new_y_min = 0
        if new_y_max >= H:
            new_y_max = H - 1
        if new_x_min < 0:
            new_x_min = 0
        if new_x_max >= W:
            new_x_max = W - 1

        image_i = batched_images[i]
        cropped = image_i[:, new_y_min:new_y_max + 1, new_x_min:new_x_max + 1]

        resized = F.interpolate(cropped.unsqueeze(0), size=(size, size), mode='bilinear', align_corners=False)
        cropped_images.append(resized)

        gx, gy = gaze_coords[i] 
        gx_cropped = gx - new_x_min
        gy_cropped = gy - new_y_min

        gx_new = gx_cropped * (size / side)
        gy_new = gy_cropped * (size / side)
        new_gaze_coords.append(torch.tensor([gx_new, gy_new]))

    return torch.cat(cropped_images, dim=0), torch.stack(new_gaze_coords, dim=0)



def generate_attention_mask(images, gaze_coords, crop_size=20):
    B, C, H, W = images.shape
    
    # 初始化全零掩码
    mask = torch.zeros((B, 1, H, W), dtype=torch.float32, device=images.device)

    for i in range(B):
        # 获取 gaze 归一化坐标，并缩放到图像尺寸
        x_g, y_g = gaze_coords[i]
        x_g = int(x_g // 16)  # 调整坐标到特征图尺度
        y_g = int(y_g // 16)

        # 计算窗口区域的左上角坐标
        x_start = max(0, min(x_g - crop_size // 2, W//16 - crop_size))
        y_start = max(0, min(y_g - crop_size // 2, H//16  - crop_size))

        # 计算窗口区域的右下角坐标
        x_end = x_start + crop_size
        y_end = y_start + crop_size

        # 在掩码上标记窗口区域为 1
        mask[i, 0, y_start:y_end, x_start:x_end] = 1
        mask = torch.nn.functional.adaptive_max_pool2d(mask, (20, 20)) 

    return mask


def token_pruning(images, gaze_coords, crop_size=20):
    B, C, H, W = images.shape
    
    cropped_images = []
    new_gaze_coords = []
    
    for i in range(B):
        x_g, y_g = gaze_coords[i]
        x_g = x_g//16
        y_g = y_g//16
        
        x_start = max(0, min(x_g - crop_size // 2, W - crop_size))
        y_start = max(0, min(y_g - crop_size // 2, H - crop_size))
        
        x_end = x_start + crop_size
        y_end = y_start + crop_size
        
        cropped_image = images[i, :, y_start:y_end, x_start:x_end]
        cropped_images.append(cropped_image)
        
        x_new = x_g - x_start
        y_new = y_g - y_start
        new_gaze_coords.append(torch.tensor([x_new, y_new]))
    
    cropped_images = torch.stack(cropped_images, dim=0)
    new_gaze_coords = torch.stack(new_gaze_coords, dim=0)
    
    return cropped_images, new_gaze_coords


def fuse_gaze_distance_encoder_output(x_BxCxHxW, x_Bx2):
    B, C, H, W = x_BxCxHxW.shape
    target_device = x_BxCxHxW.device

    downsample_factor_deformation = 1

    # H//8, W//8  640 -> 80
    HD = H // downsample_factor_deformation  # 64
    WD = W // downsample_factor_deformation  # 128
    # HS = H // self.downsample_factor
    # WS = W // self.downsample_factor

    max_dist = np.sqrt(H ** 2 + W ** 2)
    # initialize rgbaf 5 channegl
    x_ds_rgbaff_BxC2xHDxWD = torch.zeros(B, C + 2, HD, WD).to(dtype=torch.float32, device=x_BxCxHxW.device)

    # prepare focus map
    hidx_B = torch.clip(torch.round((x_Bx2[:, 0]/16) * (H - 1)), 0, W - 1)
    widx_B = torch.clip(torch.round((x_Bx2[:, 1]/16) * (W - 1)), 0, W - 1)
    grid_mtx_Bx2xHxW = gen_grid_mtx_2xHxW(H, W, device=target_device).unsqueeze(0).repeat(B, 1, 1, 1)
    dist_BxHxW = torch.sqrt((grid_mtx_Bx2xHxW[:, 0, :, :] - hidx_B[:, None, None]) ** 2 + (grid_mtx_Bx2xHxW[:, 1, :, :] - widx_B[:, None, None]) ** 2)

    # focus map with normalization distance
    # min_dist = dist_BxHxW.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
    # max_dist = dist_BxHxW.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]

    normalized_dist_BxHxW = 1.0 - dist_BxHxW / max_dist

    # Invert it so closest points are 1, farthest points are 0
    focusmap1_ds_Bx1xHxW = normalized_dist_BxHxW.unsqueeze(1)

    focusmap1_ds_Bx1xHDxWD = F.interpolate(focusmap1_ds_Bx1xHxW, (HD, WD), mode='bilinear', align_corners=True)
    focusmap2_ds_Bx1xHDxWD = torch.zeros_like(focusmap1_ds_Bx1xHDxWD)
    ds_max_indices = torch.argmax(focusmap1_ds_Bx1xHDxWD.view(B, -1), dim=1)
    focusmap2_ds_Bx1xHDxWD.view(B, -1).scatter_(1, ds_max_indices.unsqueeze(1), 1.0)

    # ave_pool_deformation = nn.AvgPool2d(kernel_size= downsample_factor_deformation, stride= downsample_factor_deformation, padding=0)

    # assign RGB color map
    # x_ds_rgbaff_BxC2xHDxWD[:, :-2, :, :] = ave_pool_deformation(x_BxCxHxW)
    x_ds_rgbaff_BxC2xHDxWD[:, :-2, :, :] = x_BxCxHxW

    x_ds_rgbaff_BxC2xHDxWD[:, -2, :, :] = focusmap1_ds_Bx1xHDxWD[:, 0, :, :]

    x_ds_rgbaff_BxC2xHDxWD[:, -1, :, :] = focusmap2_ds_Bx1xHDxWD[:, 0, :, :]

    # if self.square_focus:
    #     x_ds_rgbaff_BxC2xHDxWD[:, -2, :, :] = x_ds_rgbaff_BxC2xHDxWD[:, -2, :, :] ** 2

    return x_ds_rgbaff_BxC2xHDxWD


def fuse_gaze_distance_encoder_output_advance(x_BxCxHxW, x_Bx2):
    B, C, H, W = x_BxCxHxW.shape
    target_device = x_BxCxHxW.device

    downsample_factor_deformation = 1

    # H//8, W//8  640 -> 80
    HD = H // downsample_factor_deformation  # 64
    WD = W // downsample_factor_deformation  # 128
    # HS = H // self.downsample_factor
    # WS = W // self.downsample_factor

    max_dist = np.sqrt(H ** 2 + W ** 2)
    # initialize rgbaf 5 channegl
    x_ds_rgbaff_BxC2xHDxWD = torch.zeros(B, C + 2, HD, WD).to(dtype=torch.float32, device=x_BxCxHxW.device)

    # prepare focus map
    hidx_B = torch.clip(torch.round((x_Bx2[:, 0]) * (H - 1)), 0, W - 1)
    widx_B = torch.clip(torch.round((x_Bx2[:, 1]) * (W - 1)), 0, W - 1)
    grid_mtx_Bx2xHxW = gen_grid_mtx_2xHxW(H, W, device=target_device).unsqueeze(0).repeat(B, 1, 1, 1)
    dist_BxHxW = torch.sqrt((grid_mtx_Bx2xHxW[:, 0, :, :] - hidx_B[:, None, None]) ** 2 + (grid_mtx_Bx2xHxW[:, 1, :, :] - widx_B[:, None, None]) ** 2)

    # focus map with normalization distance
    # min_dist = dist_BxHxW.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
    # max_dist = dist_BxHxW.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]

    normalized_dist_BxHxW = 1.0 - dist_BxHxW / max_dist

    # Invert it so closest points are 1, farthest points are 0
    focusmap1_ds_Bx1xHxW = normalized_dist_BxHxW.unsqueeze(1)

    focusmap1_ds_Bx1xHDxWD = F.interpolate(focusmap1_ds_Bx1xHxW, (HD, WD), mode='bilinear', align_corners=True)
    focusmap2_ds_Bx1xHDxWD = torch.zeros_like(focusmap1_ds_Bx1xHDxWD)
    ds_max_indices = torch.argmax(focusmap1_ds_Bx1xHDxWD.view(B, -1), dim=1)
    focusmap2_ds_Bx1xHDxWD.view(B, -1).scatter_(1, ds_max_indices.unsqueeze(1), 1.0)

    # ave_pool_deformation = nn.AvgPool2d(kernel_size= downsample_factor_deformation, stride= downsample_factor_deformation, padding=0)

    # assign RGB color map
    # x_ds_rgbaff_BxC2xHDxWD[:, :-2, :, :] = ave_pool_deformation(x_BxCxHxW)
    x_ds_rgbaff_BxC2xHDxWD[:, :-2, :, :] = x_BxCxHxW

    x_ds_rgbaff_BxC2xHDxWD[:, -2, :, :] = focusmap1_ds_Bx1xHDxWD[:, 0, :, :]

    x_ds_rgbaff_BxC2xHDxWD[:, -1, :, :] = focusmap2_ds_Bx1xHDxWD[:, 0, :, :]

    # if self.square_focus:
    #     x_ds_rgbaff_BxC2xHDxWD[:, -2, :, :] = x_ds_rgbaff_BxC2xHDxWD[:, -2, :, :] ** 2

    return x_ds_rgbaff_BxC2xHDxWD

class FoveatedSam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
            self,
            image_encoder: ImageEncoderViT,
            prompt_encoder: PromptEncoder,
            decoder_max_num_input_points: int,
            mask_decoder: MaskDecoder,
            pixel_mean: List[float] = [0.485, 0.456, 0.406],
            pixel_std: List[float] = [0.229, 0.224, 0.225],
            class_num: int = 10,
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        # num_classes = 51
        # in_channels = 258
        self.classifier = DualResNet(num_classes=51, base_model='resnet101', feat_dim=2048)
        self.decoder_max_num_input_points = decoder_max_num_input_points
        self.mask_decoder = mask_decoder
        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(1, 3, 1, 1), False
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(pixel_std).view(1, 3, 1, 1), False
        )

        # self.classify_model = models.mobilenet_v2(pretrained=True)
        # self.classify_model.classifier[1] = nn.Linear(self.classify_model.last_channel, class_num)

    # @watch_time
    @torch.jit.export
    def predict_masks(
            self,
            image_embeddings: torch.Tensor,
            batched_points: torch.Tensor,
            batched_point_labels: torch.Tensor,
            multimask_output: bool,
            input_h: int,
            input_w: int,
            output_h: int = -1,
            output_w: int = -1,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Predicts masks given image embeddings and prompts. This only runs the decoder.

        Arguments:
          image_embeddings: A tensor of shape [B, C, H, W] or [B*max_num_queries, C, H, W]
          batched_points: A tensor of shape [B, max_num_queries, num_pts, 2]
          batched_point_labels: A tensor of shape [B, max_num_queries, num_pts]
        Returns:
          A tuple of two tensors:
            low_res_mask: A tensor of shape [B, max_num_queries, 256, 256] of predicted masks
            iou_predictions: A tensor of shape [B, max_num_queries] of estimated IOU scores
        """

        batch_size, max_num_queries, num_pts, _ = batched_points.shape
        num_pts = batched_points.shape[2]
        rescaled_batched_points = self.get_rescaled_pts(batched_points, input_h, input_w)

        if num_pts > self.decoder_max_num_input_points:
            rescaled_batched_points = rescaled_batched_points[
                                      :, :, : self.decoder_max_num_input_points, :
                                      ]
            batched_point_labels = batched_point_labels[
                                   :, :, : self.decoder_max_num_input_points
                                   ]
        elif num_pts < self.decoder_max_num_input_points:
            rescaled_batched_points = F.pad(
                rescaled_batched_points,
                (0, 0, 0, self.decoder_max_num_input_points - num_pts),
                value=-1.0,
            )
            batched_point_labels = F.pad(
                batched_point_labels,
                (0, self.decoder_max_num_input_points - num_pts),
                value=-1.0,
            )

        sparse_embeddings = self.prompt_encoder(
            rescaled_batched_points.reshape(
                batch_size * max_num_queries, self.decoder_max_num_input_points, 2
            ),
            batched_point_labels.reshape(
                batch_size * max_num_queries, self.decoder_max_num_input_points
            ),
        )
        sparse_embeddings = sparse_embeddings.view(
            batch_size,
            max_num_queries,
            sparse_embeddings.shape[1],
            sparse_embeddings.shape[2],
        )

        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings,
            self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            multimask_output=multimask_output,
        )
        _, num_predictions, low_res_size, _ = low_res_masks.shape

        if output_w > 0 and output_h > 0:
            output_masks = F.interpolate(
                low_res_masks, (output_h, output_w), mode="bicubic"
            )
            output_masks = torch.reshape(
                output_masks,
                (batch_size, max_num_queries, num_predictions, output_h, output_w),
            )
        else:
            output_masks = torch.reshape(
                low_res_masks,
                (
                    batch_size,
                    max_num_queries,
                    num_predictions,
                    low_res_size,
                    low_res_size,
                ),
            )

        iou_predictions = torch.reshape(
            iou_predictions, (batch_size, max_num_queries, -1)
        )
        return output_masks, iou_predictions

    def get_rescaled_pts(self, batched_points: torch.Tensor, input_h: int, input_w: int):
        return torch.stack(
            [
                torch.where(
                    batched_points[..., 0] >= 0,
                    batched_points[..., 0] * self.image_encoder.img_size / input_w,
                    -1.0,
                ),
                torch.where(
                    batched_points[..., 1] >= 0,
                    batched_points[..., 1] * self.image_encoder.img_size / input_h,
                    -1.0,
                ),
            ],
            dim=-1,
        )

    # @watch_time
    @torch.jit.export
    def get_image_embeddings(self, batched_images, batched_points, s_bin_selected_BxHMxWMx1) -> torch.Tensor:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_images: A tensor of shape [B, 3, H, W]
        Returns:
          List of image embeddings each of of shape [B, C(i), H(i), W(i)].
          The last embedding corresponds to the final layer.
        """
        batched_images = self.preprocess(batched_images)
        return self.image_encoder(batched_images, batched_points, s_bin_selected_BxHMxWMx1)

    def forward(
            self,
            batched_images: torch.Tensor,
            batched_points: torch.Tensor,
            batched_point_labels: torch.Tensor,
            scale_to_original_image_size: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_images: A tensor of shape [B, 3, H, W]
          batched_points: A tensor of shape [B, num_queries, max_num_pts, 2]
          batched_point_labels: A tensor of shape [B, num_queries, max_num_pts]

        Returns:
          A list tuples of two tensors where the ith element is by considering the first i+1 points.
            low_res_mask: A tensor of shape [B, 256, 256] of predicted masks
            iou_predictions: A tensor of shape [B, max_num_queries] of estimated IOU scores
        """
        batch_size, _, input_h, input_w = batched_images.shape

        s_bin_selected_BxHMxWMx1 = generate_attention_mask(batched_images, batched_points.squeeze(1).squeeze(1), 20)

        image_embeddings = self.get_image_embeddings(batched_images, batched_points, s_bin_selected_BxHMxWMx1)

        masks, iou_pred = self.predict_masks(
            image_embeddings,
            batched_points,
            batched_point_labels,
            multimask_output=True,
            input_h=input_h,
            input_w=input_w,
            output_h=input_h if scale_to_original_image_size else -1,
            output_w=input_w if scale_to_original_image_size else -1,
        )


        image_embeddings, points_1 = token_pruning(image_embeddings, batched_points.squeeze(1).squeeze(1), crop_size=20)
        fusion_emd = fuse_gaze_distance_encoder_output_advance(image_embeddings.cuda(), points_1.cuda())

        mean_global = torch.mean(fusion_emd, dim=(0, 2, 3))  
        std_global = torch.std(fusion_emd, dim=(0, 2, 3), unbiased=False) 

        # preprocess = transforms.Compose([
        #     transforms.Resize(256),
        #     transforms.CenterCrop(224),
        #     transforms.Normalize(mean=mean_global.tolist(), std=std_global.tolist())
        # ])
        epsilon = 1e-7
        std_fixed = [s if s > 0 else epsilon for s in std_global.tolist()]
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=mean_global.tolist(), std=std_fixed)
        ])

        fusion_emd = preprocess(fusion_emd)




        image_BxCxHxW, points_2 = crop_and_resize_with_gaze(batched_images, masks.squeeze(1).squeeze(1),batched_points.squeeze(1).squeeze(1), 256)
        fusion_crop = fuse_gaze_distance_encoder_output_advance(image_BxCxHxW.cuda(), points_2.cuda())

        mean_global = torch.mean(fusion_crop, dim=(0, 2, 3)) 
        std_global = torch.std(fusion_crop, dim=(0, 2, 3), unbiased=False)  


        epsilon = 1e-7
        std_fixed = [s if s > 0 else epsilon for s in std_global.tolist()]
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=mean_global.tolist(), std=std_fixed)
        ])

        fusion_crop = preprocess(fusion_crop)



        cls_pred = self.classifier(fusion_emd, fusion_crop)


        return masks, cls_pred, iou_pred

    # @watch_time
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        if (
                x.shape[2] != self.image_encoder.img_size
                or x.shape[3] != self.image_encoder.img_size
        ):
            x = F.interpolate(
                x,
                (self.image_encoder.img_size, self.image_encoder.img_size),
                mode="bilinear",
            )
        return (x - self.pixel_mean) / self.pixel_std


def build_foveated_sam(img_size, encoder_patch_embed_dim, encoder_num_heads, num_multimask_outputs=3, class_num=10, checkpoint_state_dict=None):
    encoder_patch_size = 16
    encoder_depth = 12
    encoder_mlp_ratio = 4.0
    encoder_neck_dims = [256, 256]
    decoder_max_num_input_points = 1
    decoder_transformer_depth = 2
    decoder_transformer_mlp_dim = 2048
    decoder_num_heads = 8
    decoder_upscaling_layer_dims = [64, 32]
    num_multimask_outputs = num_multimask_outputs
    iou_head_depth = 3
    iou_head_hidden_dim = 256
    activation = "gelu"
    normalization_type = "layer_norm"
    normalize_before_activation = False

    assert activation == "relu" or activation == "gelu"
    if activation == "relu":
        activation_fn = nn.ReLU
    else:
        activation_fn = nn.GELU

    image_encoder = ImageEncoderViT(
        img_size=img_size,
        patch_size=encoder_patch_size,
        in_chans=3,
        patch_embed_dim=encoder_patch_embed_dim,
        normalization_type=normalization_type,
        depth=encoder_depth,
        num_heads=encoder_num_heads,
        mlp_ratio=encoder_mlp_ratio,
        neck_dims=encoder_neck_dims,
        act_layer=activation_fn,
    )

    image_embedding_size = image_encoder.image_embedding_size
    encoder_transformer_output_dim = image_encoder.transformer_output_dim

    sam = FoveatedSam(
        image_encoder=image_encoder,
        prompt_encoder=PromptEncoder(
            embed_dim=encoder_transformer_output_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(img_size, img_size),
        ),
        decoder_max_num_input_points=decoder_max_num_input_points,
        mask_decoder=MaskDecoder(
            transformer_dim=encoder_transformer_output_dim,
            transformer=TwoWayTransformer(
                depth=decoder_transformer_depth,
                embedding_dim=encoder_transformer_output_dim,
                num_heads=decoder_num_heads,
                mlp_dim=decoder_transformer_mlp_dim,
                activation=activation_fn,
                normalize_before_activation=normalize_before_activation,
            ),
            num_multimask_outputs=num_multimask_outputs,
            activation=activation_fn,
            normalization_type=normalization_type,
            normalize_before_activation=normalize_before_activation,
            iou_head_depth=iou_head_depth - 1,
            iou_head_hidden_dim=iou_head_hidden_dim,
            upscaling_layer_dims=decoder_upscaling_layer_dims,
            class_num=class_num

        ),
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
        class_num=class_num,
    )
    if checkpoint_state_dict is not None:
        sam.load_state_dict(checkpoint_state_dict, strict=False)

    return sam
