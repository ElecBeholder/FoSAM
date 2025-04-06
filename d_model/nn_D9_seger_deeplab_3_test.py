from pprint import pprint
from typing import Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torchvision import transforms

from d_model.nn_A0_utils import RAM
from d_model.nn_B0_deformed_sampler import get_grid_Bx2xHSxWS, deformed_unsampler, int_rount_scale_grid, gaussian_kernel
from d_model.nn_B1_croper import get_idxs_crop4
from utility.plot_tools import plt_show, plt_imgshow, plt_multi_imgshow

from utility.torch_tools import gen_grid_mtx_2xHxW
from d_model.nn_B4_Unet6 import UNet6
from d_model.nn_B11_deeplab import CustomDeepLab
from torch.nn.functional import conv2d


class SegerZoomDeep(nn.Module):

    def __init__(self,
                 base_module_deformation: Type[nn.Module],
                 base_module: Type[nn.Module],
                 in_channels=4,
                 out_channels=1,
                 class_num=2,
                 downsample_factor=4,
                 downsample_factor_deformation=8,
                 kernel_gridsp=40 + 1,
                 kernel_gblur=40 + 1,
                 kernel_priori=20 + 1,
                 cbase_seg0=4,
                 cbase_seg=64,
                 nlayer_seg0=2,
                 nlayer_seg=4,
                 priori=True,
                 square_focus=True):
        super(SegerZoomDeep, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = out_channels

        self.downsample_factor = downsample_factor
        self.downsample_factor_deformation = downsample_factor_deformation

        self.kernel_gridsp = kernel_gridsp

        self.pad_gridsp = self.kernel_gridsp // 2

        self.gen_seg_0 = base_module_deformation(in_channels=in_channels + 2, out_channels=out_channels, base_channels=cbase_seg0, layer_num=nlayer_seg0)

        self.gen_seg = CustomDeepLab(num_input_channels = in_channels +2, num_classes=class_num)

        self.ave_pool = nn.AvgPool2d(kernel_size=self.downsample_factor, stride=self.downsample_factor, padding=0)
        self.ave_pool_deformation = nn.AvgPool2d(kernel_size=self.downsample_factor_deformation, stride=self.downsample_factor_deformation, padding=0)
        self.max_pool = nn.MaxPool2d(kernel_size=self.downsample_factor, stride=self.downsample_factor, padding=0)
        self.max_pool_deformation = nn.MaxPool2d(kernel_size=self.downsample_factor_deformation, stride=self.downsample_factor_deformation, padding=0)

        sigma = kernel_gblur // 2
        self.gaussian_blur = transforms.GaussianBlur(kernel_size=kernel_gblur, sigma=(sigma, sigma))

        self.priori = priori
        self.kernel_priori = kernel_priori
        self.square_focus = square_focus

        self.transform_rgbaff = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.55, 0.00097], std=[0.229, 0.224, 0.225, 0.20, 0.0312362])  # 针对 ImageNet 进行标准化
            ])

        self.init_value = 3.0
        priori_gaussian_1xKxK = gaussian_kernel(size=self.kernel_priori, sigma=self.kernel_priori // 6)[None, :, :]
        p_max = torch.amax(priori_gaussian_1xKxK, dim=[-1, -2], keepdim=True)
        self.priori_gaussian_1xKxK = priori_gaussian_1xKxK / p_max

    def forward(self, x_BxCxHxW: torch.Tensor, x_Bx2: torch.Tensor):
        B, C, H, W = x_BxCxHxW.shape
        target_device = x_BxCxHxW.device

        # H//8, W//8  640 -> 80
        HD = H // self.downsample_factor_deformation  # 64
        WD = W // self.downsample_factor_deformation  # 128
        HS = H // self.downsample_factor
        WS = W // self.downsample_factor

        max_dist = np.sqrt(HD ** 2 + WD ** 2)
        x_ds_rgbaff_BxC2xHDxWD = torch.zeros(B, C + 2, HD, WD).to(dtype=torch.float32, device=x_BxCxHxW.device)

        hidx_B = torch.clip(torch.round(x_Bx2[:, 0] * (H - 1)), 0, W - 1)
        widx_B = torch.clip(torch.round(x_Bx2[:, 1] * (W - 1)), 0, W - 1)
        grid_mtx_Bx2xHxW = gen_grid_mtx_2xHxW(H, W, device=target_device).unsqueeze(0).repeat(B, 1, 1, 1)
        dist_BxHxW = torch.sqrt((grid_mtx_Bx2xHxW[:, 0, :, :] - hidx_B[:, None, None]) ** 2 + (grid_mtx_Bx2xHxW[:, 1, :, :] - widx_B[:, None, None]) ** 2)

        min_dist = dist_BxHxW.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        max_dist = dist_BxHxW.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        normalized_dist_BxHxW = (dist_BxHxW - min_dist) / (max_dist - min_dist + 1e-8)

        normalized_dist_BxHxW = 1.0 - normalized_dist_BxHxW / max_dist

        focusmap1_ds_Bx1xHxW = normalized_dist_BxHxW.unsqueeze(1)

        focusmap1_ds_Bx1xHDxWD = F.interpolate(focusmap1_ds_Bx1xHxW, (HD, WD), mode='bilinear', align_corners=True)
        focusmap2_ds_Bx1xHDxWD = torch.zeros_like(focusmap1_ds_Bx1xHDxWD)
        ds_max_indices = torch.argmax(focusmap1_ds_Bx1xHDxWD.view(B, -1), dim=1)
        focusmap2_ds_Bx1xHDxWD.view(B, -1).scatter_(1, ds_max_indices.unsqueeze(1), 1.0)

        x_ds_rgbaff_BxC2xHDxWD[:, :-2, :, :] = self.downsample_x_avepool_real_BxCxHDxWD(x_BxCxHxW)

        x_ds_rgbaff_BxC2xHDxWD[:, -2, :, :] = focusmap1_ds_Bx1xHDxWD[:, 0, :, :]

        x_ds_rgbaff_BxC2xHDxWD[:, -1, :, :] = focusmap2_ds_Bx1xHDxWD[:, 0, :, :]

        out0_Bx1xHDxWD = self.gen_seg_0(self.transform_rgbaff(x_ds_rgbaff_BxC2xHDxWD))


        dmap_pred_ds_Bx1xHDxWD = torch.sigmoid(out0_Bx1xHDxWD)

        dm_v_Bx1xHDPxWDP = F.pad(dmap_pred_ds_Bx1xHDxWD, (self.pad_gridsp, self.pad_gridsp, self.pad_gridsp, self.pad_gridsp), mode='replicate')

        grid_pred_Bx2xHDxWD = get_grid_Bx2xHSxWS(dm_v_Bx1xHDPxWDP, HD, WD, kernel_size=self.kernel_gridsp)

        grid_pred_Bx2xHSxWS = F.interpolate(grid_pred_Bx2xHDxWD, (HS, WS), mode='bilinear', align_corners=True)

        grid_pred_BxHSxWSx2 = torch.flip(grid_pred_Bx2xHSxWS.permute(0, 2, 3, 1), dims=[-1])  # prepare shape and order of grid

        x_gs_rgbaff_BxC2xHSxWS = torch.zeros(B, C + 2, HS, WS).to(dtype=torch.float32, device=x_BxCxHxW.device)
        x_gs_rgbaff_BxC2xHSxWS[:, :-2, :, :] = F.grid_sample(x_BxCxHxW, grid_pred_BxHSxWSx2, mode='bilinear', align_corners=True)

        focusmap1_gs_Bx1xHSxWS = F.grid_sample(focusmap1_ds_Bx1xHxW, grid_pred_BxHSxWSx2, mode='bilinear', align_corners=True)
        focusmap2_gs_Bx1xHSxWS = torch.zeros_like(focusmap1_gs_Bx1xHSxWS)
        gs_max_indices = torch.argmax(focusmap1_gs_Bx1xHSxWS.view(B, -1), dim=1)
        focusmap2_gs_Bx1xHSxWS.view(B, -1).scatter_(1, gs_max_indices.unsqueeze(1), 1.0)

        x_gs_rgbaff_BxC2xHSxWS[:, -2, :, :] = focusmap1_gs_Bx1xHSxWS[:, 0, :, :]
        x_gs_rgbaff_BxC2xHSxWS[:, -1, :, :] = focusmap2_gs_Bx1xHSxWS[:, 0, :, :]

        out1_Bx1xHSxWS, y_pred_BxK = self.gen_seg(self.transform_rgbaff(x_gs_rgbaff_BxC2xHSxWS))

        y_pred_gs_Bx1xHSxWS = torch.sigmoid(out1_Bx1xHSxWS)

        return y_pred_gs_Bx1xHSxWS, grid_pred_BxHSxWSx2, dmap_pred_ds_Bx1xHDxWD, x_ds_rgbaff_BxC2xHDxWD, x_gs_rgbaff_BxC2xHSxWS, y_pred_BxK

    def downsample_x_avepool_real_BxCxHDxWD(self, x_BxCxHxW: torch.Tensor):
        # B, C, H, W = x_BxCxHxW.shape
        # HD = H // self.downsample_factor_deformation
        # WD = W // self.downsample_factor_deformation
        # return F.interpolate(x_BxCxHxW, (HD, WD), mode='bilinear', align_corners=True)
        return self.ave_pool_deformation(x_BxCxHxW)

    def downsample_y_avepool_real_Bx1xHDxWD(self, y_real_Bx1xHxW: torch.Tensor):
        # B, C, H, W = y_real_Bx1xHxW.shape
        # HD = H // self.downsample_factor_deformation
        # WD = W // self.downsample_factor_deformation
        # return F.interpolate(y_real_Bx1xHxW, (HD, WD), mode='bilinear', align_corners=True)
        return self.ave_pool_deformation(y_real_Bx1xHxW)
    
    def downsample_y_gridsp_real_Bx1xHSxWS(self, y_real_Bx1xHxW: torch.Tensor, grid_pred_BxHSxWSx2: torch.Tensor):
        return F.grid_sample(y_real_Bx1xHxW, grid_pred_BxHSxWSx2, mode='bilinear', align_corners=True)

    def output_y_pred_Bx1xHxW(self, y_pred_Bx1xHSxWS: torch.Tensor, grid_pred_BxHSxWSx2: torch.Tensor, x_BxCxHxW: torch.Tensor):
        B, _, HS, WS = y_pred_Bx1xHSxWS.shape
        H = HS * self.downsample_factor
        W = WS * self.downsample_factor

        grid_pred_BxHSxWSx2 = torch.flip(grid_pred_BxHSxWSx2, dims=[-1])
        grid_pred_Bx2xHSxWS = grid_pred_BxHSxWSx2.permute(0, 3, 1, 2)
        grid_pred_Bx2xHSxWS = int_rount_scale_grid(grid_pred_Bx2xHSxWS, H, W)

        return deformed_unsampler(y_pred_Bx1xHSxWS, grid_pred_Bx2xHSxWS, H, W) 

class BinarizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output 

import torchvision

def transform_cls_tensor(tensor):
    tensor = torch.nn.functional.interpolate(tensor, size=64, mode='bilinear', align_corners=False)
    tensor = torchvision.transforms.functional.center_crop(tensor, output_size=56) 
    tensor = torchvision.transforms.functional.normalize(tensor, mean=[0.485, 0.456, 0.406, 0.55, 0.00097], std=[0.229, 0.224, 0.225, 0.20, 0.0312362])
    return tensor


if __name__ == '__main__':
    pass
    in_channels = 3
    out_channels = 41
    canvas_H = 256
    canvas_W = 512
    sample_factor = 4
    kernel_size = 64 + 1
    B = 6
    injectD = 2

    x_BxCxHxW = torch.randn(B, in_channels, canvas_H, canvas_W)
    x_Bx2 = torch.randn(B, 2)
    e2e = SegerZoom(in_channels=in_channels, out_channels=out_channels, H=canvas_H, W=canvas_W, downsample_factor=sample_factor)

    ys_BxKxHSxWS, grid_BxHSxWSx2 = e2e(x_BxCxHxW, x_Bx2)

    # plt_multi_imgshow(imgs=[label_x_BxKxHxW[0, :, :, 0], label_x_BxKxHxW[0, :, :, 1]], titles=['w', 'h'], row_col=(2, 1))
    # plt.show(block=True)
