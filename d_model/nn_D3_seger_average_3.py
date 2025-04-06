from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from d_model.nn_A0_utils import try_gpu
from utility.torch_tools import gen_grid_mtx_2xHxW
from d_model.nn_B10_SegVit import CustomSegformerWithLoRA, CustomSegformer
from d_model.nn_B11_deeplab import CustomDeepLab
from d_model.nn_B14_pspnet import CustomPSPNet
from d_model.nn_B15_hrnet import CustomHRNet
from transformers import SegformerForSemanticSegmentation, SegformerConfig, AutoImageProcessor

class SegerAverage(nn.Module):
    def __init__(self, base_module: Type[nn.Module], classify_module: Type[nn.Module], class_num=2, in_channels=4, out_channels=1, downsample_factor=16, cbase_seg=32, nlayer_seg=4, seg_module=''):
        super(SegerAverage, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = out_channels

        self.downsample_factor = downsample_factor
        
        if seg_module == 'deeplab':
            self.gen_seg = CustomDeepLab(num_input_channels=in_channels+2)
        elif seg_module == 'segformer_4':
            config = SegformerConfig.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")
            config.num_labels = 1  
            self.gen_seg = CustomSegformer(config=config, num_input=in_channels+2)
        elif seg_module == 'segformer_5':
            config = SegformerConfig.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
            config.num_labels = 1  
            self.gen_seg = CustomSegformer(config=config, num_input=in_channels+2)
        elif seg_module == 'pspnet':
            self.gen_seg = CustomPSPNet(input_channels=in_channels+2, num_classes=out_channels)
        elif seg_module == 'hrnet':
            self.gen_seg = CustomHRNet(in_channels=in_channels+2,num_classes=out_channels)
        # self.gen_seg = CustomDeepLab()

        self.ave_pool = nn.AvgPool2d(kernel_size=self.downsample_factor, stride=self.downsample_factor, padding=0)
        # self.ave_pool_deformation = nn.AvgPool2d(kernel_size=self.downsample_factor_deformation, stride=self.downsample_factor_deformation, padding=0)
        self.max_pool = nn.MaxPool2d(kernel_size=self.downsample_factor, stride=self.downsample_factor, padding=0)
        # self.max_pool_deformation = nn.MaxPool2d(kernel_size=self.downsample_factor_deformation, stride=self.downsample_factor_deformation, padding=0)
        
        self.gen_cls = classify_module(in_channels=in_channels - 1 + 3, out_channels=class_num)

        self.transform_rgbaff = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.55, 0.00097], std=[0.229, 0.224, 0.225, 0.20, 0.0312362])  # 针对 ImageNet 进行标准化
        ])

    def forward(self, *data, method='forward'):
        if method == 'forward':
            x_BxCxHxW, x_Bx2 = data
            B, C, H, W = x_BxCxHxW.shape

            HS = H // self.downsample_factor  # 64
            WS = W // self.downsample_factor  # 128

            target_device = x_BxCxHxW.device

            x_ds_rgbf_BxC2xHSxWS = torch.zeros(B, C + 2, HS, WS).to(dtype=torch.float32, device=x_BxCxHxW.device)

            # prepare focus map
            hidx_B = torch.clip(torch.round(x_Bx2[:, 0] * (H - 1)), 0, W - 1)
            widx_B = torch.clip(torch.round(x_Bx2[:, 1] * (W - 1)), 0, W - 1)
            grid_mtx_Bx2xHxW = gen_grid_mtx_2xHxW(H, W, device=target_device).unsqueeze(0).repeat(B, 1, 1, 1)
            dist_BxHxW = torch.sqrt((grid_mtx_Bx2xHxW[:, 0, :, :] - hidx_B[:, None, None]) ** 2 + (grid_mtx_Bx2xHxW[:, 1, :, :] - widx_B[:, None, None]) ** 2)
            
            min_dist = dist_BxHxW.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
            max_dist = dist_BxHxW.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]

            normalized_dist_BxHxW = (dist_BxHxW - min_dist) / (max_dist - min_dist + 1e-8)
            normalized_dist_BxHxW = 1.0 - dist_BxHxW / max_dist


            focusmap1_ds_Bx1xHxW = normalized_dist_BxHxW.unsqueeze(1)
            focusmap1_ds_Bx1xHDxWD = F.interpolate(focusmap1_ds_Bx1xHxW, (HS, WS), mode='bilinear', align_corners=True)
            focusmap2_ds_Bx1xHDxWD = torch.zeros_like(focusmap1_ds_Bx1xHDxWD)

            ds_max_indices = torch.argmax(focusmap1_ds_Bx1xHDxWD.view(B, -1), dim=1)
            focusmap2_ds_Bx1xHDxWD.view(B, -1).scatter_(1, ds_max_indices.unsqueeze(1), 1.0)


            # assign RGB color map
            x_ds_rgbf_BxC2xHSxWS[:, :-2, :, :] = self.downsample_x_real_BxCxHSxWS(x_BxCxHxW)
            # assign focus map
            x_ds_rgbf_BxC2xHSxWS[:, -2, :, :] = focusmap1_ds_Bx1xHDxWD[:, 0, :, :]
            x_ds_rgbf_BxC2xHSxWS[:, -1, :, :] = focusmap2_ds_Bx1xHDxWD[:, 0, :, :]

            # gen density map
            y_pred_ds_Bx1xHSxWS = torch.sigmoid(self.gen_seg(self.transform_rgbaff(x_ds_rgbf_BxC2xHSxWS)))

            y_pred_BxK = self.gen_cls(self.transform_rgbaff(x_ds_rgbf_BxC2xHSxWS) * y_pred_ds_Bx1xHSxWS)

            return y_pred_ds_Bx1xHSxWS, x_ds_rgbf_BxC2xHSxWS, y_pred_BxK

        elif method == 'downsample_x_real_BxCxHSxWS':
            x_BxCxHxW, = data
            return self.downsample_x_real_BxCxHSxWS(x_BxCxHxW)

        elif method == 'downsample_y_maxpool_real_Bx1xHSxWS':
            y_real_Bx1xHxW, = data
            return self.downsample_y_maxpool_real_Bx1xHSxWS(y_real_Bx1xHxW)

        elif method == 'output_y_pred_Bx1xHxW':
            y_pred_Bx1xHSxWS, = data
            return self.output_y_pred_Bx1xHxW(y_pred_Bx1xHSxWS)

    def downsample_x_real_BxCxHSxWS(self, x_BxCxHxW: torch.Tensor):
        return self.ave_pool(x_BxCxHxW)

    def downsample_y_maxpool_real_Bx1xHSxWS(self, y_real_Bx1xHxW: torch.Tensor):
        return self.max_pool(y_real_Bx1xHxW)

    def output_y_pred_Bx1xHxW(self, y_pred_Bx1xHSxWS: torch.Tensor):
        B, _, HS, WS = y_pred_Bx1xHSxWS.shape
        H = HS * self.downsample_factor
        W = WS * self.downsample_factor

        return F.interpolate(y_pred_Bx1xHSxWS, (H, W), mode='bilinear', align_corners=True)


if __name__ == '__main__':
    pass
    target_device = try_gpu()
    in_channels = 3
    out_channels = 41
    canvas_H = 256
    canvas_W = 512
    sample_factor = 4
    kernel_size = 64 + 1
    B = 6
    injectD = 2

    x_BxCxHxW = torch.randn(B, in_channels, canvas_H, canvas_W).to(device=target_device)
    x_Bx2 = torch.randn(B, 2).to(device=target_device)
    e2e = SegerAverage(in_channels=in_channels, out_channels=out_channels, H=canvas_H, W=canvas_W, downsample_factor=sample_factor).to(target_device)

    ys_BxKxHSxWS, grid_BxHSxWSx2 = e2e(x_BxCxHxW, x_Bx2)

    # plt_multi_imgshow(imgs=[label_x_BxKxHxW[0, :, :, 0], label_x_BxKxHxW[0, :, :, 1]], titles=['w', 'h'], row_col=(2, 1))
    # plt.show(block=True)
