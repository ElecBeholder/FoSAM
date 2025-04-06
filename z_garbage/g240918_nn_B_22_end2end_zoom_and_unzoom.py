import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

import preset
from d_model.nn_A0_utils import MM_device_gpu
from z_garbage.g240918_nn_B_0_component import ResUnet
from utility.plot_tools import plt_multi_imgshow


def get_gaussian_kernel(kernel_size=5, sigma=1.0):
    # 创建一维的高斯核
    x = torch.arange(kernel_size) - (kernel_size - 1) / 2
    gauss_1d = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    gauss_1d = gauss_1d / gauss_1d.sum()

    # 通过外积创建2D高斯核
    gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]

    return gauss_2d


# 获取5x5的高斯核
gaussian_kernel = get_gaussian_kernel(5, 1.0)

# 将核扩展为适用于Conv2D的格式
gaussian_kernel = gaussian_kernel.view(1, 1, 5, 5)


class End2End_ZZ(nn.Module):
    def __init__(self, in_channels=5, out_channels=40, canvas_sample_H=32, canvas_sample_W=64):
        super(End2End_ZZ, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.canvas_sample_H = canvas_sample_H
        self.canvas_sample_W = canvas_sample_W

        # Generator modules
        self.generator_densitymap_sample = ResUnet(CI=self.in_channels, CO=1, layer_num=8)
        self.generator_densitymap_unsample = ResUnet(CI=self.in_channels + 1, CO=1, layer_num=4)
        self.generator_segmentation = ResUnet(CI=self.in_channels, CO=self.out_channels, layer_num=4)

    def forward(self, x_BxCxHxW: torch.Tensor, show=False, names=None):
        B, C, H, W = x_BxCxHxW.shape

        # Step 1: Downsampling (sample)
        x_BxCxHSxWS_input = F.interpolate(x_BxCxHxW, size=(self.canvas_sample_H, self.canvas_sample_W), mode='bilinear', align_corners=True)

        # Density map sampling (generator_densitymap_sample)
        dm_sp_v_Bx1xHSxWS = torch.sigmoid(self.generator_densitymap_sample(x_BxCxHSxWS_input))
        # dm_sp_v_Bx1xHSxWS = F.conv2d(dm_sp_v_Bx1xHSxWS, gaussian_kernel.to(x_BxCxHxW.device), padding=2)
        # dm_sp_v_Bx1xHSxWS = 1- x_BxCxHSxWS_input[:, -1:, :, :] + 1e-1

        dm_sp_v_Bx1xHSxWS = dm_sp_v_Bx1xHSxWS / torch.sum(dm_sp_v_Bx1xHSxWS, dim=[-2, -1], keepdim=True)
        idx_sp_v_BxHSxWSx2 = self.compute_idx_map(dm_sp_v_Bx1xHSxWS.repeat(1, 2, 1, 1))

        # Grid saple transformation based on the generated index map
        x_BxCxHSxWS = F.grid_sample(x_BxCxHxW, idx_sp_v_BxHSxWSx2, mode='bilinear', align_corners=True)

        # Segmentation on the sampled tensor
        label_sample_BxKxHSxWS = torch.softmax(self.generator_segmentation(x_BxCxHSxWS), dim=1)

        # Step 2: Upsampling (unsample)
        dm_unsp_v_Bx1xHSxWS = torch.sigmoid(self.generator_densitymap_unsample(torch.cat([x_BxCxHSxWS, dm_sp_v_Bx1xHSxWS], dim=1)))
        dm_unsp_v_Bx1xHxW = F.interpolate(dm_unsp_v_Bx1xHSxWS, size=(H, W), mode='bilinear', align_corners=True)
        # dm_unsp_v_Bx1xHxW = F.conv2d(dm_unsp_v_Bx1xHxW, gaussian_kernel.to(x_BxCxHxW.device), padding=2)

        dm_unsp_v_Bx1xHxW = dm_unsp_v_Bx1xHxW / torch.sum(dm_unsp_v_Bx1xHxW, dim=[-2, -1], keepdim=True)
        idx_unsp_v_BxHxWx2 = self.compute_idx_map(dm_unsp_v_Bx1xHxW.repeat(1, 2, 1, 1))

        # apply grid sampling for the upsampling process
        label_unsample_BxCxHxW = F.grid_sample(label_sample_BxKxHSxWS, idx_unsp_v_BxHxWx2, mode='bilinear', align_corners=True)

        x_unsample_BxCxHxW = F.grid_sample(x_BxCxHSxWS, idx_unsp_v_BxHxWx2, mode='bilinear', align_corners=True)

        # x_BxCxHSxWS = F.interpolate(x_BxCxHxW, size=(self.canvas_sample_H, self.canvas_sample_W), mode='bilinear', align_corners=True)
        #
        # label_unsample_BxCxHSxWS = self.generator_segmentation(x_BxCxHSxWS)
        #
        # x_unsample_BxCxHxW = F.interpolate(x_BxCxHSxWS, size=(H, W), mode='bilinear', align_corners=True)
        # label_unsample_BxCxHxW = F.interpolate(label_unsample_BxCxHSxWS, size=(H, W), mode='bilinear', align_corners=True)
        if show:
            plt_multi_imgshow([x_BxCxHxW[0, :3, :, :], None,
                               x_BxCxHSxWS[0, :3, :, :], dm_sp_v_Bx1xHSxWS[0, 0, :, :],
                               x_unsample_BxCxHxW[0, :3, :, :], dm_unsp_v_Bx1xHxW[0, 0, :, :]],
                              ["x_BxCxHxW[0,:3,:,:]", None,
                               "x_BxCxHSxWS[0,:3,:,:]", "dm_sp_v_Bx1xHSxWS",
                               "x_unsample_BxCxHxW[0,:3,:,:]", "dm_unsp_v_Bx1xHxW"], row_col=(3, 2))

            plt.savefig(os.path.join(preset.dpath_training_records, f'{names[0]}.ds.png'))
            plt.close('all')

        # Apply softmax to the final output
        label_unsample_BxCxHxW = label_unsample_BxCxHxW

        return label_unsample_BxCxHxW, x_unsample_BxCxHxW

    def compute_idx_map(self, dm_v_Bx2xHSxWS):
        """
        Compute the index map used for grid sampling based on the density map.
        Args:
            dm_v_Bx2xHSxWS: density map with shape (B, 2, H, W)
        Returns:
            idx_v_BxHSxWSx2: index map for grid sampling
        """
        # Add a small epsilon to avoid division by zero
        epsilon = 1e-8

        # Normalize and compute cumsum for the first channel
        dm_v_cumsum_0 = torch.cumsum(
            dm_v_Bx2xHSxWS[:, 0, :, :] / (torch.sum(dm_v_Bx2xHSxWS[:, 0, :, :], dim=-2, keepdim=True) + epsilon),
            dim=-2
        )

        # Normalize and compute cumsum for the second channel
        dm_v_cumsum_1 = torch.cumsum(
            dm_v_Bx2xHSxWS[:, 1, :, :] / (torch.sum(dm_v_Bx2xHSxWS[:, 1, :, :], dim=-1, keepdim=True) + epsilon),
            dim=-1
        )

        # Stack and apply linear transformation to get the grid index map
        dm_v_clone = torch.stack([dm_v_cumsum_0, dm_v_cumsum_1], dim=1)
        idx_v_Bx2xHSxWS = 2.0 * dm_v_clone - 1.0

        # Permute and flip to match the expected input format for grid_sample
        idx_v_BxHSxWSx2 = idx_v_Bx2xHSxWS.permute(0, 2, 3, 1)
        idx_v_BxHSxWSx2 = torch.flip(idx_v_BxHSxWSx2, dims=[-1])

        return idx_v_BxHSxWSx2


if __name__ == '__main__':
    pass
    target_device = MM_device_gpu
    B = 2
    in_channels = 5
    out_channels = 40

    e2e = End2End_ZZ().to(target_device)
    input_tensor = torch.randn(B, in_channels, 512, 1024).to(target_device)
    output_tensor = e2e(input_tensor)
