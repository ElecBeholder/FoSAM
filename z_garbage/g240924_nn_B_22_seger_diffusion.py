import torch
import torch.nn as nn
import torch.nn.functional as F

from d_model.nn_A0_utils import MM_device_gpu
from d_model.nn_B0_deformed_sampler import get_grid_Bx2xHSxWS, deformed_unsampler
from z_garbage.nn_B_0_Unet import UNet


class SegerDiffusion(nn.Module):
    def __init__(self, in_channels=3, out_channels=41, H=256, W=512, downsample_factor=4, kernel_size=64 + 1):
        super(SegerDiffusion, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = out_channels

        self.H = H
        self.W = W
        self.downsample_factor = downsample_factor
        self.HS = self.H // self.downsample_factor  # 64
        self.WS = self.W // self.downsample_factor  # 128

        self.kernel_size = kernel_size

        self.pad = self.kernel_size // 2

        self.gen_densitymap = UNet(in_channels=in_channels + 1, out_channels=1, injectD=0, base_channels=32, H=self.HS, W=self.WS)
        self.gen_segmentation = UNet(in_channels=in_channels + 1, out_channels=1, injectD=0, base_channels=32, H=self.HS, W=self.WS)


    def forward(self, x_BxCxHxW: torch.Tensor, x_Bx2: torch.Tensor):
        B, C, H, W = x_BxCxHxW.shape

        x_BxC1xHxW = torch.zeros(B, C + 1, H, W).to(dtype=torch.float32, device=x_BxCxHxW.device)
        x_BxC1xHxW[:, :-1, :, :] = x_BxCxHxW

        hidx_B = (x_Bx2[:, 0] * (H - 1)).to(torch.int64)
        widx_B = (x_Bx2[:, 1] * (W - 1)).to(torch.int64)
        x_BxC1xHxW[torch.arange(B), -1, hidx_B, widx_B] = 1.0

        # x_BxC1xHxW contains rgb + focus point

        x_BxC1xHSxWS = F.interpolate(x_BxC1xHxW, size=(self.HS, self.WS), mode='bilinear', align_corners=True)

        B, I, HS, WS = x_BxC1xHSxWS.shape

        dm_pred_Bx1xHSxWS = torch.sigmoid(self.gen_densitymap(x_BxC1xHSxWS, x_Bx2))

        dm_v_Bx1xHSPxWSP = F.pad(dm_pred_Bx1xHSxWS, (self.pad, self.pad, self.pad, self.pad), mode='replicate')

        grid_pred_BxHSxWSx2 = get_grid_Bx2xHSxWS(dm_v_Bx1xHSPxWSP, self.HS, self.WS, kernel_size=self.kernel_size)

        xs_gs_BxC1xHSxWS = F.grid_sample(x_BxC1xHxW, grid_pred_BxHSxWSx2, mode='bilinear', align_corners=True)

        # plt_multi_imgshow([x_BxC1xHSxWS[0, -1, :, :], xs_gs_BxC1xHSxWS[0, -1, :, :]], row_col=(2, 1))
        # plt.show(block=True)

        ys_pred_gs_BxKxHSxWS = torch.softmax(self.gen_segmentation(xs_gs_BxC1xHSxWS), dim=1)

        return ys_pred_gs_BxKxHSxWS, grid_pred_BxHSxWSx2, xs_gs_BxC1xHSxWS[:, :-1, :, :], dm_pred_Bx1xHSxWS

    def downsample_y_real_BxHSxWS(self, y_real_BxHxW: torch.Tensor, grid_pred_BxHSxWSx2: torch.Tensor):
        y_real_BxKxHxW = F.one_hot(y_real_BxHxW, num_classes=self.K).permute(0, 3, 1, 2).to(torch.float32)
        y_real_BxHSxWS = F.grid_sample(y_real_BxKxHxW, grid_pred_BxHSxWSx2, mode='bilinear', align_corners=True).argmax(dim=1)
        return y_real_BxHSxWS

    def output_y_pred_BxHxW(self, y_pred_BxKxHSxWS: torch.Tensor, grid_pred_BxHSxWSx2: torch.Tensor):
        y_pred_BxHxW = deformed_unsampler(y_pred_BxKxHSxWS, grid_pred_BxHSxWSx2, self.H, self.W).argmax(dim=1)
        return y_pred_BxHxW


if __name__ == '__main__':
    pass
    target_device = MM_device_gpu
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
    e2e = SegerDiffusion(in_channels=in_channels, out_channels=out_channels, H=canvas_H, W=canvas_W, downsample_factor=sample_factor).to(target_device)

    ys_BxKxHSxWS, grid_BxHSxWSx2 = e2e(x_BxCxHxW, x_Bx2)

    # plt_multi_imgshow(imgs=[label_x_BxKxHxW[0, :, :, 0], label_x_BxKxHxW[0, :, :, 1]], titles=['w', 'h'], row_col=(2, 1))
    # plt.show(block=True)
