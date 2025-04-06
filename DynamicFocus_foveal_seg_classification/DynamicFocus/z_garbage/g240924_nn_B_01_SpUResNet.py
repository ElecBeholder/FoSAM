import torch
import torch.nn as nn

from d_model.nn_A0_utils import MM_device_gpu
from z_garbage.nn_B_0_Unet import UNet


class SpUResNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        参数:
        - in_channels: 输入通道数
        - out_channels: 输出通道数
        """
        super(SpUResNet, self).__init__()
        self.unet = UNet(in_channels, out_channels)

    def forward(self, x_Bx3xHxW, x_Bx2):
        y_BxKxHxW = torch.softmax(self.unet(x_Bx3xHxW, x_Bx2), dim=1)
        return y_BxKxHxW


if __name__ == '__main__':
    pass

    target_device = MM_device_gpu
    B = 5
    in_channels = 3
    out_channels = 41
    H = 128
    W = 256

    model = SpUResNet(in_channels, out_channels, H, W).to(target_device)
    x_Bx3xHxW = torch.randn(B, in_channels, H, W).to(target_device)
    x_Bx2 = torch.randn(B, 2).to(target_device)

    y_BxKxHxW = model(x_Bx3xHxW, x_Bx2)
