import torch
import torch.nn as nn

from d_model.nn_A0_utils import calc_model_memsize

import torch
import torch.nn as nn


class ConvDown2D(nn.Module):
    """二维卷积下采样块：卷积 + 批归一化 + ReLU"""

    def __init__(self, in_channels, out_channels, kernel=3, pad=1, step=1):
        super(ConvDown2D, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=step, padding=pad),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ConvUp2D(nn.Module):
    """二维卷积上采样块：反卷积 + 批归一化 + ReLU"""

    def __init__(self, in_channels, out_channels, kernel=2, pad=0, step=2):
        super(ConvUp2D, self).__init__()
        self.conv_up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel, stride=step, padding=pad),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_up(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 模块"""

    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = self.global_avg_pool(x).view(batch_size, channels)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch_size, channels, 1, 1)
        return x * y.expand_as(x)


class UNet3(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=32):
        super(UNet3, self).__init__()
        self.cbase = base_channels

        # 编码器
        self.conv2d_same_1 = ConvDown2D(in_channels, self.cbase)
        self.pool2d_down_1 = nn.MaxPool2d(2)  # H/2 x W/2

        self.conv2d_same_2 = ConvDown2D(self.cbase, self.cbase * 2)
        self.pool2d_down_2 = nn.MaxPool2d(2)  # H/4 x W/4

        self.conv2d_same_3 = ConvDown2D(self.cbase * 2, self.cbase * 4)
        self.pool2d_down_3 = nn.MaxPool2d(2)  # H/8 x W/8

        self.conv2d_down_4 = ConvDown2D(self.cbase * 4, self.cbase * 8)
        self.pool2d_down_4 = nn.MaxPool2d(2)  # H/16 x W/16

        # Bottleneck with SEBlock
        self.conv2d_bottleneck = ConvDown2D(self.cbase * 8, self.cbase * 16)
        self.se_block = SEBlock(self.cbase * 16)

        # 解码器
        self.conv2d_up_4 = ConvUp2D(self.cbase * 16, self.cbase * 8)
        self.conv2d_conv_4 = ConvDown2D(self.cbase * 16, self.cbase * 8)

        self.conv2d_up_3 = ConvUp2D(self.cbase * 8, self.cbase * 4)
        self.conv2d_conv_3 = ConvDown2D(self.cbase * 8, self.cbase * 4)

        self.conv2d_up_2 = ConvUp2D(self.cbase * 4, self.cbase * 2)
        self.conv2d_conv_2 = ConvDown2D(self.cbase * 4, self.cbase * 2)

        self.conv2d_up_1 = ConvUp2D(self.cbase * 2, self.cbase)
        self.conv2d_conv_1 = ConvDown2D(self.cbase * 2, self.cbase)

        self.conv2d_out = nn.Conv2d(self.cbase, out_channels, kernel_size=1)

    def forward(self, x):
        # 编码器
        conv1 = self.conv2d_same_1(x)
        pool1 = self.pool2d_down_1(conv1)

        conv2 = self.conv2d_same_2(pool1)
        pool2 = self.pool2d_down_2(conv2)

        conv3 = self.conv2d_same_3(pool2)
        pool3 = self.pool2d_down_3(conv3)

        conv4 = self.conv2d_down_4(pool3)
        pool4 = self.pool2d_down_4(conv4)

        # Bottleneck with SEBlock
        bottleneck = self.conv2d_bottleneck(pool4)
        bottleneck = self.se_block(bottleneck)

        # 解码器
        up4 = self.conv2d_up_4(bottleneck)
        merge4 = torch.cat([up4, conv4], dim=1)
        conv_up4 = self.conv2d_conv_4(merge4)

        up3 = self.conv2d_up_3(conv_up4)
        merge3 = torch.cat([up3, conv3], dim=1)
        conv_up3 = self.conv2d_conv_3(merge3)

        up2 = self.conv2d_up_2(conv_up3)
        merge2 = torch.cat([up2, conv2], dim=1)
        conv_up2 = self.conv2d_conv_2(merge2)

        up1 = self.conv2d_up_1(conv_up2)
        merge1 = torch.cat([up1, conv1], dim=1)
        conv_up1 = self.conv2d_conv_1(merge1)

        out = self.conv2d_out(conv_up1)
        return out


# Example usage
if __name__ == "__main__":
    # x_10x64x1x2 = torch.randn(10, 64, 1, 2).to(device=MM_device_gpu)
    #
    # conv_down_1D = ConvDown1D(in_channels=64, out_channels=128, kernel=2, pad=0, step=1).to(device=MM_device_gpu)  # Here base_channels is set to 32
    # conv_up_1D = ConvUp1D(in_channels=128, out_channels=64, kernel=2, pad=0, step=1).to(device=MM_device_gpu)  # Here base_channels is set to 32
    #
    # x_1 = conv_down_1D(x_10x64x1x2.flatten(start_dim=-2))
    # print(x_1.shape)
    # x_2 = conv_up_1D(x_1).unsqueeze(-2)
    # print(x_2.shape)
    x_10x4x64x128 = torch.randn(10, 4, 64, 128).to(device=MM_device_gpu)
    unet = UNet3(in_channels=4, out_channels=1, base_channels=8).to(device=MM_device_gpu)
    out = unet(x_10x4x64x128)
    print(out.shape)
