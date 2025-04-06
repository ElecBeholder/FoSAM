import torch
import torch.nn as nn

from d_model.nn_A0_utils import calc_model_memsize, MM_device_gpu


class ConvDown2D(nn.Module):
    """卷积块：卷积 + 批归一化 + ReLU"""

    def __init__(self, in_channels, out_channels, kernel=1, pad=0, step=1):
        super(ConvDown2D, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=pad, stride=step),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ConvUp2D(nn.Module):
    """卷积上采样块：反卷积 + 批归一化 + ReLU"""

    def __init__(self, in_channels, out_channels, kernel=1, pad=0, step=1):
        super(ConvUp2D, self).__init__()
        self.conv_up = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel, padding=pad, stride=step),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_up(x)


class ConvDown1D(nn.Module):
    """1D卷积下采样块：卷积 + 批归一化 + ReLU"""

    def __init__(self, in_channels, out_channels, kernel=1, pad=0, step=1):
        super(ConvDown1D, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel, padding=pad, stride=step),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ConvUp1D(nn.Module):
    """1D卷积上采样块：反卷积 + 批归一化 + ReLU"""

    def __init__(self, in_channels, out_channels, kernel=1, pad=0, step=1):
        super(ConvUp1D, self).__init__()
        self.conv_up = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel, padding=pad, stride=step),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_up(x)


class UNet2(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=32):
        super(UNet2, self).__init__()
        self.cbase = base_channels // 2

        # 编码器
        # self.conv2d_in = ConvDown2D(in_channels, self.cbase * 1, kernel=1, pad=0, step=1)

        self.conv2d_same_1 = ConvDown2D(in_channels, self.cbase * 2, kernel=3, pad=1, step=1)
        self.pool2d_down_1 = nn.MaxPool2d(2)  # 64x128 ->  32x64

        self.conv2d_same_2 = ConvDown2D(self.cbase * 2, self.cbase * 4, kernel=3, pad=1, step=1)
        self.pool2d_down_2 = nn.MaxPool2d(2)  # 32x64 -> 16x32

        self.conv2d_same_3 = ConvDown2D(self.cbase * 4, self.cbase * 8, kernel=3, pad=1, step=1)
        self.pool2d_down_3 = nn.MaxPool2d(2)  # 16x32 -> 8x16

        self.conv2d_down_4 = ConvDown2D(self.cbase * 8, self.cbase * 16, kernel=2, pad=0, step=2)  # 8x16 -> 4x8
        self.conv2d_down_5 = ConvDown2D(self.cbase * 16, self.cbase * 32, kernel=2, pad=0, step=2)  # 4x8 -> 2x4
        self.conv2d_down_6 = ConvDown2D(self.cbase * 32, self.cbase * 64, kernel=2, pad=0, step=2)  # 2X4 -> 1X2

        # Bottleneck
        self.conv1d_down_7 = ConvDown1D(self.cbase * 64, self.cbase * 128, kernel=2, pad=0, step=1)  # 1X2 -> 1X1
        self.conv1d_same_bride = ConvDown1D(self.cbase * 128, self.cbase * 128, kernel=1, pad=0, step=1)  # 1X1 -> 1X1
        self.conv1d_up_7 = ConvUp1D(self.cbase * 128, self.cbase * 64, kernel=2, pad=0, step=2)  # 1X1 -> 1X2

        self.conv2d_up_6 = ConvUp2D(2 * self.cbase * 64, self.cbase * 32, kernel=2, pad=0, step=2)  # 2X4 <- 1X2
        self.conv2d_up_5 = ConvUp2D(2 * self.cbase * 32, self.cbase * 16, kernel=2, pad=0, step=2)  # 4x8 <- 2x4
        self.conv2d_up_4 = ConvUp2D(2 * self.cbase * 16, self.cbase * 8, kernel=2, pad=0, step=2)  # 8x16 <- 4x8

        self.conv2d_up_3 = ConvUp2D(2 * self.cbase * 8, self.cbase * 4, kernel=2, pad=0, step=2)  # 16x32 <- 8x16
        self.conv2d_up_2 = ConvUp2D(2 * self.cbase * 4, self.cbase * 2, kernel=2, pad=0, step=2)  # 32x64 <- 16x32
        self.conv2d_up_1 = ConvUp2D(2 * self.cbase * 2, self.cbase * 1, kernel=2, pad=0, step=2)  # 64x128 <- 32x64

        self.conv2d_out = ConvUp2D(self.cbase, out_channels, kernel=1, pad=0, step=1)

    def forward(self, x_Bx4x64x128: torch.Tensor):
        # Encoder
        conv1_BxCx64x128 = self.conv2d_same_1(x_Bx4x64x128)  # Cx64x128
        pool1_BxCx32x64 = self.pool2d_down_1(conv1_BxCx64x128)  # 32x64

        conv2_BxC2x32x64 = self.conv2d_same_2(pool1_BxCx32x64)  # 32x64
        pool2_BxC2x16x32 = self.pool2d_down_2(conv2_BxC2x32x64)  # 16x32

        conv3_BxC4x8x16 = self.conv2d_same_3(pool2_BxC2x16x32)  # 16x32
        pool3_BxC4x4x8 = self.pool2d_down_3(conv3_BxC4x8x16)  # 8x16

        conv4_BxC8x4x8 = self.conv2d_down_4(pool3_BxC4x4x8)  # 8x16 -> 4x8
        conv5_BxC16x2x4 = self.conv2d_down_5(conv4_BxC8x4x8)  # 4x8 -> 2x4
        conv6_BxC32x1x2 = self.conv2d_down_6(conv5_BxC16x2x4)  # 2x4 -> 1x2

        # Bottleneck
        bottleneck_BxC64_2 = self.conv1d_down_7(conv6_BxC32x1x2.flatten(start_dim=-2))  # 1x2 -> 1x1
        bottleneck_BxC64_2 = self.conv1d_same_bride(bottleneck_BxC64_2)  # 1x1 -> 1x1
        bottleneck_BxC32x1x2 = self.conv1d_up_7(bottleneck_BxC64_2).unsqueeze(-2)  # 1x1 -> 1x2

        # Decoder
        up6_BxC16x2x4 = self.conv2d_up_6(torch.cat([bottleneck_BxC32x1x2, conv6_BxC32x1x2], dim=1))  # 1x2 -> 2x4
        up5_BxC8x4x8 = self.conv2d_up_5(torch.cat([up6_BxC16x2x4, conv5_BxC16x2x4], dim=1))  # 2x4 -> 4x8
        up4_BxC4x8x16 = self.conv2d_up_4(torch.cat([up5_BxC8x4x8, conv4_BxC8x4x8], dim=1))  # 4x8 -> 8x16

        up3_BxC2x16x32 = self.conv2d_up_3(torch.cat([up4_BxC4x8x16, pool3_BxC4x4x8], dim=1))  # 8x16 -> 16x32
        up2_BxCx32x64 = self.conv2d_up_2(torch.cat([up3_BxC2x16x32, pool2_BxC2x16x32], dim=1))  # 16x32 -> 32x64
        up1_BxCx64x128 = self.conv2d_up_1(torch.cat([up2_BxCx32x64, pool1_BxCx32x64], dim=1))  # 32x64 -> 64x128

        out_BxOxHxW = self.conv2d_out(up1_BxCx64x128)  # 最终输出
        return out_BxOxHxW


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
    unet = UNet2(in_channels=4, out_channels=1, base_channels=8).to(device=MM_device_gpu)
    out = unet(x_10x4x64x128)
    print(out.shape)
