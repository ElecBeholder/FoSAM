import torch
import torch.nn as nn

from d_model.nn_A0_utils import calc_model_memsize, MM_device_gpu


class ConvBlock(nn.Module):
    """卷积块：卷积 + 批归一化 + ReLU"""

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=32):
        super(UNet, self).__init__()
        self.Cbase = base_channels
        self.base_channels = base_channels

        # 编码器
        self.conv1 = ConvBlock(in_channels, self.Cbase)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = ConvBlock(self.Cbase, self.Cbase * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = ConvBlock(self.Cbase * 2, self.Cbase * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = ConvBlock(self.Cbase * 4, self.Cbase * 8)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.conv5 = ConvBlock(self.Cbase * 8, self.Cbase * 16)

        Cfactor = 1

        # 解码器
        self.up_conv5 = nn.ConvTranspose2d(self.Cbase * 16 * Cfactor, self.Cbase * 8, kernel_size=2, stride=2)
        self.conv_up4 = ConvBlock(self.Cbase * 16, self.Cbase * 8)

        self.up_conv4 = nn.ConvTranspose2d(self.Cbase * 8, self.Cbase * 4, kernel_size=2, stride=2)
        self.conv_up3 = ConvBlock(self.Cbase * 8, self.Cbase * 4)

        self.up_conv3 = nn.ConvTranspose2d(self.Cbase * 4, self.Cbase * 2, kernel_size=2, stride=2)
        self.conv_up2 = ConvBlock(self.Cbase * 4, self.Cbase * 2)

        self.up_conv2 = nn.ConvTranspose2d(self.Cbase * 2, self.Cbase, kernel_size=2, stride=2)
        self.conv_up1 = ConvBlock(self.Cbase * 2, self.Cbase)

        # 最终分类层
        self.final_conv = nn.Conv2d(self.Cbase, out_channels, kernel_size=1)

    def forward(self, x_BxCxHxW: torch.Tensor):
        # 获取输入的批量大小和尺寸
        B, _, H, W = x_BxCxHxW.size()
        C = self.base_channels

        # 编码器路径
        x_enc1_BxCxHxW = self.conv1(x_BxCxHxW)  # (B, C, H, W)
        x_pool1_BxCxH2xW2 = self.pool1(x_enc1_BxCxHxW)  # (B, C, H/2, W/2)

        x_enc2_Bx2CxH2xW2 = self.conv2(x_pool1_BxCxH2xW2)  # (B, 2C, H/2, W/2)
        x_pool2_Bx2CxH4xW4 = self.pool2(x_enc2_Bx2CxH2xW2)  # (B, 2C, H/4, W/4)

        x_enc3_Bx4CxH4xW4 = self.conv3(x_pool2_Bx2CxH4xW4)  # (B, 4C, H/4, W/4)
        x_pool3_Bx4CxH8xW8 = self.pool3(x_enc3_Bx4CxH4xW4)  # (B, 4C, H/8, W/8)

        x_enc4_Bx8CxH8xW8 = self.conv4(x_pool3_Bx4CxH8xW8)  # (B, 8C, H/8, W/8)
        x_pool4_Bx8CxH16xW16 = self.pool4(x_enc4_Bx8CxH8xW8)  # (B, 8C, H/16, W/16)

        x_enc5_Bx16CxH16xW16 = self.conv5(x_pool4_Bx8CxH16xW16)  # (B, 16C, H/16, W/16)

        x_bottleneck_Bx16CFxH16xW16 = x_enc5_Bx16CxH16xW16

        # 解码器路径
        x_up5_Bx8CxH8xW8 = self.up_conv5(x_bottleneck_Bx16CFxH16xW16)  # (B, 8C, H/8, W/8)
        x_cat4_Bx16CxH8xW8 = torch.cat([x_up5_Bx8CxH8xW8, x_enc4_Bx8CxH8xW8], dim=1)  # (B, 16C, H/8, W/8)
        x_dec4_Bx8CxH8xW8 = self.conv_up4(x_cat4_Bx16CxH8xW8)  # (B, 8C, H/8, W/8)

        x_up4_Bx4CxH4xW4 = self.up_conv4(x_dec4_Bx8CxH8xW8)  # (B, 4C, H/4, W/4)
        x_cat3_Bx8CxH4xW4 = torch.cat([x_up4_Bx4CxH4xW4, x_enc3_Bx4CxH4xW4], dim=1)  # (B, 8C, H/4, W/4)
        x_dec3_Bx4CxH4xW4 = self.conv_up3(x_cat3_Bx8CxH4xW4)  # (B, 4C, H/4, W/4)

        x_up3_Bx2CxH2xW2 = self.up_conv3(x_dec3_Bx4CxH4xW4)  # (B, 2C, H/2, W/2)
        x_cat2_Bx4CxH2xW2 = torch.cat([x_up3_Bx2CxH2xW2, x_enc2_Bx2CxH2xW2], dim=1)  # (B, 4C, H/2, W/2)
        x_dec2_Bx2CxH2xW2 = self.conv_up2(x_cat2_Bx4CxH2xW2)  # (B, 2C, H/2, W/2)

        x_up2_BxCxHxW = self.up_conv2(x_dec2_Bx2CxH2xW2)  # (B, C, H, W)
        x_cat1_Bx2CxHxW = torch.cat([x_up2_BxCxHxW, x_enc1_BxCxHxW], dim=1)  # (B, 2C, H, W)
        x_dec1_BxCxHxW = self.conv_up1(x_cat1_Bx2CxHxW)  # (B, C, H, W)

        out_BxKxHxW = self.final_conv(x_dec1_BxCxHxW)  # (B, num_classes, H, W)
        return out_BxKxHxW


# Example usage
if __name__ == "__main__":
    model = UNet(in_channels=1, out_channels=1, base_channels=32).to(device=MM_device_gpu)  # Here base_channels is set to 32
    x = torch.randn(1, 1, 8, 16).to(device=MM_device_gpu)  # Batch size of 1, with 1 channel, 572x572 image size
    preds = model(x)
    print(preds.shape)  # Should output the shape of the prediction tensor
