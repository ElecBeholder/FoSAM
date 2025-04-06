import torch
import torch.nn as nn

from d_model.nn_A0_utils import calc_model_memsize


class ResNet(nn.Module):

    def __init__(self, CI, CO, hidden_size=16, layer_num=3, dropout_rate=0.0):
        super(ResNet, self).__init__()

        self.CI = CI  # Number of input channels
        self.CO = CO  # Number of output channels
        self.Z = hidden_size  # Number of hidden channels
        self.L = layer_num  # Number of residual blocks
        self.dropout_rate = dropout_rate

        # Input batch normalization and 1D convolution
        if layer_num == 0:
            self.bn_in = nn.BatchNorm1d(self.CI)
            self.linear_in_out = nn.Linear(self.CI, self.CO)
        else:
            self.bn_in = nn.BatchNorm1d(self.CI)
            self.linear_in = nn.Linear(self.CI, self.Z)

            # Lists for intermediate layers (residual blocks)
            self.bn_s = nn.ModuleList()
            self.linear_s = nn.ModuleList()
            self.dp_s = nn.ModuleList()
            self.af_s = nn.ModuleList()

            for _ in range(self.L):
                self.bn_s.append(nn.BatchNorm1d(self.Z))
                self.linear_s.append(nn.Linear(self.Z, self.Z))
                self.dp_s.append(nn.Dropout(self.dropout_rate))
                self.af_s.append(nn.PReLU())

            # Output batch normalization and 1D convolution
            self.bn_out = nn.BatchNorm1d(self.Z)
            self.conv_out = nn.Linear(self.Z, self.CO)

    def forward(self, x_BxF: torch.Tensor):
        if self.L == 0:
            y_BxE = self.linear_in_out(self.bn_in(x_BxF))
        else:
            # Initial convolution and normalization
            z_BxF = self.linear_in(self.bn_in(x_BxF))

            # Residual blocks
            for dp, af, conv, bn in zip(self.dp_s, self.af_s, self.linear_s, self.bn_s):
                z_BxF = z_BxF + dp(af(conv(bn(z_BxF))))

            # Final output layer
            y_BxE = self.conv_out(self.bn_out(z_BxF))

        return y_BxE


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


class UNetModified(nn.Module):
    def __init__(self, in_channels, out_channels, injectD, H, W, base_channels=32):
        super(UNetModified, self).__init__()
        self.Cbase = base_channels
        self.base_channels = base_channels
        self.H = H
        self.W = W

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
        if injectD > 0:
            # 处理 x_Bx2

            self.res_in_channels = 2
            self.res_out_channels = 16 * self.Cbase * (self.H // 16) * (self.W // 16)
            self.fc_xBx2 = ResNet(self.res_in_channels, self.res_out_channels, hidden_size=1024, layer_num=3, dropout_rate=0.0)
            Cfactor = 2

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

    def forward(self, x_BxCxHxW: torch.Tensor, x_Bx2: torch.Tensor = None):
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
        if x_Bx2 is not None:
            # 处理 x_Bx2
            x_Bx2_flat = x_Bx2.view(B, -1)  # (B, 2)
            x_Bx16C = self.fc_xBx2(x_Bx2_flat)  # (B, 16C)
            x_Bx16CxH16xW16 = x_Bx16C.view(B, 16 * self.Cbase, x_enc5_Bx16CxH16xW16.shape[2], x_enc5_Bx16CxH16xW16.shape[3])  # (B, 16C, 1, 1)

            # 在瓶颈处拼接
            x_bottleneck_Bx16CFxH16xW16 = torch.cat([x_enc5_Bx16CxH16xW16, x_Bx16CxH16xW16], dim=1)  # (B, 32C, H/16, W/16)

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


if __name__ == '__main__':
    pass
    # 假设输入通道数为 3，类别数为 K
    in_channels = 3
    out_channels = 41  # 请将 K 替换为您的类别数

    H = 64
    W = 128

    model_inject = UNetModified(in_channels, 1, injectD=2, H=H, W=W)
    model_pure = UNetModified(in_channels, out_channels, injectD=0, H=H, W=W)
    calc_model_memsize(model_inject)
    calc_model_memsize(model_pure)

    # 创建随机输入张量
    B, H, W = 5, 64, 128  # 示例批量大小和图像尺寸
    x_BxCxHxW = torch.randn(B, in_channels, H, W)  # 图像输入 x_BxCxHxW
    x_Bx2 = torch.randn(B, 2)  # 额外的特征输入 x_Bx2

    # 前向传播
    y_inject_BxKxHxW = model_inject(x_BxCxHxW, x_Bx2)
    y_pure_BxKxHxW = model_pure(x_BxCxHxW)

    print(y_inject_BxKxHxW.shape)  # 应该是 (B, K, H, W)
    print(y_pure_BxKxHxW.shape)  # 应该是 (B, K, H, W)
