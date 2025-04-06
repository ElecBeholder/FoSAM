import torch
import torch.nn as nn
import torch.nn.functional as F

from d_model.nn_A0_utils import calc_model_memsize
from d_model.nn_A2_loss import weighted_cosine_loss
from utility.torch_tools import str_tensor_shape


class ResNet2D(nn.Module):

    def __init__(self, CI, CO, hidden_size=16, L=3, dropout_rate=1 / 2):
        super(ResNet2D, self).__init__()

        self.CI = CI  # Number of input channels
        self.CO = CO  # Number of output channels
        self.Z = hidden_size  # Number of hidden channels
        self.L = L  # Number of residual blocks
        self.dropout_rate = dropout_rate

        # Input batch normalization and 2D convolution
        if L == 0:
            self.bn_in = nn.BatchNorm2d(self.CI)
            self.conv_in_out = nn.Conv2d(self.CI, self.CO, kernel_size=1, padding=0)
        else:
            self.bn_in = nn.BatchNorm2d(self.CI)
            self.conv_in = nn.Conv2d(self.CI, self.Z, kernel_size=1, padding=0)

            # Lists for intermediate layers (residual blocks)
            self.conv_s = nn.ModuleList()
            self.bn_s = nn.ModuleList()
            self.dp_s = nn.ModuleList()
            self.af_s = nn.ModuleList()

            for _ in range(self.L):
                self.bn_s.append(nn.BatchNorm2d(self.Z))
                self.conv_s.append(nn.Conv2d(self.Z, self.Z, kernel_size=1, padding=0))
                self.dp_s.append(nn.Dropout2d(self.dropout_rate))
                self.af_s.append(nn.PReLU())

            # Output batch normalization and 2D convolution
            self.bn_out = nn.BatchNorm2d(self.Z)
            self.conv_out = nn.Conv2d(self.Z, self.CO, kernel_size=1)

    def forward(self, x_BxCIxHxW: torch.Tensor):

        if self.L == 0:
            y_BxOxHxW = self.conv_in_out(self.bn_in(x_BxCIxHxW))
        else:
            # Initial convolution and normalization
            z_BxZxHxW = self.conv_in(self.bn_in(x_BxCIxHxW))

            # Residual blocks
            for dp, af, conv, bn in zip(self.dp_s, self.af_s, self.conv_s, self.bn_s):
                z_BxZxHxW = z_BxZxHxW + dp(af(conv(bn(z_BxZxHxW))))

            # Final output layer
            y_BxOxHxW = self.conv_out(self.bn_out(z_BxZxHxW))

        return y_BxOxHxW


class ResNet(nn.Module):

    def __init__(self, CI, CO, hidden_size=16, L=3, dropout_rate=1 / 2):
        super(ResNet, self).__init__()

        self.CI = CI  # Number of input channels
        self.CO = CO  # Number of output channels
        self.Z = hidden_size  # Number of hidden channels
        self.L = L  # Number of residual blocks
        self.dropout_rate = dropout_rate

        # Input batch normalization and 1D convolution
        if L == 0:
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


class ConvDn2D(nn.Module):
    """
    A class that performs downsampling using Conv2d with kernel_size=4, stride=4, padding=0.
    """

    def __init__(self, CI, CO, degree=2):
        super(ConvDn2D, self).__init__()
        self.factor = pow(2, degree)

        self.conv = nn.Conv2d(CI, CO, kernel_size=self.factor, stride=self.factor, padding=0)

    def forward(self, x):
        return self.conv(x)


class ConvUp2D(nn.Module):
    """
    A class that performs upsampling using ConvTranspose2d with kernel_size=4, stride=4, padding=0.
    """

    def __init__(self, CI, CO, degree=2):
        self.factor = pow(2, degree)

        super(ConvUp2D, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(CI, CO, kernel_size=self.factor, stride=self.factor, padding=0)

    def forward(self, x):
        return self.conv_transpose(x)


class ResUnet(nn.Module):
    def __init__(self, CI, CO, injectD=2):
        """
        参数:
        - in_channels: 输入通道数
        - out_channels: 输出通道数
        """
        super(ResUnet, self).__init__()

        self.CI = CI
        self.CO = CO
        self.Z = 16  # 4
        self.Zfactor = 2

        self.degree = 2  # 4x4 kernel

        self.H = 256
        self.W = 512

        self.layer_num_resnet = 1
        self.dropout_rate_resnet = 1 / 8

        Z_fm, Z_to = self.CI, self.Z * (self.Zfactor ** 1)  # 16 = 8x2^1
        self.enc1_conv = ConvDn2D(Z_fm, Z_to, degree=self.degree)  # 3x256x512 => 16x64x128

        Z_fm, Z_to = Z_to, Z_to
        self.enc2_resnet = ResNet2D(Z_fm, Z_to, hidden_size=max(Z_fm, Z_to) * 2, L=self.layer_num_resnet, dropout_rate=self.dropout_rate_resnet)  # 16x64x128 => 16x64x128

        Z_fm, Z_to = Z_to, self.Z * (self.Zfactor ** 2)  # 32 = 8x2^2
        self.enc3_conv = ConvDn2D(Z_fm, Z_to, degree=self.degree)  # 16x64x128 => 32x16x32

        Z_fm, Z_to = Z_to, Z_to
        self.enc4_resnet = ResNet2D(Z_fm, Z_to, hidden_size=max(Z_fm, Z_to) * 2, L=self.layer_num_resnet, dropout_rate=self.dropout_rate_resnet)  # 32x16x32 => 32x16x32

        Z_fm, Z_to = Z_to, self.Z * (self.Zfactor ** 3)  # 64 = 8x2^3
        self.enc5_conv = ConvDn2D(Z_fm, Z_to, degree=self.degree)  # 32x16x32 => 64x4x8

        Z_fm, Z_to = Z_to, Z_to
        self.enc6_resnet = ResNet2D(Z_fm, Z_to, hidden_size=max(Z_fm, Z_to) * 2, L=self.layer_num_resnet, dropout_rate=self.dropout_rate_resnet)  # 64x4x8 => 64x4x8

        Z_fm, Z_to = Z_to, self.Z * (self.Zfactor ** 4)  # 128 = 8x2^4
        self.enc7_conv = ConvDn2D(Z_fm, Z_to, degree=self.degree)  # 64x4x8 => 128x1x2

        self.neckK, self.neckH, self.neckW = Z_to, 1, 2
        # flatten + injection # 128x1x2 => 128*1*2+2
        KHW = self.neckK * self.neckH * self.neckW
        KHWD = self.neckK * self.neckH * self.neckW + injectD
        self.neck_resnet = ResNet(KHWD, KHW, hidden_size=max(KHWD, KHW) * 2, L=self.layer_num_resnet, dropout_rate=self.dropout_rate_resnet)  # 128*1*2+2 => 128*1*2
        # view #  128*1*2 => 128x1x2

        Z_fm, Z_to = Z_to, self.Z * (self.Zfactor ** 3)  # 64 = 8x2^3
        self.dec1_conv = ConvUp2D(Z_fm, Z_to, degree=self.degree)  # 128x1x2 => 64x4x8

        Z_fm, Z_to = Z_to, Z_to
        self.dec2_resnet = ResNet2D(Z_fm + Z_fm, Z_to, hidden_size=max(Z_fm + Z_fm, Z_to) * 2, L=self.layer_num_resnet, dropout_rate=self.dropout_rate_resnet)  # 64x4x8 => 64x4x8

        Z_fm, Z_to = Z_to, self.Z * (self.Zfactor ** 2)  # 32 = 8x2^2
        self.dec3_conv = ConvUp2D(Z_fm, Z_to, degree=self.degree)  # 64x4x8 => 32x16x32

        Z_fm, Z_to = Z_to, Z_to
        self.dec4_resnet = ResNet2D(Z_fm + Z_fm, Z_to, hidden_size=max(Z_fm + Z_fm, Z_to) * 2, L=self.layer_num_resnet, dropout_rate=self.dropout_rate_resnet)  # 32x16x32 => 32x16x32

        Z_fm, Z_to = Z_to, self.Z * (self.Zfactor ** 1)  # 32 = 4x2^2
        self.dec5_conv = ConvUp2D(Z_fm, Z_to, degree=self.degree)  # 32x16x32 => 32x64x128

        Z_fm, Z_to = Z_to, Z_to
        self.dec6_resnet = ResNet2D(Z_fm + Z_fm, Z_to, hidden_size=max(Z_fm + Z_fm, Z_to) * 2, L=self.layer_num_resnet, dropout_rate=self.dropout_rate_resnet)  # 32x64x128 => 32x64x128

        Z_fm, Z_to = Z_to, self.CO
        self.dec7_conv = ConvUp2D(Z_fm, Z_to, degree=self.degree)  # 32x64x128 => Kx256x512

        Z_fm, Z_to = Z_to, self.CO
        self.dec8_resnet = ResNet2D(Z_fm, Z_to, hidden_size=max(Z_fm, Z_to) * 2, L=self.layer_num_resnet, dropout_rate=self.dropout_rate_resnet)  # Kx256x512 => Kx256x512

    def forward(self, x_BxCx256x512: torch.Tensor, x_BxDI=None):
        x_enc1_Bx16x64x128 = self.enc1_conv(x_BxCx256x512)
        x_enc2_Bx16x64x128 = self.enc2_resnet(x_enc1_Bx16x64x128)
        x_enc3_Bx32x16x32 = self.enc3_conv(x_enc2_Bx16x64x128)
        x_enc4_Bx32x16x32 = self.enc4_resnet(x_enc3_Bx32x16x32)
        x_enc5_Bx64x4x8 = self.enc5_conv(x_enc4_Bx32x16x32)
        x_enc6_Bx64x4x8 = self.enc6_resnet(x_enc5_Bx64x4x8)
        x_enc7_Bx128x1x2 = self.enc7_conv(x_enc6_Bx64x4x8)

        x_neck_Bx128x1x2 = self.neck_resnet(torch.cat([x_enc7_Bx128x1x2.flatten(start_dim=1), x_BxDI], dim=1)).view(-1, self.neckK, self.neckH, self.neckW)

        x_dec1_Bx64x4x8 = self.dec1_conv(x_neck_Bx128x1x2)
        x_dec2_Bx64x4x8 = self.dec2_resnet(torch.cat([x_dec1_Bx64x4x8, x_enc6_Bx64x4x8], dim=1))
        x_dec3_Bx32x16x32 = self.dec3_conv(x_dec2_Bx64x4x8)
        x_dec4_Bx32x16x32 = self.dec4_resnet(torch.cat([x_dec3_Bx32x16x32, x_enc4_Bx32x16x32], dim=1))
        x_dec5_Bx16x64x128 = self.dec5_conv(x_dec4_Bx32x16x32)
        x_dec6_Bx16x64x128 = self.dec6_resnet(torch.cat([x_dec5_Bx16x64x128, x_enc2_Bx16x64x128], dim=1))
        x_dec7_BxKx256x512 = self.dec7_conv(x_dec6_Bx16x64x128)

        x_dec8_BxKx256x512 = self.dec8_resnet(x_dec7_BxKx256x512)
        return x_dec8_BxKx256x512


if __name__ == '__main__':
    pass
    
    test_mode = 'UResNet'

    if test_mode == 'ResNet2D':
        L_values = [0, 1, 2, 3]
        Z_values = [4, 16]
        CI, CO = 4, 4
        H, W = 16, 32  # Height and width of the input image

        # Create an input tensor with shape (B, CI, H, W), where B = 1, CI = 4, H = 16, W = 32
        input_tensor = torch.randn(1, CI, H, W)

        # Loop through the different configurations of L and Z
        for Z in Z_values:
            for L in L_values:
                # Initialize the model with the current L and Z values
                model = ResNet2D(CI=CI, CO=CO, hidden_size=Z, L=L)

                # Print the model's current configuration
                print(f"Testing model with Z={Z} and L={L}")
                # Print the input shape
                print(f"Input shape: {str_tensor_shape(input_tensor)}")
                # Forward pass through the model
                output_tensor = model(input_tensor)
                # Print the output shape
                print(f"Output shape: {str_tensor_shape(output_tensor)}\n")

    elif test_mode == 'ConvUP2D_ConvDown2D':
        CI, CO, H, W = 4, 8, 16, 32

        degree = 2

        # Create an example input tensor (batch_size, CI, H, W)
        input_tensor = torch.randn(1, CI, H, W)

        # ConvDown operation
        conv_down = ConvDn2D(CI=CI, CO=CO, degree=degree)
        down_output = conv_down(input_tensor)
        print(f"ConvDown: Input shape: {str_tensor_shape(input_tensor)}, Output shape: {str_tensor_shape(down_output)}")

        # ConvUp operation
        conv_up = ConvUp2D(CI=CO, CO=CI, degree=degree)
        up_output = conv_up(down_output)
        print(f"ConvUp: Input shape: {str_tensor_shape(down_output)}, Output shape: {str_tensor_shape(up_output)}")
    elif test_mode == 'UResNet':

        uresnet = ResUnet(3, 1, injectD=2)
        x_BxCIxHxW = torch.randn(5, 3, 256, 512)
        x_BxID = torch.randn(5, 2)

        output_tensor = uresnet(x_BxCIxHxW, x_BxID)

        calc_model_memsize(uresnet)

#
# # Instantiate the model with parameterized input and output channels
# in_channels = 6  # 例如，输入通道数为6
# out_channels = 40  # 例如，输出通道数为40
#
# ResNet2D
#
#
# model = UNetResNet(in_channels=in_channels, out_channels=out_channels)
#
# # Example input: 6 channels, HxW resolution
# input_tensor = torch.randn(2, in_channels, 64, 128)  # 1是batch size，256x256是HxW
# output = model(input_tensor)
#
# print("Output shape:", output.shape)  # 输出应该是 (1, 40, H, W)
#
# P = torch.tensor([[1.0, 0.0, 0.0]])  # 目标分布 (one-hot 编码)
# Q = torch.tensor([[0.0, 1.0, 0.0]])
#
# print(weighted_cosine_loss(output, output))
# print(weighted_cosine_loss(P, Q))
#
# print(weighted_cosine_loss(P, P))
