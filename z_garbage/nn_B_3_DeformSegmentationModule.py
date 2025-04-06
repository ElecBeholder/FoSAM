import torch
from torch import nn

from d_model.nn_A0_utils import calc_model_memsize
from z_garbage.nn_B_2_hrnetv2_nodownsp import HRNetV2
from z_garbage.nn_B_3_c1 import C1


class DeformSegmentationModule(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DeformSegmentationModule, self).__init__()

        self.hrnet = HRNetV2(in_channels)
        self.c1 = C1(num_class=out_channels, fc_dim=720)

    def forward(self, x):
        return self.c1(self.hrnet(x))


if __name__ == '__main__':
    in_channels = 4
    out_channels = 1  # 请将 K 替换为您的类别数

    H = 64
    W = 128

    B = 5

    hrnet = HRNetV2()
    c1 = C1(num_class=2, fc_dim=720)
    calc_model_memsize(hrnet)
    calc_model_memsize(c1)

    # dsm = DeformSegmentationModule(in_channels=4, out_channels=1)

    x_BxCxHxW = torch.randn(B, in_channels, H, W)

    # z = dsm(x_BxCxHxW)

    # print(z.shape)
