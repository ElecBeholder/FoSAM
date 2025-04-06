from torch import nn


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    "3x3 convolution + BN + relu"
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )

class C1(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1, self).__init__()
        self.use_softmax = use_softmax
        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None, res=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        x = self.conv_last(x)
        if res is not None:
            x += res

        # if self.use_softmax: # is True during inference
        #     x = nn.functional.interpolate(
        #         x, size=segSize, mode='bilinear', align_corners=False)
        #     x = nn.functional.softmax(x, dim=1)
        # else:
        #     x = nn.functional.log_softmax(x, dim=1)

        return x

if __name__ == '__main__':
    pass

    c1 = C1(num_class=2, fc_dim=720, use_softmax=False)
    print(c1)
