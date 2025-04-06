import torch
import torch.nn as nn


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
