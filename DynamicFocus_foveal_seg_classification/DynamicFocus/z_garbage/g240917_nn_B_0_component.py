class ResUnet(nn.Module):
    def __init__(self, CI, CO, H, W, initZ=4, degree=2, layer_num=3, injectD=2):
        """
        参数:
        - in_channels: 输入通道数
        - out_channels: 输出通道数
        """
        super(ResUnet, self).__init__()

        self.CI = CI
        self.CO = CO
        self.Z = initZ
        self.Zfactor = 2
        self.ZS = initZ * self.Zfactor
        self.degree = degree
        self.H = H
        self.W = W

        self.layer_num = layer_num
        self.layer_num_resnet = 2
        self.dropout_rate_resnet = 1 / 2

        self.convs_dn = nn.ModuleList()
        self.resnets_dn = nn.ModuleList()
        self.convs_up = nn.ModuleList()
        self.resnets_up = nn.ModuleList()

        curZ = self.Z
        for i in range(0, self.layer_num):
            if i == 0:
                self.convs_dn.append(ConvDn2D(self.CI, curZ, degree=self.degree))
            else:
                curZ_x_Zfactor = curZ * self.Zfactor
                self.convs_dn.append(ConvDn2D(curZ, curZ_x_Zfactor, degree=self.degree))
                curZ = curZ_x_Zfactor

            self.resnets_dn.append(ResNet2D(curZ, curZ, curZ, L=self.layer_num_resnet, dropout_rate=self.dropout_rate_resnet))

        for i in range(0, self.layer_num):
            if i < self.layer_num - 1:
                curZ_d_Zfactor = curZ // self.Zfactor
                self.convs_up.append(ConvUp2D(curZ, curZ_d_Zfactor, degree=self.degree))
                curZ = curZ_d_Zfactor

            else:
                curZ_d_Zfactor = self.ZS
                self.convs_up.append(ConvUp2D(self.ZS, self.ZS, degree=self.degree))
                curZ = curZ_d_Zfactor
            self.resnets_up.append(ResNet2D(curZ, curZ, curZ, L=self.layer_num_resnet, dropout_rate=self.dropout_rate_resnet))

        # Bottleneck
        self.neckZ = self.ZS * (self.Zfactor * (layer_num - 1))
        self.neckH = self.H // ((2 ** self.degree) ** layer_num)
        self.neckW = self.W // ((2 ** self.degree) ** layer_num)
        self.injectD = injectD

        self.neckZHW = self.neckZ * self.neckH * self.neckW
        # print(self.neckZHW, self.injectD)

        self.bottleneck = ResNet(self.neckZHW + self.injectD, self.neckZHW, Z=self.neckZHW, L=self.layer_num_resnet, dropout_rate=self.dropout_rate_resnet)

    def forward(self, x_BxCIxHxW: torch.Tensor, x_BxID=None):
        # Encoder

        debug = True
        layers = []

        if debug: print(str_tensor_shape(x_BxCIxHxW))
        x_BxZSxHSxWS = self.channel_caster_input(x_BxCIxHxW)
        if debug: print(str_tensor_shape(x_BxZSxHSxWS))
        for i in range(1, self.layer_num):
            x_BxZSxHSxWS = self.resnets_dn[i - 1](x_BxZSxHSxWS)
            layers.append(x_BxZSxHSxWS)

            if debug: print(i, str_tensor_shape(x_BxZSxHSxWS))
            x_BxZSxHSxWS = self.convs_dn[i - 1](x_BxZSxHSxWS)
            if debug: print(i, str_tensor_shape(x_BxZSxHSxWS))

        x_BxZSHSWS = torch.flatten(x_BxZSxHSxWS, start_dim=1)

        if debug: print(str_tensor_shape(x_BxZSHSWS))

        xi_BxZSHSWSDI = x_BxZSHSWS
        if x_BxID is not None:
            xi_BxZSHSWSDI = torch.cat([x_BxZSHSWS, x_BxID], dim=1)

        if debug: print(str_tensor_shape(xi_BxZSHSWSDI))

        xi_BxZSHSWS = self.bottleneck(xi_BxZSHSWSDI)
        if debug: print(str_tensor_shape(xi_BxZSHSWS))

        xi_BxZSxHSxWS = xi_BxZSHSWS.view(-1, self.neckZ, self.neckH, self.neckW)
        if debug: print(str_tensor_shape(xi_BxZSxHSxWS))

        for i in range(1, self.layer_num):
            xi_BxZSxHSxWS = self.convs_up[i - 1](xi_BxZSxHSxWS)
            if debug: print(i, str_tensor_shape(xi_BxZSxHSxWS))
            xi_BxZSxHSxWS = self.resnets_up[i - 1](xi_BxZSxHSxWS)
            if debug: print(i, str_tensor_shape(xi_BxZSxHSxWS))
        y_BxCOxHxW = self.channel_caster_output(xi_BxZSxHSxWS)
        if debug: print(str_tensor_shape(y_BxCOxHxW))
        return y_BxCOxHxW
