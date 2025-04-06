from segmentation_models_pytorch import PSPNet
import torch
import torch.nn as nn

class CustomPSPNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=1, activation='softmax'):
        super(CustomPSPNet, self).__init__()

        self.base_model = PSPNet(encoder_name="resnet50", encoder_weights="imagenet", classes=num_classes, activation=None)
        # self.base_model = PSPNet(encoder_name="resnet50", encoder_weights=None, classes=num_classes, activation=None)
        if input_channels != 3: 
            original_conv1 = self.base_model.encoder.conv1
            self.base_model.encoder.conv1 = nn.Conv2d(
                input_channels,
                original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=original_conv1.bias
            )
        
    def forward(self, x):
        x = self.base_model.encoder(x)
        x = self.base_model.decoder(*x)
        masks = self.base_model.segmentation_head(x)
        return masks

if __name__ == '__main__':
    x = torch.randn(10, 5, 256, 256, requires_grad=True)

    model = CustomPSPNet(input_channels=5, num_classes=1)
    model.train()  # 切换到训练模式
    from ptflops import get_model_complexity_info
    with torch.cuda.device(0):  # Use GPU if available, otherwise, use CPU
        macs, params = get_model_complexity_info(model, (5, 256, 256), as_strings=True,
                                                print_per_layer_stat=True, verbose=True)
    print(f"FLOPs: {macs}, Parameters: {params}")

