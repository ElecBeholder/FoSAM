import torch
import torch.nn as nn
import timm

class CustomHRNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CustomHRNet, self).__init__()
        
        self.backbone = timm.create_model('hrnet_w48', pretrained=True, features_only=True)
        # self.backbone = timm.create_model('hrnet_w48', pretrained=False, features_only=True)
        
        if in_channels != 3:  
            self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        out_channels = self.backbone.feature_info[-1]['num_chs']
        
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(out_channels, num_classes, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  
        )

    def forward(self, x):
        features = self.backbone(x)[-1]  
        seg_mask = self.segmentation_head(features)
        return seg_mask

if __name__ == '__main__':
    model = CustomHRNet(in_channels=5, num_classes=1)
    print(model)
    model.train()
    from ptflops import get_model_complexity_info
    with torch.cuda.device(0):  # Use GPU if available, otherwise, use CPU
        macs, params = get_model_complexity_info(model, (5, 256, 256), as_strings=True,
                                                print_per_layer_stat=True, verbose=True)
    print(f"FLOPs: {macs}, Parameters: {params}")

