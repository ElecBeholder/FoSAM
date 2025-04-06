import torch
import torch.nn as nn
import torch.optim as optim
from modeling.efficient_sam.build_efficient_sam import build_efficient_sam_vitt
from modeling.efficient_sam.efficient_sam import EfficientSam
from PIL import Image
from torchvision import transforms
import numpy as np



class CustomSAM(nn.Module):
    def __init__(self):
        super(CustomSAM, self).__init__()
        self.model = build_efficient_sam_vitt()
        self.model.pixel_mean = torch.tensor([0.485, 0.456, 0.406, 0.5]).view(1,4,1,1)
        self.model.pixel_std = torch.tensor([0.229, 0.224, 0.225, 0.25]).view(1,4,1,1)

        old_conv = self.model.image_encoder.patch_embed.proj

        new_conv = nn.Conv2d(
            in_channels=4, 
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None  
        )

        self.model.image_encoder.patch_embed.proj = new_conv
        self.classifier = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)

        for name, param in self.model.named_parameters():
            if 'decoder' in name:
                param.requires_grad = False
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)

    def forward(self, images, batched_points, batched_point_labels):
        out, _ = self.model(images, batched_points, batched_point_labels, True)
        out = out.squeeze(1)
        out = self.classifier(out)
        return out

if __name__ == '__main__':
    model = CustomSAM()
    print(model)