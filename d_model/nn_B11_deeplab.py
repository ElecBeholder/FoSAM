import torch
import torch.nn as nn
from torchvision import models
from peft import get_peft_model, LoraConfig
import collections

class CustomDeepLab(nn.Module):
    def __init__(self, num_input_channels=5, num_classes=1):
        super(CustomDeepLab, self).__init__()
        
        # self.deeplab = models.segmentation.deeplabv3_resnet50(pretrained=True)

        from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

        self.deeplab = models.segmentation.deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
       
        self.deeplab.backbone.conv1 = nn.Conv2d(
            in_channels=num_input_channels, 
            out_channels=64,           
            kernel_size=7, 
            stride=2, 
            padding=3,
            bias=True
        )
        self.deeplab.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        nn.init.normal_(self.deeplab.classifier[4].weight.data)
        nn.init.normal_(self.deeplab.backbone.conv1.weight.data)

        for name, param in self.deeplab.named_parameters():
            param.requires_grad = True
            if 'weight' in name:
                param.requires_grad = True
            elif 'bias' in name:
                param.requires_grad = True
        self.deeplab.classifier[4].weight.requires_grad = True
        self.deeplab.backbone.conv1.weight.requires_grad = True

    def forward(self, x):
        output = self.deeplab(x)['out']
        return output

class CustomDeeplabWithLoRA(nn.Module):
    pass

def register_hooks(model):
    hooks = []

    def hook_fn(module, input, output):
        class_name = module.__class__.__name__
        module_idx = len(hooks)
        if isinstance(output, torch.Tensor):
            print(f"Forward Hook {module_idx}: {class_name} | Output Shape: {output.shape}")
        elif isinstance(output, collections.OrderedDict):
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    print(f"Forward Hook {module_idx}: {class_name} | Output Shape for {key}: {value.shape}")
        else:
            print(f"Forward Hook {module_idx}: {class_name} | Output type: {type(output)}")
        hooks.append(module.register_forward_hook(hook_fn))

    model.apply(lambda module: module.register_forward_hook(hook_fn))


def test_with_dummy_input():
    model = CustomDeepLab().to('cuda')
    for name,param in model.named_parameters():
        if param.requires_grad:
            print(f'{name} need grad')
        else:
            print(name)
    dummy_input = torch.randn(10, 6, 160, 160).to('cuda')  
    dummy_target = torch.randn(10, 1, 160, 160).to('cuda')  
    output = model(dummy_input)

if __name__ == "__main__":
    test_with_dummy_input()