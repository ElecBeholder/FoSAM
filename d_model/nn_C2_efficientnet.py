import torch
import torchvision.models as models
from torchinfo import summary
import torch.nn as nn
from torchvision.models.shufflenetv2 import ShuffleNet_V2_X1_0_Weights

class CustomShuffleNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CustomShuffleNet, self).__init__()
        self.model = models.shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        self.model.conv1[0] = nn.Conv2d(in_channels, 24, kernel_size=3, stride=2, padding=1, bias=False)
        
        self.model.fc = nn.Linear(self.model.fc.in_features, 512)
        self.classifier = nn.Linear(512, out_channels)
    
    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x

def check_gradients(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}")
        else:
            print(f'{name} no grad')

if __name__ == '__main__':
    custom_model = CustomShuffleNet(6, 10).to('cuda')  

    input_tensor = torch.randn(1, 6, 224, 224).to('cuda')
    target = torch.randint(0, 10, (1,)).to('cuda') 

    criterion = nn.CrossEntropyLoss()
    output = custom_model(input_tensor)
    loss = criterion(output, target)

    custom_model.zero_grad()
    loss.backward()

    print("\nGradient flow in custom model:")
    check_gradients(custom_model)
