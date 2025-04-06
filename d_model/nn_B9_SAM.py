import torch
import torch.nn as nn
from utility.build_sam import sam_model_registry

class CustomSAM(nn.Module):
    def __init__(self, input_channel=4, output_channel=1, sam_model_type="vit_h", checkpoint_path=None, img_resolution=None):
        super(CustomSAM, self).__init__()

        checkpoint_path = '/home/lwx/test/DynamicFocus/p_pretrained_model/sam_vit_h_4b8939.pth' if not checkpoint_path else checkpoint_path
        self.sam = sam_model_registry[sam_model_type](checkpoint=checkpoint_path, custom_img_size=img_resolution)
        
        for param in self.sam.parameters():
            param.requires_grad = False
        
        self.custom_input_conv = nn.Conv2d(input_channel, 3, kernel_size=3, padding=1)
        
        self.custom_output_conv = nn.Conv2d(1, output_channel, kernel_size=1)
        
        
    def forward(self, x, points=None, boxes=None):
        x = self.custom_input_conv(x)

        input = [
            {'image': x[i],
            "original_size": x[0].shape[1:],
            "point_coords": points[i],
            "point_labels": torch.ones(points[i].shape[0])} for i in range(len(x))
        ]
        low_res_mask = self.sam.forward(input, multimask_output=False)

        low_res_mask = torch.stack([x["masks"] for x in low_res_mask], dim=0)
        low_res_mask = low_res_mask.squeeze(2)
        
        output_mask = self.custom_output_conv(low_res_mask)
        
        return output_mask

if __name__ == '__main__':
    input_channel = 4 
    output_channel = 1  
    checkpoint_path = '/home/lwx/test/DynamicFocus/p_pretrained_model/sam_vit_h_4b8939.pth'  # Path to your SAM checkpoint
    
    model = CustomSAM(input_channel=input_channel, output_channel=output_channel, checkpoint_path=checkpoint_path, img_resolution=160)
    
    input_tensor = torch.randn(1, input_channel, 160, 160) 
    
    points = torch.tensor([[[500, 500]]], dtype=torch.float32)
    
    output_mask = model(input_tensor, points=points)
    
    print("Output Mask Shape:", output_mask.shape)
