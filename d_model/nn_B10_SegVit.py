import torch
from transformers import SegformerForSemanticSegmentation, SegformerConfig, AutoImageProcessor
from torch import nn
from peft import get_peft_model, LoraConfig, TaskType

class CustomSegformer(SegformerForSemanticSegmentation):
    def __init__(self, config, num_input=5):
        super().__init__(config)
        
        old_conv = self.segformer.encoder.patch_embeddings[0].proj
        new_in_channels = num_input

        new_conv = nn.Conv2d(
            in_channels=new_in_channels, 
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None  
        )

        self.segformer.encoder.patch_embeddings[0].proj = new_conv
        
    def forward(self, pixel_values):
        return_dict =  self.config.use_return_dict
        output_hidden_states = (
           self.config.output_hidden_states
        )

        outputs = self.segformer(
            pixel_values,
            output_hidden_states=True,  
            return_dict=return_dict,
        )

        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]

        logits = self.decode_head(encoder_hidden_states)
        upsampled_logits = nn.functional.interpolate(
                logits, size=pixel_values.shape[-2:], mode="bilinear", align_corners=False
            )
        return upsampled_logits 


class CustomSegformerWithLoRA(CustomSegformer):
    def __init__(self):
        config = SegformerConfig.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        config.num_labels = 1  
        super().__init__(config)
        self.image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

        target = set()
        for name, module in self.segformer.named_modules():
            if name.endswith('self.query') or name.endswith('self.key') or name.endswith('self.value') or name.endswith("mlp.dense1") or name.endswith("mlp.dense2") or name.endswith("dwconv.dwconv") or name.endswith("output.dense") or name.endswith('.0.proj'):
                target.add(name)

        lora_config = LoraConfig(
            r=8,  
            lora_alpha=16,  
            lora_dropout=0.1,  
            inference_mode=False,
            target_modules=target
        )
        self.return_dict = True
        
        self.segformer = get_peft_model(self.segformer, lora_config)
        # print(self.segformer)

    def forward(self, pixel_values):
        outputs = self.segformer(pixel_values=pixel_values, output_hidden_states=True, return_dict=self.return_dict,)
        
        encoder_hidden_states = outputs.hidden_states if self.return_dict else outputs[1]
        outputs = self.decode_head(encoder_hidden_states)
        upsampled_logits = nn.functional.interpolate(
                outputs, size=pixel_values.shape[-2:], mode="bilinear", align_corners=False
            )
        return upsampled_logits 

    def train(self, mode=True):
        super(CustomSegformerWithLoRA, self).train(mode)

def test_with_dummy_input(model):
    dummy_input = torch.randn(3, 6, 160, 160)
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}") 

def test_with_gradient(model):
    dummy_input = torch.randn(3, 6, 160, 160).to('cuda')
    dummy_target = torch.randn(3, 1, 160, 160).to('cuda') 

    model.train()
    output = model(dummy_input)

    criterion = nn.MSELoss()
    loss = criterion(output, dummy_target)
    loss.backward()

    print("\nChecking gradients for trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            print(f"{name} has gradient: {param.grad.norm().item()}")
        elif param.requires_grad and param.grad is None:
            print(f"{name} has NO gradient!")

def main():
    config = SegformerConfig.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
    config.num_labels = 1  
    model = CustomSegformer(config).to('cuda')  
    with torch.cuda.device(0):  # Use GPU if available, otherwise, use CPU
            macs, params = get_model_complexity_info(model, (5, 256, 256), as_strings=True,
                                                    print_per_layer_stat=True, verbose=True)
    print(f"FLOPs: {macs}, Parameters: {params}")
    # test_with_gradient(model)



if __name__ == "__main__":
    from ptflops import get_model_complexity_info
    main()
