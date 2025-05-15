import torch.nn as nn
from thop import profile
import torch
class LightweightViT(nn.Module):
    """
    轻量级 Vision Transformer，使用 PyTorch 预定义组件
    """
    def __init__(self, dim=384, depth=3, heads=6, mlp_dim=None):
        super(LightweightViT, self).__init__()
        from torch.nn import TransformerEncoderLayer, TransformerEncoder
        
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim or dim * 4
        
        # 使用 PyTorch 的标准 TransformerEncoderLayer
        encoder_layer = TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=self.mlp_dim,
            batch_first=True,
            activation='gelu'
        )
        
        # 创建 TransformerEncoder
        self.transformer = TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 添加 Layer Norm
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        """
        参数:
            x: 输入张量 [B, C, H, W]
        返回:
            output: 输出特征 [B, H, W, C]
        """
        B, C, H, W = x.shape
        
        # 转置为 [B, H, W, C] 格式
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        
        # 保存原始形状
        orig_shape = x.shape
        
        # 重塑为序列格式进行处理 [B, H*W, C]
        x = x.reshape(B, H * W, C)
        
        # 通过 Transformer 层
        x = self.transformer(x)  # [B, H*W, C]
        
        # 应用最终的 Layer Norm
        x = self.norm(x)
        
        # 重塑回原始空间格式 [B, H, W, C]
        x = x.reshape(orig_shape)
        
        return x
    
model = LightweightViT(dim=384, depth=3, heads=3)
img_data_ds = torch.ones(1, 384, 10, 10)
flops, params = profile(model, inputs=(img_data_ds,))
print(f"GMACs: {flops / 1e9:.4f} GMACs, Params: {params / 1e6:.4f} M") #3G, 15M