import torch
import pdb
import random
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import torchvision.utils as vutils
from builtins import any as b_any

from scipy.io import loadmat
import numpy as np
from PIL import Image
from PIL import ImageFilter
import time
import os
import shutil
from scipy import ndimage
import scipy.interpolate
import cv2
import torchvision.models as models
from pytorch_toolbelt.losses.dice import DiceLoss

from torch.autograd import Variable
import matplotlib.pyplot as plt
import math

class GaussianPredictor(nn.Module):
    def __init__(self, input_size=20, input_channels=386):
        super(GaussianPredictor, self).__init__()
        self.input_size = input_size
        dim = 128
        num_heads = 4
        num_layers = 4
        mlp_dim = 256
        dropout = 0.1
        self.embedding = nn.Conv2d(input_channels, dim, kernel_size=1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        encoder_layers = []
        for _ in range(num_layers):
            encoder_layers.append(
                nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=num_heads,
                    dim_feedforward=mlp_dim,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True
                )
            )
        self.transformer_layers = nn.ModuleList(encoder_layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 7)
        self.dropout = nn.Dropout(dropout)
        self.attn_pool = nn.Linear(dim, 1)
        self.gaze_proj = nn.Linear(2, dim)

    def forward(self, x, gaze_coords=None):
        B, C, H, W = x.shape
        
        x = self.embedding(x)  # [B, dim, H, W]
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 1+H*W, dim]
        if gaze_coords is not None:
            gaze_emb = self.gaze_proj(gaze_coords)  # [B, dim]
            gaze_emb = gaze_emb.unsqueeze(1)  # [B, 1, dim]
            x = torch.cat([gaze_emb, x], dim=1)  # [B, 2+H*W, dim]
        for layer in self.transformer_layers:
            x = layer(x)
        cls_feature = x[:, 0]
        if gaze_coords is not None:
            tokens = x[:, 2:]
        else:
            tokens = x[:, 1:]
        attn_weights = F.softmax(self.attn_pool(tokens).squeeze(-1), dim=1)
        attn_feature = (tokens * attn_weights.unsqueeze(-1)).sum(dim=1)
        
        x = cls_feature + attn_feature
        
        x = self.norm(x)
        x = self.head(self.dropout(x))  # [B, 7]
        
        return x

def dynamic_topk(feature_map_BxCxHxW, saliency_map_BxHxW, cut_ratio=0.005, min_tokens=30):

    B, C, H, W = feature_map_BxCxHxW.shape
    saliency_flat_BxHW = saliency_map_BxHxW.view(B, -1)
    
    max_s, _ = torch.max(saliency_flat_BxHW, dim=1, keepdim=True)  # [B, 1]
    
    threshold = max_s * cut_ratio  # [B, 1]
    
    mask_BxHW = (saliency_flat_BxHW > threshold).float()  # [B, H*W]
    
    sample_token_counts = torch.sum(mask_BxHW, dim=1).int()  # [B]
    
    for b in range(B):
        if sample_token_counts[b] < min_tokens:
            saliency_b = saliency_flat_BxHW[b]
            _, indices_sorted = torch.sort(saliency_b, descending=True)
            top_indices = indices_sorted[:min_tokens]
            new_mask = torch.zeros_like(mask_BxHW[b])
            new_mask[top_indices] = 1.0
            mask_BxHW[b] = new_mask
            sample_token_counts[b] = min_tokens
    
    max_tokens = torch.max(sample_token_counts).item()
    
    selected_feature_map = []
    all_indices = []
    
    feature_flat_BxCxHW = feature_map_BxCxHxW.view(B, C, -1)
    
    for b in range(B):
        indices = torch.nonzero(mask_BxHW[b] > 0).squeeze(1)  # [K_b]
        count = indices.size(0)
        
        indices_exp = indices.unsqueeze(0).expand(C, -1)  # [C, count]
        selected_features = torch.gather(feature_flat_BxCxHW[b], 1, indices_exp)  # [C, count]
        
        if count < max_tokens:
            feature_padding = torch.zeros(C, max_tokens - count, device=selected_features.device)
            selected_features = torch.cat([selected_features, feature_padding], dim=1)  # [C, max_tokens]
            
            index_padding = torch.full((max_tokens - count,), -1, dtype=indices.dtype, device=indices.device)
            indices = torch.cat([indices, index_padding], dim=0)  # [max_tokens]
            
        selected_feature_map.append(selected_features)
        all_indices.append(indices)
    
    selected_feature_map = torch.stack(selected_feature_map, dim=0)  # [B, C, max_tokens]
    all_indices = torch.stack(all_indices, dim=0)  # [B, max_tokens]
    
    return selected_feature_map, all_indices, sample_token_counts

class CompressNet(nn.Module):
    def __init__(self):
        super(CompressNet, self).__init__()
        self.conv_last = nn.Conv2d(24,1,kernel_size=1,padding=0,stride=1)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.act(x)
        out = self.conv_last(x)
        return out

class DeformSegmentationModule(nn.Module):
    def __init__(self, cfg):
        super(DeformSegmentationModule, self).__init__()

        self.epsilon = 1e-6
        self.debug_count = 0
        self.criterion = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.temperature = 0.5

        self.backbone = LightweightViT(dim=384, depth=2, heads=3)
        self.grid_size_x = 20
        self.grid_size_y = 20
        self.padding_size_y = 10
        self.padding_size_x = 10
        self.classifier = nn.Linear(384, 51)
        self.gaussian_predictor = nn.Linear(384, 5)
        self.dropout = nn.Dropout(0.1)
        
        self.xs = None
        self.prediction = None
        self.embeddings = None
        self.focus_embedding = None
        self.mu_xs = None
        self.mu_ys = None
        self.sigma_xs = None
        self.sigma_ys = None
        self.rhos = None
        
        self.token_count_list = []
    
    def forward(self,
                img_data,
                img_data_ds,
                focus_point,
                Y_bx1xHxW,
                Y_cls_bx1,
                cut_ratio=0.5,
                min_tokens=50):
        batch_size = img_data.shape[0]
        
        embeddings = self.backbone(img_data_ds)  # [B, 40, 40, 384]

        self.embeddings = embeddings
        ts = embeddings.shape[1]
        H, W = ts, ts
        focus_x = torch.clamp((focus_point[:, 0, 0, 0] / 640 * W).long(), 0, W-1)
        focus_y = torch.clamp((focus_point[:, 0, 0, 1] / 640 * H).long(), 0, H-1)
        focus_embedding = embeddings[torch.arange(batch_size), focus_y, focus_x, :]  # [B, 384]
        dim = embeddings.shape[-1]
        all_params = self.gaussian_predictor(self.dropout(embeddings.view(-1, dim))) # [B*H*W, 5]
        
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(0, 1, H, device=all_params.device),
            torch.linspace(0, 1, W, device=all_params.device),
            indexing='ij'
        )
        x_grid = x_grid.unsqueeze(0).unsqueeze(-1).expand(batch_size, H, W, 1)
        y_grid = y_grid.unsqueeze(0).unsqueeze(-1).expand(batch_size, H, W, 1)
        
        x_offset = torch.tanh(all_params[:, 0]).view(batch_size, H, W, 1) * 0.0
        y_offset = torch.tanh(all_params[:, 1]).view(batch_size, H, W, 1) * 0.0
        
        mu_xs = torch.clamp(x_grid + x_offset, 0, 1)  # [B, H, W, 1]
        mu_ys = torch.clamp(y_grid + y_offset, 0, 1)  # [B, H, W, 1]
        sigma_xs = (F.softplus(all_params[:, 2] + 1e-6)).view(batch_size, H, W, 1)  # [B, H, W, 1]
        sigma_ys = (F.softplus(all_params[:, 3] + 1e-6)).view(batch_size, H, W, 1)  # [B, H, W, 1]
        rhos = torch.tanh(all_params[:, 4]).view(batch_size, H, W, 1)  # [B, H, W, 1]
        
        mu_x = mu_xs[torch.arange(batch_size), focus_y, focus_x, 0].view(batch_size, 1, 1) # [B, 1, 1]
        mu_y = mu_ys[torch.arange(batch_size), focus_y, focus_x, 0].view(batch_size, 1, 1) # [B, 1, 1]
        sigma_x = sigma_xs[torch.arange(batch_size), focus_y, focus_x, 0].view(batch_size, 1, 1) # [B, 1, 1]
        sigma_y = sigma_ys[torch.arange(batch_size), focus_y, focus_x, 0].view(batch_size, 1, 1) # [B, 1, 1]
        rho = rhos[torch.arange(batch_size), focus_y, focus_x, 0].view(batch_size, 1, 1) # [B, 1, 1]
        
        x = torch.linspace(0, 1, W, device=all_params.device)
        y = torch.linspace(0, 1, H, device=all_params.device)

        xx, yy = torch.meshgrid(x, y, indexing='xy')
        xx = xx.unsqueeze(0).expand(batch_size, H, W)
        yy = yy.unsqueeze(0).expand(batch_size, H, W)

        norm_const = 1 / (2 * torch.pi * sigma_x * sigma_y * torch.sqrt(1 - rho**2))
        z_x1 = (xx - mu_x) / sigma_x
        z_y1 = (yy - mu_y) / sigma_y
        exponent = (z_x1**2 - 2 * rho * z_x1 * z_y1 + z_y1**2) / (2 * (1 - rho**2))
        saliency_map_BxHxW = norm_const * torch.exp(-exponent)

        saliency_map_BxHxW = F.interpolate(
            saliency_map_BxHxW.unsqueeze(1),  # [B, 1, H, W]
            size=(40, 40),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)  # [B, 40, 40]
            
        self.xs = saliency_map_BxHxW
        self.focus_embedding = focus_embedding
        self.focus_y = focus_y
        self.focus_x = focus_x
        self.mu_xs = mu_xs
        self.mu_ys = mu_ys
        self.sigma_xs = sigma_xs
        self.sigma_ys = sigma_ys
        self.rhos = rhos
        
        selected_feature_map, indices, sample_token_counts = dynamic_topk(
            img_data, 
            saliency_map_BxHxW.detach(), 
            cut_ratio=cut_ratio,
            min_tokens=min_tokens
        )
        
        self.token_count_list.append(sample_token_counts.detach().cpu().numpy())
        
        cur_Y_bx1xHxW = Y_bx1xHxW[:, :, :, :].to(selected_feature_map.device)
        B, _, H, W = cur_Y_bx1xHxW.shape
        ds_factor_h = H // embeddings.shape[1]
        ds_factor_w = W // embeddings.shape[2]
        downsampled_mask = F.avg_pool2d(cur_Y_bx1xHxW, kernel_size=(ds_factor_h, ds_factor_w), 
                                       stride=(ds_factor_h, ds_factor_w))  # [B, 1, H_emb, W_emb]
        downsampled_mask = downsampled_mask.permute(0, 2, 3, 1)  # [B, H_emb, W_emb, 1]
        _, _, nll_loss = self.make_prediction(downsampled_mask)
        all_fg_predictions = self.all_fg_embeddings_predictions
        sample_losses = []
        class_label = Y_cls_bx1.squeeze(1).to(downsampled_mask.device)
        for b in range(B):
            sample_preds = all_fg_predictions[b]  # Tensor of shape [N, 51]
            
            expanded_label = class_label[b].expand(sample_preds.size(0))

            sample_loss = self.criterion(sample_preds / self.temperature, expanded_label)
            sample_losses.append(sample_loss)
        
        loss = torch.stack(sample_losses).mean()
        loss_nll = nll_loss.mean()

        return selected_feature_map, indices, loss, loss_nll

    def make_prediction(self, mask_downsampled=None):
        batch_size, H, W, _ = mask_downsampled.shape
        predictions = []
        all_fg_embeddings_preds = []
        all_embeddings_preds = []
        num_embeddings_per_sample = []
        similarity_list = []
        nll_loss_list = []
        
        for b in range(batch_size):
            current_mask_downsampled = mask_downsampled[b, :, :, 0]  # [H, W]
            selected_indices = torch.nonzero(current_mask_downsampled > 0.2)  # [N, 2]
            un_selected_indices = torch.nonzero(current_mask_downsampled <= 0.2)  # [N, 2]
            
            pred_all = self.classifier(self.embeddings[b].view(-1 ,self.embeddings.shape[-1]))
            all_embeddings_preds.append(pred_all)
            if len(selected_indices) == 0:
                selected_embedding = self.focus_embedding[b]  # [384]
                pred = self.classifier(selected_embedding)  # [51]
                predictions.append(pred)
                all_fg_embeddings_preds.append(pred.unsqueeze(0))
                num_embeddings_per_sample.append(1)
                fg_embeddings = selected_embedding
                mask_unselect = torch.ones_like(current_mask_downsampled).type(torch.bool)
                mask_unselect[self.focus_y[b], self.focus_x[b]] = 0
                bg_embeddings = self.embeddings[b].view(-1 ,self.embeddings.shape[-1])[mask_unselect.view(-1)]
                
                mu_x = self.mu_xs[b, self.focus_y[b], self.focus_x[b], 0]
                mu_y = self.mu_ys[b, self.focus_y[b], self.focus_x[b], 0]
                sigma_x = self.sigma_xs[b, self.focus_y[b], self.focus_x[b], 0]
                sigma_y = self.sigma_ys[b, self.focus_y[b], self.focus_x[b], 0]
                rho = self.rhos[b, self.focus_y[b], self.focus_x[b], 0]
            else:
                selected_embeddings = torch.stack([
                    self.embeddings[b, idx[0], idx[1], :] 
                    for idx in selected_indices
                ])  # [N, 384]
                fg_embeddings = selected_embeddings.mean(dim=0)
                bg_embeddings = self.embeddings[b, un_selected_indices[:, 0], un_selected_indices[:, 1], :]
                
                individual_preds = self.classifier(selected_embeddings)  # [N, 51]
                
                all_fg_embeddings_preds.append(individual_preds)
                num_embeddings_per_sample.append(len(selected_embeddings))
                
                pred = individual_preds.mean(dim=0)  # [51]
                predictions.append(pred)
                
                mu_x = self.mu_xs[b, selected_indices[:, 0], selected_indices[:, 1], 0]
                mu_y = self.mu_ys[b, selected_indices[:, 0], selected_indices[:, 1], 0]
                sigma_x = self.sigma_xs[b, selected_indices[:, 0], selected_indices[:, 1], 0]
                sigma_y = self.sigma_ys[b, selected_indices[:, 0], selected_indices[:, 1], 0]
                rho = self.rhos[b, selected_indices[:, 0], selected_indices[:, 1], 0]
                

            focus_embedding_norm = fg_embeddings / (fg_embeddings.norm() + 1e-6)
            flat_embeddings_norm = bg_embeddings / (bg_embeddings.norm(dim=1, keepdim=True) + 1e-6)
            similarity_list.append(torch.matmul(flat_embeddings_norm, focus_embedding_norm))
            
            mask_indices = torch.nonzero(current_mask_downsampled)
            y_coords = mask_indices[:, 0].float() / H
            x_coords = mask_indices[:, 1].float() / W
            mu_x = mu_x.reshape(-1, 1).repeat(1, len(x_coords))
            mu_y = mu_y.reshape(-1, 1).repeat(1, len(x_coords))
            sigma_x = sigma_x.reshape(-1, 1).repeat(1, len(x_coords))
            sigma_y = sigma_y.reshape(-1, 1).repeat(1, len(x_coords))
            rho = rho.reshape(-1, 1).repeat(1, len(x_coords))
            term1 = torch.log(2 * math.pi * sigma_x * sigma_y * torch.sqrt(1 - rho**2))
            z_term = ((x_coords - mu_x)**2 / (sigma_x**2)) - \
                     (2 * rho * (x_coords - mu_x) * (y_coords - mu_y)) / (sigma_x * sigma_y) + \
                     ((y_coords - mu_y)**2 / (sigma_y**2))
            term2 = z_term / (2 * (1 - rho**2))
            nll_loss = (term1 + term2).mean()

            nll_loss_list.append(nll_loss)

        self.prediction = torch.stack(predictions)
        self.all_fg_embeddings_predictions = all_fg_embeddings_preds
        self.all_embeddings_predictions = all_embeddings_preds
        self.num_embeddings_per_sample = num_embeddings_per_sample
        
        return self.prediction, torch.cat(similarity_list), torch.stack(nll_loss_list)


class LightweightViT(nn.Module):
    def __init__(self, dim=384, depth=3, heads=6, mlp_dim=None):
        super(LightweightViT, self).__init__()
        from torch.nn import TransformerEncoderLayer, TransformerEncoder
        
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim or dim * 4
        
        encoder_layer = TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=self.mlp_dim,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        orig_shape = x.shape
        x = x.reshape(B, H * W, C)
        x = self.transformer(x)  # [B, H*W, C]
        x = self.norm(x)
        x = x.reshape(orig_shape)
        return x