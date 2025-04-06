import torch
import pdb
import random
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import torchvision.utils as vutils
from . import resnet, resnext, mobilenet, hrnetv2_nodownsp
from DynamicFocus_foveal_seg_classification.lib.nn import SynchronizedBatchNorm2d
from DynamicFocus_foveal_seg_classification.dataset import imresize, b_imresize
from builtins import any as b_any

from DynamicFocus_foveal_seg_classification.lib.utils import as_numpy
from DynamicFocus_foveal_seg_classification.utils import colorEncode
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
from DynamicFocus_foveal_seg_classification.saliency_network import saliency_network_resnet18, fov_simple, saliency_network_resnet18_stride1, FovSimModule
from DynamicFocus_foveal_seg_classification.models.model_utils import Resnet, ResnetDilated, MobileNetV2Dilated, C1DeepSup, C1, PPM, PPMDeepsup, UPerNet
from DynamicFocus_foveal_seg_classification.DynamicFocus.utility.torch_tools import gen_grid_mtx_2xHxW
BatchNorm2d = SynchronizedBatchNorm2d
from pytorch_toolbelt.losses.dice import DiceLoss

from torch.autograd import Variable
import matplotlib.pyplot as plt


class GaussianPredictor_old(nn.Module):
    def __init__(self, input_size=20, input_channels=386):
        super(GaussianPredictor_old, self).__init__()
        BN_MOMENTUM = 0.1
        self.input_size = input_size
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=5*20, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=5*20, out_channels=5*20, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=5*20, out_channels=16, kernel_size=3, stride=1, padding=1)
        #self.fc1 = nn.Linear(16*(input_size//4)*(input_size//4), 64)
        self.fc1 = nn.Linear(16*(input_size//4)*(input_size//4), 7)
        #self.fc2 = nn.Linear(64 + 2, 7)  # 修改为7个输出参数，增加了第二个高斯分布的sigma_x和sigma_y
        self.norm1 = BatchNorm2d(5*20, momentum=BN_MOMENTUM)
        self.norm2 = BatchNorm2d(5*20, momentum=BN_MOMENTUM)
        self.norm3 = BatchNorm2d(16, momentum=BN_MOMENTUM)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, gaze_coords=None):
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.norm2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.norm3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        #x = F.relu(self.fc1(x))
        x = self.fc1(self.dropout(x))
        
        # 如果提供了注视点坐标，将其连接到fc1的输出
        #if gaze_coords is not None:
        #    x = torch.cat([x, gaze_coords], dim=1)
            
        #x = self.fc2(x)
        return x

class GaussianPredictor(nn.Module):
    def __init__(self, input_size=20, input_channels=386):
        super(GaussianPredictor, self).__init__()
        self.input_size = input_size
        dim = 128  # 隐藏层维度
        num_heads = 4  # 多头注意力的头数
        num_layers = 3  # Transformer层数
        mlp_dim = 256  # MLP中间层维度
        dropout = 0.1  # dropout率
        
        # 映射到Transformer维度
        self.embedding = nn.Conv2d(input_channels, dim, kernel_size=1)
        
        # 添加特殊的[CLS]令牌用于全局表示
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        
        # Transformer编码器层
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
        
        # 输出层 - 预测7个高斯参数
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 7)
        self.dropout = nn.Dropout(dropout)
        
        # 注意力池化 - 学习如何汇总token特征
        self.attn_pool = nn.Linear(dim, 1)
        
        # 如果提供gaze_coords，则使用投影层
        self.gaze_proj = nn.Linear(2, dim)

    def forward(self, x, gaze_coords=None):
        B, C, H, W = x.shape
        
        # 映射到Transformer维度
        x = self.embedding(x)  # [B, dim, H, W]
        
        # 重塑为序列
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, dim]
        
        # 添加[CLS]令牌
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 1+H*W, dim]
        
        # 如果有gaze_coords，将其融合到序列中
        if gaze_coords is not None:
            gaze_emb = self.gaze_proj(gaze_coords)  # [B, dim]
            gaze_emb = gaze_emb.unsqueeze(1)  # [B, 1, dim]
            x = torch.cat([gaze_emb, x], dim=1)  # [B, 2+H*W, dim]
        
        # 通过Transformer层
        for layer in self.transformer_layers:
            x = layer(x)
        
        # 混合使用[CLS]令牌和注意力池化
        cls_feature = x[:, 0]  # 使用[CLS]令牌作为特征
        
        # 对其余token进行注意力池化
        if gaze_coords is not None:
            tokens = x[:, 2:]  # 跳过[CLS]和gaze tokens
        else:
            tokens = x[:, 1:]  # 仅跳过[CLS] token
            
        attn_weights = F.softmax(self.attn_pool(tokens).squeeze(-1), dim=1)  # [B, H*W]
        attn_feature = (tokens * attn_weights.unsqueeze(-1)).sum(dim=1)  # [B, dim]
        
        # 组合特征
        x = cls_feature + attn_feature  # 简单相加特征
        
        # 规范化和预测
        x = self.norm(x)
        x = self.head(self.dropout(x))  # [B, 7]
        
        return x

def generate_saliency_map(param, H=40, W=40, gaze_coords=None):
    B = param.size(0)
    
    # 第一个高斯分布参数
    mu_x = torch.sigmoid(param[:, 0]).view(B, 1, 1)
    mu_y = torch.sigmoid(param[:, 1]).view(B, 1, 1)
    sigma_x1 = (F.softplus(param[:, 2]) + 1e-6).view(B, 1, 1)
    sigma_y1 = (F.softplus(param[:, 3]) + 1e-6).view(B, 1, 1)
    rho = (torch.tanh(param[:, 4])).view(B, 1, 1)
    
    # 第二个高斯分布参数
    sigma_x2 = (F.softplus(param[:, 5]) + 1e-6).view(B, 1, 1)
    sigma_y2 = (F.softplus(param[:, 6]) + 1e-6).view(B, 1, 1)
    
    # 第二个高斯分布使用注视点作为均值
    if gaze_coords is not None:
        gaze_mu_x = gaze_coords[:, 1].view(B, 1, 1)  # 注视点x坐标
        gaze_mu_y = gaze_coords[:, 0].view(B, 1, 1)  # 注视点y坐标
    else:
        # 如果没有注视点信息，使用第一个高斯的均值
        gaze_mu_x = mu_x
        gaze_mu_y = mu_y

    x = torch.linspace(0, 1, W, device=param.device)
    y = torch.linspace(0, 1, H, device=param.device)

    xx, yy = torch.meshgrid(x, y, indexing='xy')
    xx = xx.unsqueeze(0).expand(B, H, W)
    yy = yy.unsqueeze(0).expand(B, H, W)

    # 计算第一个高斯分布
    norm_const1 = 1 / (2 * torch.pi * sigma_x1 * sigma_y1 * torch.sqrt(1 - rho**2))
    z_x1 = (xx - mu_x) / sigma_x1
    z_y1 = (yy - mu_y) / sigma_y1
    exponent1 = (z_x1**2 - 2 * rho * z_x1 * z_y1 + z_y1**2) / (2 * (1 - rho**2))
    saliency_map1 = norm_const1 * torch.exp(-exponent1)

    # 计算第二个高斯分布（无相关系数，即rho=0）
    norm_const2 = 1 / (2 * torch.pi * sigma_x2 * sigma_y2)
    z_x2 = (xx - gaze_mu_x) / sigma_x2
    z_y2 = (yy - gaze_mu_y) / sigma_y2
    exponent2 = (z_x2**2 + z_y2**2) / 2
    saliency_map2 = norm_const2 * torch.exp(-exponent2)

    # 将两个高斯分布相加
    saliency_map = saliency_map1 + saliency_map2
    
    return saliency_map


#def sample_gumbel(shape, eps=1e-20, device='cuda'):
#    U = torch.rand(shape, device=device)
#    return -torch.log(-torch.log(U + eps) + eps)

def differentiable_topk_mask(scores_BxHW, K, temperature):
    #gumbel_noise = sample_gumbel(scores_BxHW.shape, device='cuda')
    logits_BxHW = scores_BxHW# + 0.001*gumbel_noise
    soft_mask_BxHW = F.softmax(logits_BxHW / temperature, dim=1)
    _, indices = torch.topk(logits_BxHW, K, dim=1)
    hard_mask_BxHW = torch.zeros_like(logits_BxHW)
    hard_mask_BxHW.scatter_(1, indices, 1.0)
    # STE trick：forward 用硬 mask，但 backward 用 soft_mask 的梯度

    Lambda = logits_BxHW.shape[1]
    mask_BxHW = (hard_mask_BxHW - Lambda * soft_mask_BxHW).detach() + Lambda * soft_mask_BxHW
    #mask_BxHW = soft_mask_BxHW

    #mask_BxHW = (hard_mask_BxHW - soft_mask_BxHW).detach() + soft_mask_BxHW
    #if random.randint(0, 100) == 1:
    #    plt.imsave('../l_sam_experiment/selected_mask_{}.png'.format(random.randint(1,100)), mask_BxHW.detach().clone().view(8,40,40)[0,:,:].cpu().numpy(), cmap='gray')
    return mask_BxHW, indices

def differentiable_topk(feature_map_BxCxHxW, saliency_map_BxHxW, H_s=20, W_s=20, temperature=0.1):
    B, C, H, W = feature_map_BxCxHxW.shape
    K = H_s * W_s
    saliency_flat_BxHW = saliency_map_BxHxW.view(B, -1)
    mask_BxHW, indices = differentiable_topk_mask(saliency_flat_BxHW, K, temperature)
    feature_flat_BxCxHW = feature_map_BxCxHxW.view(B, C, -1)
    output_feature_map_BxCxHW = feature_flat_BxCxHW * mask_BxHW.unsqueeze(1)
    indices_exp = indices.unsqueeze(1).expand(B, C, K)
    selected_feature_map_BxCxK = torch.gather(output_feature_map_BxCxHW, 2, indices_exp)
    _, sort_order = torch.sort(indices, dim=1)
    sort_order_exp = sort_order.unsqueeze(1).expand(B, C, K)
    selected_feature_map_sorted_BxCxK = torch.gather(selected_feature_map_BxCxK, 2, sort_order_exp)
    indices = torch.gather(indices, 1, sort_order)
    return selected_feature_map_sorted_BxCxK.view(B, C, H_s, W_s), indices

class SoftDiceLossV1(nn.Module):
    '''
    soft-dice loss, useful in binary segmentation
    '''
    def __init__(self,
                 p=2,
                 smooth=0):
        super(SoftDiceLossV1, self).__init__()
        self.p = p
        self.smooth = smooth

    def forward(self, logits, labels):
        '''
        inputs:
            logits: tensor of shape (N, H, W, ...)
            label: tensor of shape(N, H, W, ...)
        output:
            loss: tensor of shape(1, )
        '''
        logits = torch.permute(logits, (0,2,3,1)).cuda()

        probs = torch.sigmoid(logits)
        #print(logits.shape, labels.shape)
        #print(probs.max(), probs.min())
        #print(labels.max())
        numer = (probs * labels).sum()
        denor = (probs.pow(self.p) + labels.pow(self.p)).sum()
        loss = 1. - (2 * numer + self.smooth) / (denor + self.smooth)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()
    
    def forward(self, y_pred_ds_Bx1xHSxWS):
        batch_size = y_pred_ds_Bx1xHSxWS.size(0)
        h = y_pred_ds_Bx1xHSxWS.size(2)
        w = y_pred_ds_Bx1xHSxWS.size(3)

        h_tv = torch.abs(y_pred_ds_Bx1xHSxWS[:, :, 1:, :] - y_pred_ds_Bx1xHSxWS[:, :, :-1, :]).sum()
        w_tv = torch.abs(y_pred_ds_Bx1xHSxWS[:, :, :, 1:] - y_pred_ds_Bx1xHSxWS[:, :, :, :-1]).sum()

        count_h = (y_pred_ds_Bx1xHSxWS.size(2) - 1) * y_pred_ds_Bx1xHSxWS.size(3)
        count_w = y_pred_ds_Bx1xHSxWS.size(2) * (y_pred_ds_Bx1xHSxWS.size(3) - 1)
        total_variation_loss = (h_tv / count_h + w_tv / count_w) / batch_size

        return total_variation_loss   

# class DiceLoss(nn.Module):
#     """Dice Loss PyTorch
#         Created by: Zhang Shuai
#         Email: shuaizzz666@gmail.com
#         dice_loss = 1 - 2*p*t / (p^2 + t^2). p and t represent predict and target.
#     Args:
#         weight: An array of shape [C,]
#         predict: A float32 tensor of shape [N, C, *], for Semantic segmentation task is [N, C, H, W]
#         target: A int64 tensor of shape [N, *], for Semantic segmentation task is [N, H, W]
#     Return:
#         diceloss
#     """
#     def __init__(self, weight=None):
#         super(DiceLoss, self).__init__()
#         if weight is not None:
#             weight = torch.Tensor(weight)
#             self.weight = weight / torch.sum(weight) # Normalized weight
#         self.smooth = 1e-5

#     def forward(self, predict, target):
#         N, C = predict.size()[:2]
#         predict = predict.view(N, C, -1) # (N, C, *)
#         target = target.view(N, 1, -1) # (N, 1, *)

#         predict = F.softmax(predict, dim=1) # (N, C, *) ==> (N, C, *)
#         ## convert target(N, 1, *) into one hot vector (N, C, *)
#         target_onehot = torch.zeros(predict.size()).cuda()  # (N, 1, *) ==> (N, C, *)
#         target_onehot.scatter_(1, target, 1)  # (N, C, *)

#         intersection = torch.sum(predict * target_onehot, dim=2)  # (N, C)
#         union = torch.sum(predict.pow(2), dim=2) + torch.sum(target_onehot, dim=2)  # (N, C)
#         ## p^2 + t^2 >= 2*p*t, target_onehot^2 == target_onehot
#         dice_coef = (2 * intersection + self.smooth) / (union + self.smooth)  # (N, C)

#         if hasattr(self, 'weight'):
#             if self.weight.type() != predict.type():
#                 self.weight = self.weight.type_as(predict)
#                 dice_coef = dice_coef * self.weight * C  # (N, C)
#         dice_loss = 1 - torch.mean(dice_coef)  # 1

#         return dice_loss
        
def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

def fillMissingValues_tensor(target_for_interp, copy=False, interp_mode='tri'):
    """
    fill missing values in a tenor

    input shape: [num_classes, h, w]
    output shape: [num_classes, h, w]
    """

    if copy:
        target_for_interp = target_for_interp.clone()

    def getPixelsForInterp(img):
        """
        Calculates a mask of pixels neighboring invalid values -
           to use for interpolation.

        input shape: [num_classes, h, w]
        output shape: [num_classes, h, w]
        """

        invalid_mask = torch.isnan(img)
        kernel = torch.tensor(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), device=invalid_mask.device).unsqueeze(0).unsqueeze(0).expand(1,invalid_mask.shape[0],3,3).float()

        #dilate to mark borders around invalid regions
        if max(invalid_mask.shape) > 512:
            dr = max(invalid_mask.shape)/512
            input = invalid_mask.float().unsqueeze(0)
            shape_ori = (invalid_mask.shape[-2], int(invalid_mask.shape[-1]))
            shape_scaled = (int(invalid_mask.shape[-2]/dr), int(invalid_mask.shape[-1]/dr))
            input_scaled = F.interpolate(input, shape_scaled, mode='nearest').squeeze(0)
            invalid_mask_scaled = input_scaled.unsqueeze(0) # b,c,w,h

            dilated_mask_scaled = torch.clamp(F.conv2d(invalid_mask_scaled, kernel, padding=(1, 1)), 0, 1)
            dilated_mask_scaled_t = dilated_mask_scaled.float()
            dilated_mask = F.interpolate(dilated_mask_scaled_t, shape_ori, mode='nearest').squeeze(0)
        else:

            dilated_mask = torch.clamp(F.conv2d(invalid_mask.float().unsqueeze(0),
                                                kernel, padding=(1, 1)), 0, 1).squeeze(0)

        # pixelwise "and" with valid pixel mask (~invalid_mask)
        masked_for_interp = dilated_mask *  (~invalid_mask).float()
        # Add 4 zeros corner points required for interp2d
        masked_for_interp[:,0,0] *= 0
        masked_for_interp[:,0,-1] *= 0
        masked_for_interp[:,-1,0] *= 0
        masked_for_interp[:,-1,-1] *= 0
        masked_for_interp[:,0,0] += 1
        masked_for_interp[:,0,-1] += 1
        masked_for_interp[:,-1,0] += 1
        masked_for_interp[:,-1,-1] += 1

        return masked_for_interp.bool(), invalid_mask

    def getPixelsForInterp_NB(img):
        """
        Calculates a mask of pixels neighboring invalid values -
           to use for interpolation.
        """
        # mask invalid pixels
        img = img.cpu().numpy()
        invalid_mask = np.isnan(img)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        #dilate to mark borders around invalid regions
        if max(invalid_mask.shape) > 512:
            dr = max(invalid_mask.shape)/512
            input = torch.tensor(invalid_mask.astype('float')).unsqueeze(0)
            shape_ori = (invalid_mask.shape[-2], int(invalid_mask.shape[-1]))
            shape_scaled = (int(invalid_mask.shape[-2]/dr), int(invalid_mask.shape[-1]/dr))
            input_scaled = F.interpolate(input, shape_scaled, mode='nearest').squeeze(0)
            invalid_mask_scaled = np.array(input_scaled).astype('bool')
            dilated_mask_scaled = cv2.dilate(invalid_mask_scaled.astype('uint8'), kernel,
                              borderType=cv2.BORDER_CONSTANT, borderValue=int(0))
            dilated_mask_scaled_t = torch.tensor(dilated_mask_scaled.astype('float')).unsqueeze(0)
            dilated_mask = F.interpolate(dilated_mask_scaled_t, shape_ori, mode='nearest').squeeze(0)
            dilated_mask = np.array(dilated_mask).astype('uint8')
        else:
            dilated_mask = cv2.dilate(invalid_mask.astype('uint8'), kernel,
                              borderType=cv2.BORDER_CONSTANT, borderValue=int(0))

        # pixelwise "and" with valid pixel mask (~invalid_mask)
        masked_for_interp = dilated_mask *  ~invalid_mask
        return masked_for_interp.astype('bool'), invalid_mask

    # Mask pixels for interpolation
    if interp_mode == 'nearest':
        interpolator=scipy.interpolate.NearestNDInterpolator
        mask_for_interp, invalid_mask = getPixelsForInterp_NB(target_for_interp)
    elif interp_mode == 'BI':
        interpolator=scipy.interpolate.LinearNDInterpolator
        mask_for_interp, invalid_mask = getPixelsForInterp_NB(target_for_interp)
    else:
        # interpolator=Interp2D(target_for_interp.shape[-2], target_for_interp.shape[-1])
        mask_for_interp, invalid_mask = getPixelsForInterp(target_for_interp)
        if invalid_mask.float().sum() == 0:
            return target_for_interp

    if interp_mode == 'nearest' or interp_mode == 'BI':
        #print('1111111111111111')
        points = np.argwhere(mask_for_interp)  #np array
        #values = target_for_interp[mask_for_interp] #tensor
        values = target_for_interp[mask_for_interp].cpu().numpy() #np
        #print(type(points), type(values))
    else:
        print('222222222222222222')
        points = torch.where(mask_for_interp[0]) # tuple of 2 for (h, w) indices
        points = torch.cat([t.unsqueeze(0) for t in points]) # [2, number_of_points]
        points = points.permute(1,0) # shape: [number_of_points, 2]
        values = target_for_interp.clone()[mask_for_interp].view(mask_for_interp.shape[0],-1).permute(1,0) # shape: [number_of_points, num_classes]
    interp = interpolator(points, values) # return [num_classes, h, w]
    if interp_mode == 'nearest' or interp_mode == 'BI':
        #print('227')
        target_for_interp[invalid_mask] = torch.tensor(interp(np.argwhere(np.array(invalid_mask)))).float().cuda()
        #print('227 done')
    else:
        if not (interp.shape == target_for_interp.shape == invalid_mask.shape and interp.device == target_for_interp.device == invalid_mask.device):
            print('SHAPE: interp={}; target_for_interp={}; invalid_mask={}\n'.format(interp.shape, target_for_interp.shape, invalid_mask.shape))
            print('DEVICE: interp={}; target_for_interp={}; invalid_mask={}\n'.format(interp.device, target_for_interp.device, invalid_mask.device))
        try:
            target_for_interp[invalid_mask] = interp[torch.where(invalid_mask)].clone()
        except:
            print('interp: {}\n'.format(interp))
            print('invalid_mask: {}\n'.format(invalid_mask))
            print('target_for_interp: {}\n'.format(target_for_interp))
        else:
            pass
    return target_for_interp

def create_map(B, hidx_B, widx_B, height=80, width=80, radius=25, max_value=0.5, min_value=0.05):
    """
    Creates a map with shape (B, 1, height, width). Each batch has a point at (hidx, widx)
    with value max_value, decreasing in a cosine style to min_value within a radius.

    Parameters:
        B (int): Batch size.
        hidx_B (torch.Tensor): Tensor of shape (B, 1), y-coordinates.
        widx_B (torch.Tensor): Tensor of shape (B, 1), x-coordinates.
        height (int): Height of the map (default=80).
        width (int): Width of the map (default=80).
        radius (int): Radius within which the value decreases (default=25).
        max_value (float): Maximum value at the center (default=0.0025).
        min_value (float): Minimum value at the edge of the radius (default=2.5e-6).

    Returns:
        torch.Tensor: A map of shape (B, 1, height, width).
    """
    # Initialize the map with zeros
    map_tensor = torch.zeros((B, 1, height, width))+2.5e-6

    # Create a grid for distance calculation
    y_grid = torch.arange(height).view(1, height, 1).repeat(1, 1, width)
    x_grid = torch.arange(width).view(1, 1, width).repeat(1, height, 1)

    # Iterate over each batch
    for b in range(B):
        # Extract coordinates for the current batch
        h, w = hidx_B[b].item(), widx_B[b].item()

        # Compute the distance from the (h, w) point
        distance = torch.sqrt((y_grid - h)**2 + (x_grid - w)**2)

        # Cosine decay formula
        decay = 0.5 * (1 + torch.cos(torch.clamp(distance / radius * torch.pi, max=torch.pi)))
        #decay = torch.exp(-distance / radius)

        # Scale decay to the specified range [min_value, max_value]
        scaled_decay = min_value + (max_value - min_value) * decay
        map_tensor[b, 0] = torch.where(distance <= radius, scaled_decay, min_value)

    return map_tensor

def smooth_map_with_gaussian(map_tensor, sigma=3):
    """
    Smooths the input map using a Gaussian filter.

    Parameters:
        map_tensor (torch.Tensor): The map tensor of shape (B, 1, H, W).
        sigma (int): Standard deviation for the Gaussian filter.

    Returns:
        torch.Tensor: Smoothed map.
    """
    # Define the size of the Gaussian kernel based on sigma
    kernel_size = 2 * int(3 * sigma) + 1  # Covers ±3 standard deviations
    # Create 1D Gaussian kernel
    x = torch.arange(kernel_size) - kernel_size // 2
    gauss_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    gauss_1d = gauss_1d / gauss_1d.sum()  # Normalize

    # Create 2D Gaussian kernel
    gauss_2d = torch.outer(gauss_1d, gauss_1d).unsqueeze(0).unsqueeze(0)

    B = map_tensor.shape[0]
    gauss_2d = gauss_2d.expand(1, 1, kernel_size, kernel_size)
    gauss_2d = gauss_2d.to(map_tensor.device)

    # Apply Gaussian filter to each map
    smoothed_map = F.conv2d(map_tensor, gauss_2d, padding=kernel_size // 2, groups=1)
    return smoothed_map

class CompressNet(nn.Module):
    def __init__(self):
        super(CompressNet, self).__init__()
        # if cfg.MODEL.saliency_net == 'fovsimple':
        self.conv_last = nn.Conv2d(24,1,kernel_size=1,padding=0,stride=1)
        # else:
        # self.conv_last = nn.Conv2d(256,1,kernel_size=1,padding=0,stride=1)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.act(x)
        out = self.conv_last(x)
        return out

class SegmentationModuleBase(nn.Module):
    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

    def pixel_acc(self, pred_all, label_all):
        # accuracy with forground class
        torch.set_printoptions(threshold=10000)
        bs = pred_all.shape[0]
        acc_accu = 0.
        for i in range(bs):
            pred, label= pred_all[i:i+1, :,:], label_all[i:i+1, :, :] #pred: BxCxHxW  label: BxHxW
            #print('pred shape', pred.shape)
            _, preds = torch.max(pred, dim=1) #BxHxW
            #valid = (label > 0).long()     #bg is class 0
            #valid1 = (preds > 0).long()
            valid = (label < 50).long()
            valid1 = (preds < 50).long()
            acc_sum = torch.sum(valid * (preds == label).long())   # this is the intersectory pixels based on class
            #acc_sum = torch.sum(valid * (valid == valid1).long()) #binary intersection
            pixel_sum = torch.sum(valid)   # this is the summation of number of ground truth pixel
            pixel_sum1 = torch.sum(valid1)  # this is the summation of number of predicted true pixel
            pixel_sum_final = ((valid + valid1) > 0).sum().long()
            acc = acc_sum.float() / (pixel_sum_final.float() + 1e-10)
            acc_accu += acc
            #print(acc_sum.float(), pixel_sum.float(), pixel_sum_final.float(), pixel_sum1.float(), acc)   #tensor(6241., device='cuda:0') tensor(13009., device='cuda:0') tensor(21388., device='cuda:0') tensor(16672., device='cuda:0') tensor(0.2918, device='cuda:0')
        return acc_accu/bs
    
    def fg_bin_pixel_acc(self, pred_all, label_all):
        # accuracy with foreground binary
        torch.set_printoptions(threshold=10000)
        bs = pred_all.shape[0]
        acc_accu = 0.
        for i in range(bs):
            pred, label= pred_all[i:i+1, :,:], label_all[i:i+1, :, :] #pred: BxCxHxW  label: BxHxW
            #print('pred shape', pred.shape)
            _, preds = torch.max(pred, dim=1) #BxHxW
            #valid = (label > 0).long()
            #valid1 = (preds > 0).long()
            valid = (label < 50).long()
            valid1 = (preds < 50).long()
            acc_sum = torch.sum(valid * (valid == valid1).long())   # this is the intersectory pixels based on binary
            #acc_sum = torch.sum(valid * (valid == valid1).long()) #binary intersection
            pixel_sum = torch.sum(valid)   # this is the summation of number of ground truth pixel
            pixel_sum1 = torch.sum(valid1)  # this is the summation of number of predicted true pixel
            pixel_sum_final = ((valid + valid1)> 0).sum().long()
            acc = acc_sum.float() / (pixel_sum_final.float() + 1e-10)
            acc_accu += acc
            #print(acc_sum.float(), pixel_sum.float(), pixel_sum_final.float(), pixel_sum1.float(), acc)   #tensor(6241., device='cuda:0') tensor(13009., device='cuda:0') tensor(21388., device='cuda:0') tensor(16672., device='cuda:0') tensor(0.2918, device='cuda:0')
        return acc_accu/bs
    
    def fbg_cls_pixel_acc(self, pred_all, label_all):
        # accuracy with foreground binary
        torch.set_printoptions(threshold=10000)
        bs = pred_all.shape[0]
        acc_accu = 0.
        for i in range(bs):
            pred, label= pred_all[i:i+1, :,:], label_all[i:i+1, :, :] #pred: BxCxHxW  label: BxHxW
            _, preds = torch.max(pred, dim=1) #BxHxW
            #print(f'unique number in pred {torch.unique(preds)}')
            #print(f'unique number in label {torch.unique(label)}')
            pred_unique = torch.unique(preds)
            # if pred_unique.numel() > 1:
            #     pred_fg = pred_unique[pred_unique != 50].item()
            # else:
            #     pred_fg = pred_unique.item()
            #pred_fg = pred_unique[pred_unique != 50].item()
            label_unique = torch.unique(label)
            # if label_unique.numel() > 1:
            #     label_fg = label_unique[label_unique != 50].item()
            # else:
            #     label_fg = label_unique.item()
            #label_fg = label_unique[label_unique!= 50].item()
            #valid_fg = (label >= 0).long()
            #valid1_fg = (preds >= 0).long()
            valid_fg = (label < 50).long()
            valid1_fg = (preds < 50).long()
            acc_sum_fg = torch.sum(valid_fg * (label == preds).long())   # this is the intersectory pixels based on binary
            #acc_sum = torch.sum(valid * (valid == valid1).long()) #binary intersection
            pixel_sum_final_fg = ((valid_fg + valid1_fg)> 0).sum().long()
            acc_fg = acc_sum_fg.float() / (pixel_sum_final_fg.float() + 1e-10)

            valid_bg = (label == 50).long()
            valid1_bg = (preds== 50).long()
            acc_sum_bg = torch.sum(valid_bg * (label == preds).long())   # this is the intersectory pixels based on binary
            #acc_sum = torch.sum(valid * (valid == valid1).long()) #binary intersection
            pixel_sum_final_bg = ((valid_bg + valid1_bg)> 0).sum().long()
            acc_bg = acc_sum_bg.float() / (pixel_sum_final_bg.float() + 1e-10)

            acc_accu += acc_fg*0.5+acc_bg*0.5
            #acc_accu += torch.tensor(pred_fg == label_fg).float() 
            #print(acc_sum.float(), pixel_sum.float(), pixel_sum_final.float(), pixel_sum1.float(), acc)   #tensor(6241., device='cuda:0') tensor(13009., device='cuda:0') tensor(21388., device='cuda:0') tensor(16672., device='cuda:0') tensor(0.2918, device='cuda:0')
        return acc_accu/bs
    
    def fbg_bin_pixel_acc(self, pred_all, label_all):
        # accuracy with foreground binary
        torch.set_printoptions(threshold=10000)
        bs = pred_all.shape[0]
        acc_accu = 0.
        for i in range(bs):
            pred, label= pred_all[i:i+1, :,:], label_all[i:i+1, :, :] #pred: BxCxHxW  label: BxHxW
            #print('pred shape', pred.shape)
            _, preds = torch.max(pred, dim=1) #BxHxW
            valid_fg = (label < 50).long()
            valid1_fg = (preds < 50).long()
            acc_sum_fg = torch.sum(valid_fg * (valid_fg == valid1_fg).long())   # this is the intersectory pixels based on binary
            #acc_sum = torch.sum(valid * (valid == valid1).long()) #binary intersection
            pixel_sum_final_fg = ((valid_fg + valid1_fg)> 0).sum().long()
            acc_fg = acc_sum_fg.float() / (pixel_sum_final_fg.float() + 1e-10)
            
            valid_bg = (label == 50).long()
            valid1_bg = (preds == 50).long()
            acc_sum_bg = torch.sum(valid_bg * (valid_bg == valid1_bg).long())   # this is the intersectory pixels based on binary
            #acc_sum = torch.sum(valid * (valid == valid1).long()) #binary intersection
            pixel_sum_final_bg = ((valid_bg + valid1_bg)> 0).sum().long()
            acc_bg = acc_sum_bg.float() / (pixel_sum_final_bg.float() + 1e-10)
            acc_accu += acc_fg*0.5+acc_bg*0.5
            #print(acc_sum.float(), pixel_sum.float(), pixel_sum_final.float(), pixel_sum1.float(), acc)   #tensor(6241., device='cuda:0') tensor(13009., device='cuda:0') tensor(21388., device='cuda:0') tensor(16672., device='cuda:0') tensor(0.2918, device='cuda:0')
        return acc_accu/bs
    
class DeformSegmentationModule(nn.Module):
    def __init__(self, cfg):
        super(DeformSegmentationModule, self).__init__()
        self.gaussian_predictor = GaussianPredictor(40, 386)
        self.wang_localization = FovSimModule(cfg)
        self.temp_idx = 0
        self.grid_size_x = 20
        self.grid_size_y = 20
        # wang ---------------
        self.padding_size_y = 10
        self.padding_size_x = 10
        # wang ---------------

        #self.crit = crit   # change here!!!!!!!!!!
        #self.crit = FocalLoss(gamma=7.0)
        # self.crit = DiceLoss('multiclass')
        #self.crit = SoftDiceLossV1()

        # if cfg.TRAIN.opt_deform_LabelEdge or cfg.TRAIN.deform_joint_loss:
        #     self.crit_mse = nn.MSELoss()
        self.cfg = cfg
        # self.print_original_y = True
        self.wang_net_compress = CompressNet()
        # if self.cfg.MODEL.saliency_output_size_short == 0:
        #     self.grid_size_x = cfg.TRAIN.saliency_input_size[0]
        # else:
        #     self.grid_size_x = self.cfg.MODEL.saliency_output_size_short
        # self.grid_size_y = cfg.TRAIN.saliency_input_size[1] // (cfg.TRAIN.saliency_input_size[0]//self.grid_size_x)
        # self.padding_size_x = self.cfg.MODEL.gaussian_radius
        # if self.cfg.MODEL.gaussian_ap == 0.0:
        #     gaussian_ap = cfg.TRAIN.saliency_input_size[1] // cfg.TRAIN.saliency_input_size[0]
        # else:
        #     gaussian_ap = self.cfg.MODEL.gaussian_ap
        # self.padding_size_y = int(gaussian_ap * self.padding_size_x)
        self.global_size_x = self.grid_size_x+2*self.padding_size_x
        self.global_size_y = self.grid_size_y+2*self.padding_size_y
        # self.input_size = cfg.TRAIN.saliency_input_size
        self.input_size_net = (20,20)
        self.input_size_net_eval = (20,20)
        # #print('here!!!!!!!!!!!!', cfg.TRAIN.task_input_size_eval)
        # if len(self.input_size_net_eval) == 0:
        #     self.input_size_net_infer = self.input_size_net
        # else:
        self.input_size_net_infer = (40,40)
        #wang ----------------
        gaussian_weights = torch.FloatTensor(makeGaussian(2*self.padding_size_x+1, fwhm = 8)) # TODO: redo seneitivity experiments on gaussian radius as corrected fwhm (effective gaussian radius)
        #wang ----------------
        gaussian_weights = b_imresize(gaussian_weights.unsqueeze(0).unsqueeze(0), (2*self.padding_size_x+1,2*self.padding_size_y+1), interp='bilinear')
        gaussian_weights = gaussian_weights.squeeze(0).squeeze(0)

        self.filter = nn.Conv2d(1, 1, kernel_size=(2*self.padding_size_x+1,2*self.padding_size_y+1),bias=False)
        self.filter.weight[0].data[:,:,:] = gaussian_weights

        self.P_basis = torch.zeros(2,self.grid_size_x+2*self.padding_size_x, self.grid_size_y+2*self.padding_size_y).cuda()
        # initialization of u(x,y),v(x,y) range from 0 to 1
        for k in range(2):
            for i in range(self.global_size_x):
                for j in range(self.global_size_y):
                    self.P_basis[k,i,j] = k*(i-self.padding_size_x)/(self.grid_size_x-1.0)+(1.0-k)*(j-self.padding_size_y)/(self.grid_size_y-1.0)

        # self.save_print_grad = [{'saliency_grad': 0.0, 'check1_grad': 0.0, 'check2_grad': 0.0} for _ in range(cfg.TRAIN.num_gpus)]
    
    def unorm(self, img):
        if 'GLEASON' in self.cfg.DATASET.list_train:
            mean=[0.748, 0.611, 0.823]
            std=[0.146, 0.245, 0.119]
        elif 'Digest' in self.cfg.DATASET.list_train:
            mean=[0.816, 0.697, 0.792]
            std=[0.160, 0.277, 0.198]
        elif 'ADE' in self.cfg.DATASET.list_train:
            mean=[0.485, 0.456, 0.406]
            std=[0.229, 0.224, 0.225]
        elif 'CITYSCAPE' in self.cfg.DATASET.list_train or 'Cityscape' in self.cfg.DATASET.list_train:
            mean=[0.485, 0.456, 0.406]
            std=[0.229, 0.224, 0.225]
        elif 'histo' in self.cfg.DATASET.list_train:
            mean=[0.8223, 0.7783, 0.7847]
            std=[0.210, 0.216, 0.241]
        elif 'DeepGlob' in self.cfg.DATASET.list_train or 'deepglob' in self.cfg.DATASET.root_dataset:
            mean=[0.282, 0.379, 0.408]
            std=[0.089, 0.101, 0.127]
        elif 'Face_single_example' in self.cfg.DATASET.root_dataset or 'Face_single_example' in self.cfg.DATASET.list_train:
            mean=[0.282, 0.379, 0.408]
            std=[0.089, 0.101, 0.127]
        elif 'Histo' in self.cfg.DATASET.root_dataset or 'histomri' in self.cfg.DATASET.list_train or 'histomri' in self.cfg.DATASET.root_dataset:
            mean=[0.8223, 0.7783, 0.7847]
            std=[0.210, 0.216, 0.241]
        else:
            raise Exception('Unknown root for normalisation!')
        for t, m, s in zip(img, mean, std):
            t.mul_(s).add_(m)
        return img
    
    def re_initialise(self, cfg, this_size):# dealing with varying input image size such as pcahisto dataset
        this_size_short = min(this_size)
        this_size_long = max(this_size)
        scale_task_rate_1 = this_size_short // min(cfg.TRAIN.dynamic_task_input)
        scale_task_rate_2 = this_size_long // max(cfg.TRAIN.dynamic_task_input)
        scale_task_size_1 = tuple([int(x//scale_task_rate_1) for x in this_size])
        scale_task_size_2 = tuple([int(x//scale_task_rate_2) for x in this_size])
        if scale_task_size_1[0]*scale_task_size_1[1] < scale_task_size_2[0]*scale_task_size_2[1]:
            scale_task_size = scale_task_size_1
        else:
            scale_task_size = scale_task_size_2

        cfg.TRAIN.task_input_size = scale_task_size
        cfg.TRAIN.saliency_input_size = tuple([int(x*cfg.TRAIN.dynamic_saliency_relative_size) for x in scale_task_size])

        if self.cfg.MODEL.saliency_output_size_short == 0:
            self.grid_size_x = cfg.TRAIN.saliency_input_size[0]
        else:
            self.grid_size_x = self.cfg.MODEL.saliency_output_size_short
        self.grid_size_y = cfg.TRAIN.saliency_input_size[1] // (cfg.TRAIN.saliency_input_size[0]//self.grid_size_x)
        self.global_size_x = self.grid_size_x+2*self.padding_size_x
        self.global_size_y = self.grid_size_y+2*self.padding_size_y

        self.input_size = cfg.TRAIN.saliency_input_size
        self.input_size_net = cfg.TRAIN.task_input_size

        if len(self.input_size_net_eval) == 0:
            self.input_size_net_infer = self.input_size_net
        else:
            self.input_size_net_infer = self.input_size_net_eval
        self.P_basis = torch.zeros(2,self.grid_size_x+2*self.padding_size_x, self.grid_size_y+2*self.padding_size_y).cuda()
        # initialization of u(x,y),v(x,y) range from 0 to 1
        for k in range(2):
            for i in range(self.global_size_x):
                for j in range(self.global_size_y):
                    self.P_basis[k,i,j] = k*(i-self.padding_size_x)/(self.grid_size_x-1.0)+(1.0-k)*(j-self.padding_size_y)/(self.grid_size_y-1.0)

    # wang ----------------
    def inverse_grid_sample(self, output, grid, mode='bilinear', align_corners=None, kernel_size=5, sigma=1.0):
        B, C, H_out, W_out = output.shape
        H_in, W_in = H_out * 2, W_out * 2
        device = output.device
        N = H_out * W_out

        # 将归一化 grid 映射到高分辨率坐标
        if align_corners:
            x_src = (grid[..., 0] + 1) / 2 * (W_in - 1)
            y_src = (grid[..., 1] + 1) / 2 * (H_in - 1)
        else:
            x_src = (grid[..., 0] + 1) * 0.5 * W_in - 0.5
            y_src = (grid[..., 1] + 1) * 0.5 * H_in - 0.5
        x_src_flat = x_src.reshape(B, N)
        y_src_flat = y_src.reshape(B, N)

        # 基础索引：以 floor 为中心
        base_x = torch.floor(x_src_flat)  # [B, N]
        base_y = torch.floor(y_src_flat)  # [B, N]

        # 构造二维 offset 网格
        k = kernel_size
        r = (k - 1) // 2
        offsets = torch.arange(-r, r+1, device=device).float()  # [k]
        dx, dy = torch.meshgrid(offsets, offsets, indexing='ij')   # both: [k, k]

        # 生成完整 k×k 邻域候选索引，形状 [B, k, k, N]
        candidate_x = base_x[:, None, None, :] + dx[None, :, :, None]
        candidate_y = base_y[:, None, None, :] + dy[None, :, :, None]
        candidate_x = candidate_x.clamp(0, W_in - 1)
        candidate_y = candidate_y.clamp(0, H_in - 1)

        # 计算候选点与真实采样位置的欧氏距离平方，并计算高斯权重
        dist_sq = (candidate_x - x_src_flat[:, None, None, :])**2 + (candidate_y - y_src_flat[:, None, None, :])**2
        weight = torch.exp(-dist_sq / (2 * sigma**2))  # [B, k, k, N]

        # 展平候选窗口：形状 [B, k*k, N]
        weight_flat = weight.view(B, k*k, N)
        candidate_x_flat = candidate_x.view(B, k*k, N)
        candidate_y_flat = candidate_y.view(B, k*k, N)

        # 构造降采样时 grid 的归一化坐标，形状 [N]
        if align_corners:
            j_coords = torch.linspace(-1, 1, W_out, device=device)
            i_coords = torch.linspace(-1, 1, H_out, device=device)
        else:
            j_coords = (2 * (torch.arange(W_out, device=device).float() + 0.5) / W_out - 1)
            i_coords = (2 * (torch.arange(H_out, device=device).float() + 0.5) / H_out - 1)
        X_norm_flat = j_coords.repeat(H_out)    # [N]
        Y_norm_flat = i_coords.unsqueeze(1).repeat(1, W_out).reshape(-1)  # [N]

        # 扩展归一化坐标到候选维度 [B, k*k, N]
        X_norm_exp = X_norm_flat.unsqueeze(0).unsqueeze(1).expand(B, k*k, N)
        Y_norm_exp = Y_norm_flat.unsqueeze(0).unsqueeze(1).expand(B, k*k, N)
        contr0 = weight_flat * X_norm_exp  # [B, k*k, N]
        contr1 = weight_flat * Y_norm_exp  # [B, k*k, N]

        # 利用 scatter_add 将候选贡献累加到高分辨率空间中
        M = H_in * W_in
        inv_grid_acc = torch.zeros(B, 2, M, device=device)
        weight_sum_acc = torch.zeros(B, M, device=device)
        # 转换候选索引为线性索引（long 类型）
        linear_idx = (candidate_y_flat.long() * W_in + candidate_x_flat.long()).view(B, -1)
        weight_flat_all = weight_flat.view(B, -1)
        contr0_flat = contr0.view(B, -1)
        contr1_flat = contr1.view(B, -1)
        weight_sum_acc.scatter_add_(1, linear_idx, weight_flat_all)
        inv_grid_acc[:, 0, :].scatter_add_(1, linear_idx, contr0_flat)
        inv_grid_acc[:, 1, :].scatter_add_(1, linear_idx, contr1_flat)
        weight_sum_acc = weight_sum_acc.clamp_min(1e-6)
        inv_grid_acc[:, 0, :].div_(weight_sum_acc)
        inv_grid_acc[:, 1, :].div_(weight_sum_acc)

        # 重塑为逆向采样 grid，形状 [B, H_in, W_in, 2]，并利用 grid_sample 得到重构图像
        inv_grid = inv_grid_acc.view(B, 2, H_in, W_in).permute(0, 2, 3, 1).clamp(-1, 1)
        return F.grid_sample(output, inv_grid, mode=mode, align_corners=align_corners)
    # wang ----------------

    def create_grid(self, x, segSize=None, x_inv=None):
        #print('grid size', self.grid_size_x, self.grid_size_y)
        P = torch.autograd.Variable(torch.zeros(1,2,self.grid_size_x+2*self.padding_size_x, self.grid_size_y+2*self.padding_size_y, device=x.device),requires_grad=False)

        P[0,:,:,:] = self.P_basis.to(x.device) # [1,2,w,h], 2 corresponds to u(x,y) and v(x,y)
        P = P.expand(x.size(0),2,self.grid_size_x+2*self.padding_size_x, self.grid_size_y+2*self.padding_size_y)
        # input x is saliency map xs
        x_cat = torch.cat((x,x),1)
        # EXPLAIN: denominator of Eq. (3)
        p_filter = self.filter(x)
        x_mul = torch.mul(P,x_cat).view(-1,1,self.global_size_x,self.global_size_y)
        all_filter = self.filter(x_mul).view(-1,2,self.grid_size_x,self.grid_size_y)
        # EXPLAIN: numerator of Eq. (3)
        x_filter = all_filter[:,0,:,:].contiguous().view(-1,1,self.grid_size_x,self.grid_size_y)
        y_filter = all_filter[:,1,:,:].contiguous().view(-1,1,self.grid_size_x,self.grid_size_y)
        # EXPLAIN: Eq. (3)
        x_filter = x_filter/p_filter
        y_filter = y_filter/p_filter
        # EXPLAIN: fit F.grid_sample format (coordibates in the range [-1,1])
        xgrids = x_filter*2-1
        ygrids = y_filter*2-1
        xgrids = torch.clamp(xgrids,min=-1,max=1)
        ygrids = torch.clamp(ygrids,min=-1,max=1)
        # EXPLAIN: reshape
        xgrids = xgrids.view(-1,1,self.grid_size_x,self.grid_size_y)
        ygrids = ygrids.view(-1,1,self.grid_size_x,self.grid_size_y)
        grid = torch.cat((xgrids,ygrids),1)

        if segSize is not None:# inference
            
            grid = nn.Upsample(size=(20,20), mode='bilinear')(grid)
            #print('grid shape', grid.shape)
        else:
            #print('input_size_net shape', self.input_size_net)
            grid = nn.Upsample(size=self.input_size_net, mode='bilinear')(grid)
            #print('grid shape', grid.shape)
        # EXPLAIN: grid_y for downsampling label y, to handle segmentation architectures whose prediction are not same size with input x
        # if segSize is None:# training
        #     #print('here!!!!!!!!!!!!!!!!!', self.cfg.DATASET.segm_downsampling_rate)
        #     grid_y = nn.Upsample(size=tuple(np.array(self.input_size_net)//self.cfg.DATASET.segm_downsampling_rate), mode='bilinear')(grid)
        #     #print('grid_y shape', grid_y.shape)
        # else:# inference
        grid_y = nn.Upsample(size=(20,20), mode='bilinear')(grid)

        grid = torch.transpose(grid,1,2)
        grid = torch.transpose(grid,2,3)

        grid_y = torch.transpose(grid_y,1,2)
        grid_y = torch.transpose(grid_y,2,3)

        # return grid, grid_y


        #### inverse deformation
        # if segSize is not None and x_inv is not None:
        #     grid_reorder = grid.permute(3,0,1,2)
        #     grid_inv = torch.autograd.Variable(torch.zeros((2,grid_reorder.shape[1],segSize[0],segSize[1]), device=grid_reorder.device))
        #     grid_inv[:] = float('nan')
        #     u_cor = (((grid_reorder[0,:,:,:]+1)/2)*(segSize[1]-1)).int().long().view(grid_reorder.shape[1],-1)
        #     v_cor = (((grid_reorder[1,:,:,:]+1)/2)*(segSize[0]-1)).int().long().view(grid_reorder.shape[1],-1)
        #     x_cor = torch.arange(0,grid_reorder.shape[3], device=grid_reorder.device).unsqueeze(0).expand((grid_reorder.shape[2],grid_reorder.shape[3])).reshape(-1)
        #     x_cor = x_cor.unsqueeze(0).expand(u_cor.shape[0],-1).float()
        #     y_cor = torch.arange(0,grid_reorder.shape[2], device=grid_reorder.device).unsqueeze(-1).expand((grid_reorder.shape[2],grid_reorder.shape[3])).reshape(-1)
        #     y_cor = y_cor.unsqueeze(0).expand(u_cor.shape[0],-1).float()
        #     grid_inv[0][torch.arange(grid_reorder.shape[1]).unsqueeze(-1),v_cor,u_cor] = torch.autograd.Variable(x_cor)
        #     grid_inv[1][torch.arange(grid_reorder.shape[1]).unsqueeze(-1),v_cor,u_cor] = torch.autograd.Variable(y_cor)
        #     grid_inv[0] = grid_inv[0]/grid_reorder.shape[3]*2-1
        #     grid_inv[1] = grid_inv[1]/grid_reorder.shape[2]*2-1
        #     grid_inv = grid_inv.permute(1,2,3,0)
        #     return grid, grid_inv
        # else:
        #     return grid, grid_y

        grid_reorder = grid.permute(3,0,1,2)
        grid_inv = torch.autograd.Variable(torch.zeros((2,grid_reorder.shape[1],segSize[0],segSize[1]), device=grid_reorder.device))
        grid_inv[:] = float('nan')
        u_cor = (((grid_reorder[0,:,:,:]+1)/2)*(segSize[1]-1)).int().long().view(grid_reorder.shape[1],-1)
        v_cor = (((grid_reorder[1,:,:,:]+1)/2)*(segSize[0]-1)).int().long().view(grid_reorder.shape[1],-1)
        x_cor = torch.arange(0,grid_reorder.shape[3], device=grid_reorder.device).unsqueeze(0).expand((grid_reorder.shape[2],grid_reorder.shape[3])).reshape(-1)
        x_cor = x_cor.unsqueeze(0).expand(u_cor.shape[0],-1).float()
        y_cor = torch.arange(0,grid_reorder.shape[2], device=grid_reorder.device).unsqueeze(-1).expand((grid_reorder.shape[2],grid_reorder.shape[3])).reshape(-1)
        y_cor = y_cor.unsqueeze(0).expand(u_cor.shape[0],-1).float()
        grid_inv[0][torch.arange(grid_reorder.shape[1]).unsqueeze(-1),v_cor,u_cor] = torch.autograd.Variable(x_cor)
        grid_inv[1][torch.arange(grid_reorder.shape[1]).unsqueeze(-1),v_cor,u_cor] = torch.autograd.Variable(y_cor)
        grid_inv[0] = grid_inv[0]/grid_reorder.shape[3]*2-1
        grid_inv[1] = grid_inv[1]/grid_reorder.shape[2]*2-1
        grid_inv = grid_inv.permute(1,2,3,0)
        return grid, grid_inv

    def ignore_label(self, label, ignore_indexs):
        label = np.array(label)
        temp = label.copy()
        for k in ignore_indexs:
            label[temp == k] = 0
        return label
    
    def forward(self, img_data, img_original, focus_point, s_bin_selected_BxHMxWMx1, segSize=None,):
        self.input_size = (20, 20)

        #img_original = torch.nn.functional.interpolate(img_original, size=(40, 40), mode='bilinear', align_corners=False).detach()
        
        x = img_data
        _, _, H_HS, W_HS = x.shape

        x_Bx2 = focus_point.squeeze(1).squeeze(1)

        B, _ = x_Bx2.shape
        #HS, WS = self.input_size
        HS, WS = H_HS, W_HS
        max_dist = np.sqrt(HS ** 2 + WS ** 2)

        widx_B = (x_Bx2[:, 0] * (WS - 1)) / 640
        hidx_B = (x_Bx2[:, 1] * (HS - 1)) / 640

        grid_mtx_Bx2xHxW = gen_grid_mtx_2xHxW(HS, WS, device='cuda').unsqueeze(0).repeat(B, 1, 1, 1)
        dist_BxHxW = torch.sqrt((grid_mtx_Bx2xHxW[:, 0, :, :] - hidx_B[:, None, None]) ** 2 + (grid_mtx_Bx2xHxW[:, 1, :, :] - widx_B[:, None, None]) ** 2)
        focusmap_Bx1xHxW = (dist_BxHxW / max_dist).unsqueeze(1)**2

        fp_tensor = torch.zeros((B, 1, HS, WS)).to(x)
        for b in range(B):
            fp_tensor[b, 0, hidx_B[b].round().int(), widx_B[b].round().int()] = 1

        #x_low = b_imresize(x, self.input_size, interp='bilinear')

        #x_low = torch.cat((x_low, focusmap_Bx1xHxW), dim=1)
        #x_low = torch.cat((x_low, fp_tensor), dim=1)
        x_new = torch.cat((x, focusmap_Bx1xHxW), dim=1)
        x_new = torch.cat((x_new, fp_tensor), dim=1)
        
        #img_original = torch.cat((img_original, focusmap_Bx1xHxW), dim=1)
        #img_original = torch.cat((img_original, fp_tensor), dim=1)

        # 归一化注视点坐标到0-1范围（图像左上角为原点）
        normalized_gaze_coords = torch.stack([
            x_Bx2[:, 1] / 640,  # y坐标归一化
            x_Bx2[:, 0] / 640   # x坐标归一化
        ], dim=1)  # 形状为[B, 2]
        
                                             
        #pred = self.gaussian_predictor(x_low, normalized_gaze_coords)
        pred = self.gaussian_predictor(x_new, normalized_gaze_coords)
        xs = generate_saliency_map(pred.detach(), H_HS, W_HS, normalized_gaze_coords)

        self.gaussian_pred_Bx7 = pred  # 更改为Bx7保存所有参数
        
        #if self.training:
        #focusmap_resized = F.interpolate(focusmap_Bx1xHxW, size=(H_HS, W_HS), mode='bilinear', align_corners=False)
        #focus_weight = 1.0 / (focusmap_resized.squeeze(1) + 1e-6)
        #focus_weight = focus_weight / focus_weight.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        #gaussian_focus_loss = F.mse_loss(xs, focus_weight)
        #pdb.set_trace()
        #self.gaussian_focus_loss = gaussian_focus_loss
        #self.temp_idx += 1
        #if random.randint(0, 100) == 1:
        #    #plt.imsave('../l_sam_experiment/{}_img.png'.format(self.temp_idx), img_data.detach().clone()[0,-1,:,:].cpu().numpy(), cmap='gray')
        #    plt.imsave('../l_sam_experiment/{}_img.png'.format(self.temp_idx), img_original.detach().clone()[0,0,:,:].cpu().numpy(), cmap='gray')
        #    plt.imsave('../l_sam_experiment/{}_selected_mask.png'.format(self.temp_idx), xs.detach().clone()[0,:,:].cpu().numpy(), cmap='gray')
        #    plt.imsave('../l_sam_experiment/{}_selected_mask_GT.png'.format(self.temp_idx), focus_weight.detach().clone()[0,:,:].cpu().numpy(), cmap='gray')

        #x_low_localized = self.wang_localization(x_low)

        #xs = self.net_compress(xs)
        #xs = self.net_compress(xs) + (10/(focusmap_Bx1xHxW+1)**5)
        #xs = self.wang_net_compress(x_low_localized) + (1/(focusmap_Bx1xHxW+1e-6))
        #xs = s_bin_selected_BxHMxWMx1

        #xs = xs.view(-1,self.grid_size_x*self.grid_size_y) # N,1,W*H
        #xs = nn.Softmax()(xs) # N,W*H
        #assert not torch.isnan(xs).any(), "xs contains NaN values!"
        #xs = xs.view(-1,1,self.grid_size_x,self.grid_size_y)

        #xs = b_imresize(xs, (H_HS, W_HS), interp='bilinear')

        HS, WS = self.input_size
        return differentiable_topk(x, xs, HS, WS, temperature=0.1)
    
    #def forward(self, img_data, focus_point, s_bin_selected_BxHMxWMx1, segSize=None,):
    #    upsample = False
    #    self.input_size = (20, 20)
    #    segSize = (20, 20)
    #    self.cfg.TRAIN.def_saliency_pad_mode = 'replication'

    #    x = img_data
    #    _, _, H_HS, W_HS = x.shape
    #    t = time.time()
    #    ori_size = (x.shape[-2],x.shape[-1])

    #    ######################## enter here!!!!!!!!!!!!!!!!!!##########
    #    x_Bx2 = focus_point.squeeze(1).squeeze(1)

    #    B, _ = x_Bx2.shape
    #    HS, WS = self.input_size
    #    max_dist = np.sqrt(HS ** 2 + WS ** 2)

    #    widx_B = (x_Bx2[:, 0] * (WS - 1)) / 640
    #    hidx_B = (x_Bx2[:, 1] * (HS - 1)) / 640

    #    grid_mtx_Bx2xHxW = gen_grid_mtx_2xHxW(HS, WS, device='cuda').unsqueeze(0).repeat(B, 1, 1, 1)
    #    dist_BxHxW = torch.sqrt((grid_mtx_Bx2xHxW[:, 0, :, :] - hidx_B[:, None, None]) ** 2 + (grid_mtx_Bx2xHxW[:, 1, :, :] - widx_B[:, None, None]) ** 2)
    #    focusmap_Bx1xHxW = (dist_BxHxW / max_dist).unsqueeze(1)**2

    #    fp_tensor = torch.zeros((B, 1, HS, WS)).to(x)
    #    for b in range(B):
    #        fp_tensor[b, 0, hidx_B[b].round().int(), widx_B[b].round().int()] = 1

    #    ###############################################################
    #    # EXPLAIN: compute its lower resolution version Xlr
    #    x_low = b_imresize(x, self.input_size, interp='bilinear')
    #    # x_low = x

    #    #############enter here!!!!##################################################
    #    x_low = torch.cat((x_low, focusmap_Bx1xHxW), dim=1)
    #    x_low = torch.cat((x_low, fp_tensor), dim=1)

    #    xs = self.localization(x_low)

    #    # wang ----------------
    #    #xs = self.net_compress(xs) + (10*s_bin_selected_BxHMxWMx1)
    #    xs = self.net_compress(xs) + (10/(focusmap_Bx1xHxW+1)**16)
    #    # wang ----------------

    #    # xs = s_bin_selected_BxHMxWMx1
    #    # xs = nn.Upsample(size=(self.grid_size_x,self.grid_size_y), mode='bilinear')(xs)

    #    xs = xs.view(-1,self.grid_size_x*self.grid_size_y) # N,1,W*H

    #    xs = nn.Softmax()(xs) # N,W*H
    #    assert not torch.isnan(xs).any(), "xs contains NaN values!"
    #    xs = xs.view(-1,1,self.grid_size_x,self.grid_size_y) #saliency map
 

    #    # EXPLAIN: pad to avoid boundary artifact following A. Recasens et,al. (2018)
    #    # if self.cfg.MODEL.uniform_sample != '':
    #    #     #print('aaaaaaaaaaaaaaa\n\n\n\n')
    #    #     xs = xs*0 + 1.0/(self.grid_size_x*self.grid_size_y)
    #    if self.cfg.TRAIN.def_saliency_pad_mode == 'replication':
    #        #print('bbbbbbbbbbbbbbbbbbbb\n\n\n\n')
    #        # enter here!!!!!!!!!!!!!!!!!
    #        xs_hm = nn.ReplicationPad2d((self.padding_size_y, self.padding_size_y, self.padding_size_x, self.padding_size_x))(xs) # padding by replicate the edges in the map
    #        #print(self.padding_size_y, self.padding_size_y, self.padding_size_x, self.padding_size_x)   # 30 30 15 15 
    #        #print(xs.shape, xs_hm.shape)   # torch.Size([1, 1, 64, 128]) torch.Size([1, 1, 94, 188])
    #        #print(xs)
    #        #print(xs_hm)
    #    elif self.cfg.TRAIN.def_saliency_pad_mode == 'reflect':
    #        #print('cccccccccccccccccc\n\n\n\n')
    #        xs_hm = F.pad(xs, (self.padding_size_y, self.padding_size_y, self.padding_size_x, self.padding_size_x), mode='reflect')
    #    elif self.cfg.TRAIN.def_saliency_pad_mode == 'zero':
    #        #print('dddddddddddddddddd\n\n\n\n')
    #        xs_hm = F.pad(xs, (self.padding_size_y, self.padding_size_y, self.padding_size_x, self.padding_size_x), mode='constant')

    #    # EXPLAIN: at inference, calculate both the non-uniform downsampler (grid) and upsampler (grid_inv)
    #    xs_inv = 1-xs_hm
    #    grid, _ = self.create_grid(xs_hm, segSize=segSize, x_inv=xs_inv)

    #    # EXPLAIN: computes the downsampled image X^ =Gd(X,d)
    #    # if self.cfg.MODEL.uniform_sample == 'BI':
    #    #     x_sampled = nn.Upsample(size=self.input_size_net_infer, mode='bilinear')(x)
    #    # else:

    #    # wang ----------------
    #    x_sampled = F.grid_sample(x, grid, mode='nearest')
    #    # wang ----------------
    #    
    #    # wang ----------------
    #    # 获取第一个batch的grid
    #    #grid_sample = grid[0].detach().cpu().numpy()  # 第一个batch的grid [20,20,2]
    #    #
    #    ## 创建画布
    #    #plt.figure(figsize=(10, 10))
    #    #
    #    ## 绘制40x40的特征图边界 (用归一化坐标表示，范围[-1,1])
    #    #plt.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], 'b-', linewidth=2, label='40x40 Feature Map')
    #    #
    #    ## 绘制采样点
    #    #for i in range(grid_sample.shape[0]):
    #    #    for j in range(grid_sample.shape[1]):
    #    #        x, y = grid_sample[i, j]  # 获取归一化后的(x,y)坐标
    #    #        plt.plot(x, y, 'r.', markersize=3)
    #    #
    #    ## 设置图的范围和标题
    #    #plt.xlim([-1.1, 1.1])
    #    #plt.ylim([-1.1, 1.1])
    #    #plt.title('Grid Sampling Visualization (20x20 points on 40x40 feature map)')
    #    #plt.xlabel('X-axis (normalized)')
    #    #plt.ylabel('Y-axis (normalized)')
    #    #plt.grid(True, linestyle='--', alpha=0.7)
    #    #plt.legend()
    #    #
    #    ## 保存图像
    #    #plt.savefig("../l_sam_experiment/grid.png", dpi=300, bbox_inches='tight')
    #    #plt.close()
    #    
    #    # wang ----------------
    #    
    #    # print (x_sampled.shape)
    #    # x_sampled = nn.Upsample(size=self.input_size_net_infer,mode='bilinear')(x_sampled)
    #    
    #    return x_sampled, grid
   

class SegmentationModule(SegmentationModuleBase):
    def __init__(self, net_enc, net_dec, crit, cfg, deep_sup_scale=None, net_fov_res=None):
        super(SegmentationModule, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.crit = crit
        self.cfg = cfg
        self.deep_sup_scale = deep_sup_scale
        self.net_fov_res = net_fov_res

    # @torchsnooper.snoop()
    def forward(self, feed_dict, *, segSize=None, F_Xlr_acc_map=False, writer=None, count=None, feed_dict_info=None, feed_batch_count=None):
        # training
        if segSize is None:
            if self.deep_sup_scale is not None: # use deep supervision technique
                (pred, pred_deepsup) = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True))
            elif self.net_fov_res is not None:
                pred = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True), res=self.net_fov_res(feed_dict['img_data']))
            else:
                pred = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True))

            loss = self.crit(pred, feed_dict['seg_label'])
            if self.deep_sup_scale is not None:
                loss_deepsup = self.crit(pred_deepsup, feed_dict['seg_label'])
                loss = loss + loss_deepsup * self.deep_sup_scale

            acc = self.pixel_acc(pred, feed_dict['seg_label'])
            return loss, acc
        # inference
        else:
            if self.net_fov_res is not None:
                pred = self.decoder(self.encoder(feed_dict['img_data'].contiguous(), return_feature_maps=True), segSize=segSize, res=self.net_fov_res(feed_dict['img_data']))
            else:
                pred = self.decoder(self.encoder(feed_dict['img_data'].contiguous(), return_feature_maps=True), segSize=segSize)
            if self.cfg.VAL.write_pred:
                _, pred_print = torch.max(pred, dim=1)
                colors = loadmat('data/color150.mat')['colors']
                pred_color = colorEncode(as_numpy(pred_print.squeeze(0)), colors)
                pred_print = torch.from_numpy(pred_color.astype(np.uint8)).unsqueeze(0).permute(0,3,1,2)
                print('train/pred size: {}'.format(pred_print.shape))
                pred_print = vutils.make_grid(pred_print, normalize=False, scale_each=True)
                writer.add_image('train/pred', pred_print, count)

            if F_Xlr_acc_map:
                loss = self.crit(pred, feed_dict['seg_label'])
                return pred, loss
            else:
                return pred

class ModelBuilder:
    # custom weights initialization
    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
    @staticmethod
    def build_encoder(arch='resnet50', fc_dim=2048, weights='', dilate_rate=4):
        pretrained = True if len(weights) == 0 else False
        arch = arch.lower()
        #print(arch)  # hrnetv2_nodownsp
        #exit()
        if arch == 'mobilenetv2dilated':
            orig_mobilenet = mobilenet.__dict__['mobilenetv2'](pretrained=pretrained)
            net_encoder = MobileNetV2Dilated(orig_mobilenet, dilate_scale=dilate_rate)
        elif arch == 'resnet18':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet18dilated':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=dilate_rate)
        elif arch == 'resnet34':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet34dilated':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=dilate_rate)
        elif arch == 'resnet50':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet50dilated':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=dilate_rate)
        elif arch == 'resnet101':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet101dilated':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=dilate_rate)
        elif arch == 'resnext101':
            orig_resnext = resnext.__dict__['resnext101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnext) # we can still use class Resnet
        elif arch == 'hrnetv2_nodownsp':
            print('use hrnet!!!!!!!!!!!!!')
            net_encoder = hrnetv2_nodownsp.__dict__['hrnetv2_nodownsp'](pretrained=False)
        # elif arch == 'segformer':
        #     print('use segformer!!!!!!!!!!!!!')
        #     net_encoder = segformer.__dict__['segformer'](pretrained=False)
        else:
            raise Exception('Architecture undefined!')

        # encoders are usually pretrained
        # net_encoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print(f'here!!!!!!!!!!!!!!!!!!  Loading weights for net_encoder {weights}')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_encoder

    @staticmethod
    def build_decoder(arch='upernet',
                      fc_dim=2048, num_class=150,
                      weights='', use_softmax=False):
        arch = arch.lower()
        if arch == 'c1_deepsup':
            net_decoder = C1DeepSup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'c1':
            net_decoder = C1(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'ppm':
            net_decoder = PPM(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'ppm_deepsup':
            net_decoder = PPMDeepsup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'upernet_lite':
            net_decoder = UPerNet(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=256)
        elif arch == 'upernet':
            net_decoder = UPerNet(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=512)
        else:
            raise Exception('Architecture undefined!')

        net_decoder.apply(ModelBuilder.weights_init)
        # net_decoder.load_state_dict(
        #     torch.load('/root/autodl-tmp/lvis_Tin_80_80_ours_1_8_9_58pm_mse_dicefocal_kernel45_12w_largerhr/decoder_epoch_120.pth', map_location=lambda storage, loc: storage), strict=False)
        if len(weights) > 0:
            print('Loading weights for net_decoder')
            net_decoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_decoder

    # @staticmethod
def build_net_saliency():
    # define saliency network
    # Spatial transformer localization-network
    # if cfg.MODEL.track_running_stats:
    #     if cfg.MODEL.saliency_net == 'resnet18':
    #         net_saliency = saliency_network_resnet18()
    #     elif cfg.MODEL.saliency_net == 'resnet18_stride1':
    #         net_saliency = saliency_network_resnet18_stride1()
    #     elif cfg.MODEL.saliency_net == 'fovsimple':
    #         net_saliency = fov_simple(cfg)
    net_saliency = saliency_network_resnet18_stride1()
    # if len(weights) == 0:
    #     net_saliency.apply(ModelBuilder.weights_init)
    #     # net_saliency.load_state_dict(
    #     #     torch.load('/root/autodl-tmp/lvis_Tin_80_80_ours_1_8_9_58pm_mse_dicefocal_kernel45_12w_largerhr/saliency_epoch_120.pth', map_location=lambda storage, loc: storage), strict=False)
    # else:
    #     print('Loading weights for net_saliency')
    #     net_saliency.load_state_dict(
    #         torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
    return net_saliency

# @staticmethod
def build_net_compress():
    net_compress = CompressNet()

    return net_compress
