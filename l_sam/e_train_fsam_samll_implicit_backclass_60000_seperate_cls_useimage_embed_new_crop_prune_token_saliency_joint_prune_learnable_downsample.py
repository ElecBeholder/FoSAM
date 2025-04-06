import os
import sys
import pdb

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from datetime import datetime

from torch import nn

from d_model.nn_A1_tools import merge_seg_cls, merge_seg_label
from d_model.nn_A1_tools import merge_seg_cls_new, merge_seg_label_new
from l_sam.forveated_sam.foveated_sam_implicit_backclass_cls_useimage_embed_crop_prune_token_advance_res101_saliency_joint_prune_learnable_downsample import build_foveated_sam
from utility.fctn import save_json

from d_model.nn_A2_loss import BMSELoss, dice_loss
import preset
from d_model.nn_A0_utils import freeze_parameters, unfreeze_parameters, init_weights_random
import os
import platform
import torch.nn.functional as F
import matplotlib
import pandas as pd
from d_model.nn_A3_metrics import evaluate_segmentation, evaluate_classification_overall, compute_multiclass_iou
from torchmetrics.classification import MulticlassJaccardIndex
from pytorch_toolbelt.losses.dice import DiceLoss
from l_sam.forveated_sam.efficient_sam_encoder_saliency import average_pool, get_merge_map_edge, get_merge_map_object
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

system = platform.system()
if system == "Windows":
    matplotlib.use('TkAgg') 

os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='/home/external/DynamicFocus_new_ziqi/l_sam_experiment')

from tqdm import trange

from e_preprocess_scripts.a_preprocess_tools import CustomDataLoader
from e_preprocess_scripts.b2_preprocess_lvis import DatasetLVIS, PreprocessLVIS
from e_preprocess_scripts.b3_preprocess_cityscapes import wrap_name, DatasetCityScapes, PreprocessCityscapes

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def pooling_function(x, downsample_factor=1, pool_type='avg'):
    """
    Applies average pooling or max pooling to the input tensor based on downsample_factor.

    Args:
        x (torch.Tensor): The input tensor of shape (B, C, H, W).
        downsample_factor (int): The downsampling factor. If 1, returns the input tensor directly.
        pool_type (str): Pooling type, either 'average' or 'max'. Default is 'average'.

    Returns:
        torch.Tensor: The pooled tensor or the input tensor if downsample_factor is 1.
    """
    if downsample_factor == 1:
        return x  # Return the input directly

    # Choose pooling operation
    if pool_type == 'avg':
        pooled = F.avg_pool2d(x, kernel_size=downsample_factor, stride=downsample_factor)
    elif pool_type == 'max':
        pooled = F.max_pool2d(x, kernel_size=downsample_factor, stride=downsample_factor)
    else:
        raise ValueError(f"Unsupported pool_type '{pool_type}'. Use 'average' or 'max'.")

    return pooled


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)

        bce_loss = - (target * torch.log(pred) + (1 - target) * torch.log(1 - pred))

        modulating_factor = (target * (1 - pred) + (1 - target) * pred) ** self.gamma

        alpha_factor = self.alpha * target + (1 - self.alpha) * (1 - target)

        focal_loss = alpha_factor * modulating_factor * bce_loss
        return focal_loss.mean()


class BatchDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        """
        Dice Loss computed per sample in the batch using tensor operations.
        Args:
            smooth: A small constant to avoid division by zero.
        """
        super(BatchDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred: Tensor, predicted probability map, shape [B, 1, H, W].
            target: Tensor, ground truth binary mask (0 or 1), shape [B, 1, H, W].

        Returns:
            mean_dice_loss: Scalar, mean Dice Loss over the batch.
        """

        pred = pred.view(pred.size(0), -1)  
        target = target.view(target.size(0), -1)  

        intersection = (pred * target).sum(dim=1)  
        union = pred.sum(dim=1) + target.sum(dim=1)
        dice_coeff = (2 * intersection + self.smooth) / (union + self.smooth)

        dice_loss = 1.0 - dice_coeff

        mean_dice_loss = dice_loss.mean()
        return mean_dice_loss


class GaussianMaskFittingLoss(nn.Module):
    """
    计算预测的高斯分布与mask分布之间的拟合损失。
    将mask中的1看作样本点，计算这些点的分布特征，
    然后使用损失函数衡量预测的高斯分布与这些点的分布之间的差异。
    """
    def __init__(self, epsilon=1e-6):
        super(GaussianMaskFittingLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, gaussian_params, fp_x_coords, fp_y_coords, mask):
        """
        Args:
            gaussian_params: 预测的高斯分布参数 [B, 5]
            mask: 分割掩码 [B, 1, H, W]
        Returns:
            loss: 高斯分布与mask分布之间的拟合损失
        """
        B, _, H, W = mask.shape
        loss = 0.0
        loss_pos = 0.0
        loss_var = 0.0
        loss_pho = 0.0
        
        for b in range(B):
            # 获取当前批次的mask和高斯参数
            cur_mask = mask[b, 0]  # [H, W]
            cur_params = gaussian_params[b]  # [5]
            
            # 提取预测的高斯分布参数
            mu_x = (torch.tanh(cur_params[0])/2 + fp_x_coords[b])  # 归一化到[0,1]
            mu_y = (torch.tanh(cur_params[1])/2 + fp_y_coords[b])  # 归一化到[0,1]
            sigma_x = F.softplus(cur_params[2]) + self.epsilon  # 确保方差为正
            sigma_y = F.softplus(cur_params[3]) + self.epsilon  # 确保方差为正
            rho = torch.tanh(cur_params[4])  # 归一化到[-1,1]
            
            # 找出mask中值为1的点的坐标
            mask_indices = torch.nonzero(cur_mask > 0.5)  # [N, 2]，其中N是mask中1的数量
            
            if len(mask_indices) == 0:
                # 如果mask中没有前景像素，则跳过当前样本
                continue
                
            # 将坐标归一化到[0,1]范围
            y_coords = mask_indices[:, 0].float() / (H - 1)  # [N]
            x_coords = mask_indices[:, 1].float() / (W - 1)  # [N]
            
            # 计算mask点的分布特征（均值和方差）
            mask_mean_x = x_coords.mean()
            mask_mean_y = y_coords.mean()
            mask_var_x = x_coords.var() + self.epsilon
            mask_var_y = y_coords.var() + self.epsilon
            
            # 计算协方差
            mask_cov = ((x_coords - mask_mean_x) * (y_coords - mask_mean_y)).mean()
            mask_corr = mask_cov / (torch.sqrt(mask_var_x * mask_var_y) + self.epsilon)
            
            # 计算预测高斯分布与mask分布之间的KL散度
            # KL(P||Q) for multivariate Gaussian
            det_ratio = (sigma_x * sigma_y * (1 - rho**2)) / (mask_var_x * mask_var_y * (1 - mask_corr**2) + self.epsilon)
            trace_term = (mask_var_x / (sigma_x**2 + self.epsilon)) + (mask_var_y / (sigma_y**2 + self.epsilon))
            
            mean_diff_x = mask_mean_x - mu_x
            mean_diff_y = mask_mean_y - mu_y
            
            # 简化的马氏距离计算
            mahalanobis_dist = (mean_diff_x**2 / (sigma_x**2 + self.epsilon)) + (mean_diff_y**2 / (sigma_y**2 + self.epsilon))
            
            # 简化的KL散度计算
            kl_div = 0.5 * (torch.log(det_ratio + self.epsilon) - 2 + trace_term + mahalanobis_dist)
            
            # 添加L2损失以直接匹配参数
            #l2_loss = ((mu_x - mask_mean_x)**2 + (mu_y - mask_mean_y)**2 + 
            #           (sigma_x - torch.sqrt(mask_var_x))**2 + (sigma_y - torch.sqrt(mask_var_y))**2 + 
            #           (rho - mask_corr)**2)
            
            # 综合损失
            l2_loss_pos = (mu_x - mask_mean_x)**2 + (mu_y - mask_mean_y)**2 
            l2_loss_var = (sigma_x - torch.sqrt(mask_var_x))**2 + (sigma_y - torch.sqrt(mask_var_y))**2
            l2_loss_rho = (rho - mask_corr)**2

            loss_pos += l2_loss_pos.detach().cpu().item()
            loss_var += l2_loss_var.detach().cpu().item()
            loss_pho += l2_loss_rho.detach().cpu().item()

            sample_loss = l2_loss_rho + l2_loss_var + 10 * l2_loss_pos #kl_div + l2_loss
            loss += sample_loss

        #print("\n pos:{}, var:{}, rho:{}, all:{}\n".format(10*loss_pos/B, loss_var/B, loss_pho/B, loss.detach().cpu().item()/B))
            
        return loss / B if B > 0 else torch.tensor(0.0, device=mask.device)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()


BFLoss = BinaryFocalLoss()
BMSE = BMSELoss()

celoss = nn.CrossEntropyLoss()

diceloss = DiceLoss(mode='multiclass', from_logits=True)
diceloss_1 = BatchDiceLoss()
gaussian_mask_loss = GaussianMaskFittingLoss().to(device)  # 初始化高斯分布拟合损失
sigmoid = nn.Sigmoid()
focaloss = FocalLoss()

if __name__ == '__main__':

    old_state_dict = torch.load("/home/external/DynamicFocus_new_ziqi/l_sam/efficient_sam_vits.singlemask.pt")['model']
    # old_state_dict = torch.load("/root/autodl-tmp/DynamicFocus/l_sam/efficient_sam_vits.pt")

    # # copy mask 1 to 0
    # keys_copy = ['mask_decoder.output_hypernetworks_mlps.1.layers.0.0.weight',
    #              'mask_decoder.output_hypernetworks_mlps.1.layers.0.0.bias',
    #              'mask_decoder.output_hypernetworks_mlps.1.layers.1.0.weight',
    #              'mask_decoder.output_hypernetworks_mlps.1.layers.1.0.bias',
    #              'mask_decoder.output_hypernetworks_mlps.1.fc.weight',
    #              'mask_decoder.output_hypernetworks_mlps.1.fc.bias']
    
    # for key in keys_copy:
    #     old_state_dict['model'][key.replace('_mlps.1.', '_mlps.0.')] = old_state_dict['model'][key]
    
    # old_state_dict['model']['mask_decoder.mask_tokens.weight'][0] = old_state_dict['model']['mask_decoder.mask_tokens.weight'][1]
    # old_state_dict['model']['mask_decoder.iou_prediction_head.fc.weight'][0, :] = old_state_dict['model']['mask_decoder.iou_prediction_head.fc.weight'][1, :]
    # old_state_dict['model']['mask_decoder.iou_prediction_head.fc.bias'][0] = old_state_dict['model']['mask_decoder.iou_prediction_head.fc.bias'][1]
    
    # # delete key
    
    # keys_del = ['mask_decoder.output_hypernetworks_mlps.1.layers.0.0.weight',
    #             'mask_decoder.output_hypernetworks_mlps.1.layers.0.0.bias',
    #             'mask_decoder.output_hypernetworks_mlps.1.layers.1.0.weight',
    #             'mask_decoder.output_hypernetworks_mlps.1.layers.1.0.bias',
    #             'mask_decoder.output_hypernetworks_mlps.1.fc.weight',
    #             'mask_decoder.output_hypernetworks_mlps.1.fc.bias',
    #             'mask_decoder.output_hypernetworks_mlps.2.layers.0.0.weight',
    #             'mask_decoder.output_hypernetworks_mlps.2.layers.0.0.bias',
    #             'mask_decoder.output_hypernetworks_mlps.2.layers.1.0.weight',
    #             'mask_decoder.output_hypernetworks_mlps.2.layers.1.0.bias',
    #             'mask_decoder.output_hypernetworks_mlps.2.fc.weight',
    #             'mask_decoder.output_hypernetworks_mlps.2.fc.bias',
    #             'mask_decoder.output_hypernetworks_mlps.3.layers.0.0.weight',
    #             'mask_decoder.output_hypernetworks_mlps.3.layers.0.0.bias',
    #             'mask_decoder.output_hypernetworks_mlps.3.layers.1.0.weight',
    #             'mask_decoder.output_hypernetworks_mlps.3.layers.1.0.bias',
    #             'mask_decoder.output_hypernetworks_mlps.3.fc.weight',
    #             'mask_decoder.output_hypernetworks_mlps.3.fc.bias']
    
    # old_state_dict['model']['mask_decoder.mask_tokens.weight'] = old_state_dict['model']['mask_decoder.mask_tokens.weight'][:1]
    # old_state_dict['model']['mask_decoder.iou_prediction_head.fc.weight'] = old_state_dict['model']['mask_decoder.iou_prediction_head.fc.weight'][:1, :]
    # old_state_dict['model']['mask_decoder.iou_prediction_head.fc.bias'] = old_state_dict['model']['mask_decoder.iou_prediction_head.fc.bias'][:1]
    
    # for key,item in old_state_dict['model'].items():
    #     print(key,item.shape)
    
    # for key_del in keys_del:
    #     del old_state_dict['model'][key_del]
    # torch.save(old_state_dict, os.path.join(preset.dpath_esam_weights, 'efficient_sam_vits.singlemask_train_lvis.pt'))


    if True:
        mode = 'lvis'

        avg_pool = nn.Identity()
        max_pool = nn.Identity()
        HW_RAW_SIZE = 0
        HW_image_input_size = 0
        class_num = 1

        if mode == 'lvis':
            downsample_factor = 1
            HW_RAW_SIZE = 640
            # class_num = 58
            class_num = 51
            HW_image_input_size = 640
            avg_pool = nn.AvgPool2d(kernel_size=downsample_factor, stride=downsample_factor) if downsample_factor > 1 else nn.Identity()
            max_pool = nn.MaxPool2d(kernel_size=downsample_factor, stride=downsample_factor) if downsample_factor > 1 else nn.Identity()

            # if preset.pc_name != 'XPS':
            # dataset_train = DatasetLVIS('sp60000', dataset_partition='train')
            # dataloader_train = CustomDataLoader(dataset_train, xrange=trange, cache=False)
            # dataset_valid = DatasetLVIS('sp12000', dataset_partition='valid')
            # dataloader_valid = CustomDataLoader(dataset_valid, xrange=trange, cache=False)
            #dataset_train = DatasetLVIS('sp20000', dataset_partition='train')
            #dataloader_train = CustomDataLoader(dataset_train, xrange=trange, cache=False)
            #dataset_valid = DatasetLVIS('sp4000', dataset_partition='valid')
            #dataloader_valid = CustomDataLoader(dataset_valid, xrange=trange, cache=False)
            dataset_train = DatasetLVIS('sp640_60000', dataset_partition='train')
            dataloader_train = CustomDataLoader(dataset_train, xrange=trange, cache=False)
            dataset_valid = DatasetLVIS('sp640_12000', dataset_partition='valid')
            dataloader_valid = CustomDataLoader(dataset_valid, xrange=trange, cache=False)
            # else:
            #     dataset_train = DatasetLVIS('sp640_1000', dataset_partition='train')
            #     dataloader_train = CustomDataLoader(dataset_train, xrange=trange, cache=False)
            #     dataset_valid = DatasetLVIS('sp640_200', dataset_partition='valid')
            #     dataloader_valid = CustomDataLoader(dataset_valid, xrange=trange, cache=False)
            pplv_train = PreprocessLVIS(dataset_partition='train')
            # cids_monitored = pplv_train.get_cids_monitored(take_num_class=57)
            cids_monitored = pplv_train.get_cids_monitored(take_num_class=50)

            get_name = lambda k: wrap_name(pplv_train.id2catyinfo[cids_monitored[k]]['name'])

        elif mode == 'cityscapes':

            downsample_factor = 1
            HW_RAW_SIZE = 1024
            class_num = 42
            HW_image_input_size = HW_RAW_SIZE // downsample_factor

            avg_pool = nn.AvgPool2d(kernel_size=downsample_factor, stride=downsample_factor) if downsample_factor > 1 else nn.Identity()
            max_pool = nn.MaxPool2d(kernel_size=downsample_factor, stride=downsample_factor) if downsample_factor > 1 else nn.Identity()

            # if preset.pc_name != 'XPS':
            dataset_train = DatasetCityScapes('sp1024_20000', dataset_partition='train', cropH=HW_RAW_SIZE, cropW=HW_RAW_SIZE)
            dataloader_train = CustomDataLoader(dataset_train, xrange=trange, cache=False)

            dataset_valid = DatasetCityScapes('sp1024_4000', dataset_partition='valid', cropH=HW_RAW_SIZE, cropW=HW_RAW_SIZE)
            dataloader_valid = CustomDataLoader(dataset_valid, xrange=trange, cache=False)
            # else:
            #     dataset_train = DatasetCityScapes('sp1024_100', dataset_partition='train', cropH=HW_RAW_SIZE, cropW=HW_RAW_SIZE)
            #     dataloader_train = CustomDataLoader(dataset_train, xrange=trange, cache=False)

            #     dataset_valid = DatasetCityScapes('sp1024_20', dataset_partition='valid', cropH=HW_RAW_SIZE, cropW=HW_RAW_SIZE)
            #     dataloader_valid = CustomDataLoader(dataset_valid, xrange=trange, cache=False)

            ppcs_train = PreprocessCityscapes(dataset_partition='train')

            get_name = lambda k: wrap_name(ppcs_train.idx2label[k])


        # Build the EfficientSAM-Ti model.
        efficientsam_ti = build_foveated_sam(img_size=HW_image_input_size,
                                             encoder_patch_embed_dim=384,
                                             encoder_num_heads=6,
                                             num_multimask_outputs=1,
                                             class_num=class_num,
                                             checkpoint_state_dict=old_state_dict,
                                             ).to(device)

        efficientsam_ti_custom = efficientsam_ti
        efficientsam_ti_custom.to(device)

        freeze_parameters(efficientsam_ti_custom)
        # unfreeze_parameters(efficientsam_ti_custom.mask_decoder)
        # unfreeze_parameters(efficientsam_ti_custom.classify_model)

        unfreeze_parameters(efficientsam_ti_custom.mask_decoder)
        unfreeze_parameters(efficientsam_ti_custom.classifier)
        unfreeze_parameters(efficientsam_ti_custom.image_encoder.segmentation_module)
                        


        lr1 = 5e-4
        # lr2 = 3e-4
        # lr2 = 5e-4 # setting 1 slow 70 epoch 0.027
        lr2 = 1e-2
        lr3 = 1e-2
        num_epochs = 200

        for name, param in efficientsam_ti_custom.image_encoder.segmentation_module.named_parameters():
            param.requires_grad_(True)

        optimizer = torch.optim.Adam(
            [
                {"params": efficientsam_ti_custom.mask_decoder.parameters(), "lr": lr1},
                {"params": efficientsam_ti_custom.classifier.parameters(), "lr": lr2},
                {"params": efficientsam_ti_custom.image_encoder.segmentation_module.parameters(), "lr": lr3},
            ]
        )
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=2, mode='max', min_lr=1e-6)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, mode='max', min_lr=1e-6)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,30,45,60,75,90,105,130,145,160,175], gamma=0.5)

        for name, param in efficientsam_ti_custom.named_parameters():
            print(f"{name}: {param.requires_grad}")
        dpath = os.path.join(preset.dpath_training_records, f'{efficientsam_ti_custom.__class__.__name__}_{mode}_{HW_RAW_SIZE}x{HW_RAW_SIZE}_{HW_image_input_size}x{HW_image_input_size}_{datetime.now().strftime("%Y%m%d_%H%M")}')

        os.makedirs(dpath, exist_ok=True)
        # multi_class_miou_evaluator = MulticlassJaccardIndex(num_classes=class_num + 1).to(device=device)
        global_step = 0
        for epoch in trange(num_epochs):
            batch_size = 8

            optimizer.zero_grad()
            global_step = global_step + 1
            loss_mean = []

            if epoch > 0:
                loss_diceloss_s = []
                loss_cme_s = []
                loss_s = []
                efficientsam_ti_custom.train()
                cls_predictions_bx1xK_s = []
                Y_cls_bx1_s = []

                for bidx, bparts in enumerate(dataloader_train.get_iterator(batch_size=batch_size, device=torch.device("cpu"), shuffle=True, xrange=trange)):
                    for name, param in efficientsam_ti_custom.image_encoder.segmentation_module.wang_net_compress.named_parameters():
                        param.requires_grad_(True)
                    X_bx4xHxW, F_bx2, Y_bx1xHxW, Y_cls_bx1 = bparts
                    X_bx4xHxW = avg_pool(X_bx4xHxW)
                    Y_bx1xHxW = max_pool(Y_bx1xHxW)
                    """
                    image_1x3xHxW.shape torch.Size([1, 3, 512, 512])
                    input_points_1x1xNx2.shape torch.Size([1, 1, 1, 2])
                    input_labels_1x1xN.shape torch.Size([1, 1, 1])
                    """

                    image_bx3xHxW = X_bx4xHxW[:, :3, :, :].to(device=device)

                    input_points_bx1xNx2 = (F_bx2[:, None, None, [1, 0]] * HW_image_input_size).to(device=device, dtype=torch.int64)
                    b, _ = F_bx2.shape
                    input_labels_bx1xN = torch.ones(b, 1, 1).to(device=device, dtype=torch.int64)

                    #add 
                    # s_BxHTxWTx1 = (average_pool(Y_bx1xHxW, 16).permute(0, 2, 3, 1))
                    # s_bin_selected_BxHMxWMx1 = (get_merge_map_object(s_BxHTxWTx1).to(device=device) > 0.5)

                    output_masks_bx1x1xHxW, cls_predictions_bx1xK, iou_predictions_bx1x1 = efficientsam_ti_custom(
                        image_bx3xHxW,
                        input_points_bx1xNx2,
                        input_labels_bx1xN,
                    )

                    seg_pred_bx1xHxW = nn.functional.sigmoid(output_masks_bx1x1xHxW[:, 0, :, :, :]).float() - 0.5
                    cur_Y_bx1xHxW = Y_bx1xHxW[:, :, :, :].to(device=device)

                    Y_cls_b = Y_cls_bx1.squeeze(1).to(device=device)

                    cls_pred_bxK = cls_predictions_bx1xK.squeeze(1)

                    cls_predictions_bx1xK_s.append(cls_predictions_bx1xK.squeeze(1))
                    Y_cls_bx1_s.append(Y_cls_bx1+1)
                    # print(cls_pred_bxK)

                    pred_bx1KxHxW = merge_seg_cls_new(seg_pred_bx1xHxW, cls_pred_bxK)
                    label_bxHxW = merge_seg_label_new(cur_Y_bx1xHxW, Y_cls_b)
                    # with open('output_lvis_640_cls_implicit_backclass_3e4_12000.txt', 'a', encoding='utf-8') as f:
                    #     print(pred_bx1KxHxW.size(), file=f)
                    #     print(Y_cls_b, file=f)
                    # print(label_bxHxW)


                    # loss = diceloss(seg_pred_bx1xHxW, cur_Y_bx1xHxW) + celoss(pred_bx1KxHxW, label_bxHxW)
                    #loss = diceloss(pred_bx1KxHxW, label_bxHxW)
                    # 获取高斯分布参数和计算拟合损失
                    gaussian_params = efficientsam_ti_custom.image_encoder.segmentation_module.gaussian_pred_Bx5
                    gaussian_fitting_loss = gaussian_mask_loss(gaussian_params, F_bx2[:, 1].to(device), F_bx2[:, 0].to(device), cur_Y_bx1xHxW)
                    
                    # 结合原来的分割损失和新的高斯拟合损失
                    seg_loss = diceloss(pred_bx1KxHxW, label_bxHxW)
                    
                    # 设置高斯拟合损失的权重
                    lambda_gaussian = 1  # 可以根据需要调整权重
                    
                    # 总损失
                    loss = seg_loss + lambda_gaussian * gaussian_fitting_loss
                    loss_mean.append(loss.item())

                    # loss_s.append(loss.detach())

                    optimizer.zero_grad()
                    loss.backward()
                    # for name, param in efficientsam_ti_custom.named_parameters():
                    #     print(f"{name} grad: {param.grad is not None}")
                    optimizer.step()

                writer.add_scalar('Loss_cls_train', np.array(loss_mean).mean(), global_step)

                predictions = torch.argmax(torch.cat(cls_predictions_bx1xK_s, dim=0).squeeze(1), dim=1).detach().to('cuda') 
                targets = torch.cat(Y_cls_bx1_s, dim=0).squeeze(1).detach().T.to('cuda')

                mask = (predictions != 0).to('cuda') 

                filtered_predictions = predictions[mask]
                filtered_targets = targets[mask]

                accuracy = np.sum(filtered_predictions.cpu().numpy() == filtered_targets.cpu().numpy()) / len(filtered_predictions)

                print(f"Filtered Accuracy Train: {accuracy}")
                writer.add_scalar('Filtered Accuracy Train', accuracy, global_step)

                # print(f'\nEpoch [{epoch + 1}/{num_epochs}], Loss[Dice]: {np.array(loss_s).mean():.4f} kr={scheduler.get_last_lr()}', end='')

            if True:
                efficientsam_ti_custom.eval()
                with torch.no_grad():
                    col2elems = {
                        'label': [],
                        'seg_miou': [],
                        'seg_fg_iou': [],
                        'seg_bg_iou': [],
                        'cls_fg_iou': [],
                        'cls_miou': [],
                    }

                    seg_miou_s, seg_fg_iou_s, seg_bg_iou_s = [], [], []
                    cls_overall_miou_bsize_s = []
                    cls_predictions_bx1xK_s = []
                    Y_cls_bx1_s = []

                    cls_fs_s = []
                    ks = []
                    loss_mean = []

                    for bidx, bparts in enumerate(dataloader_valid.get_iterator(batch_size=8, device=torch.device("cpu"), shuffle=True, xrange=range)):
                        X_bx4xHxW, F_bx2, Y_bx1xHxW, Y_cls_bx1 = bparts
                        X_bx4xHxW = avg_pool(X_bx4xHxW)
                        Y_bx1xHxW = max_pool(Y_bx1xHxW)

                        image_bx3xHxW = X_bx4xHxW[:, :3, :, :].to(device=device)
                        input_points_bx1xNx2 = (F_bx2[:, None, None, [1, 0]] * HW_image_input_size).to(device=device, dtype=torch.int64)
                        b, _ = F_bx2.shape
                        input_labels_bx1xN = torch.ones(b, 1, 1).to(device=device, dtype=torch.int64)

                        # output_masks_bx1x1xHxW, cls_predictions_bx1xK, iou_predictions_bx1x1 = efficientsam_ti_custom(
                        #     image_bx3xHxW,
                        #     input_points_bx1xNx2,
                        #     input_labels_bx1xN,
                        # )

                        # wang ----------------
                        if bidx == 0:
                            for i in range(8):
                                import matplotlib.pyplot as plt
                                arr = image_bx3xHxW[i,0,:,:].cpu().numpy()
                                plt.imsave('../l_sam_experiment/{}_img.png'.format(i), arr, cmap='gray')
                                arr = Y_bx1xHxW[i,0,:,:].cpu().numpy()
                                plt.imsave('../l_sam_experiment/{}_mask.png'.format(i), arr, cmap='gray')
                                arr = torch.zeros(40, 40)
                                arr[input_points_bx1xNx2[i,0,0,1].cpu().numpy()//16, input_points_bx1xNx2[i,0,0,0].cpu().numpy()//16] = 1
                                plt.imsave('../l_sam_experiment/{}_point.png'.format(i), arr, cmap='gray')
                        # wang ----------------

                        output_masks_bx1x1xHxW, cls_predictions_bx1xK, iou_predictions_bx1x1 = efficientsam_ti_custom(
                            image_bx3xHxW,
                            input_points_bx1xNx2,
                            input_labels_bx1xN,
                        )

                        # wang ----------------
                        if bidx == 0:
                            for i in range(8):
                                arr = output_masks_bx1x1xHxW[i,0,0,:,:].cpu().numpy()
                                plt.imsave('../l_sam_experiment/{}_outputmask.png'.format(i), arr, cmap='gray')
                        # wang ----------------

                        
                        mask_bx3xHxW = torch.ge(output_masks_bx1x1xHxW[:, 0, :, :, :], 0)
                        mask_bx1xHxW = mask_bx3xHxW[:, :1, :, :].to(device=device)
                        Y_bx1xHxW = Y_bx1xHxW[:, :, :, :].to(device=device)

                        # wang ----------------
                        if bidx == 0:
                            for i in range(8):
                                arr = mask_bx1xHxW[i,0,:,:].cpu().numpy()
                                plt.imsave('../l_sam_experiment/{}_outputmask_t.png'.format(i), arr, cmap='gray')
                        # wang ----------------

                        # Evaluate segmentation with updated mIoU calculation
                        seg_miou, seg_fg_iou, seg_bg_iou = evaluate_segmentation(mask_bx1xHxW, Y_bx1xHxW)

                        cls_predictions_bx1xK_s.append(cls_predictions_bx1xK.squeeze(1))
                        Y_cls_bx1_s.append(Y_cls_bx1+1)

                        Y_cls_b = Y_cls_bx1.squeeze(1).to(device=device)

                        seg_pred_bx1xHxW = nn.functional.sigmoid(output_masks_bx1x1xHxW[:, 0, :, :, :]).float() - 0.5
                        cls_pred_bxK = cls_predictions_bx1xK.squeeze(1)
                        pred_bx1KxHxW = merge_seg_cls_new(seg_pred_bx1xHxW, cls_pred_bxK)

                        # pred_bxHxW = merge_seg_label(mask_bx1xHxW, torch.argmax(cls_predictions_bx1xK.squeeze(1), dim=1))

                        # pred_bxHxW = torch.argmax(pred_bx1KxHxW, dim=1).squeeze(1)
                        label_bxHxW = merge_seg_label(Y_bx1xHxW, Y_cls_b)

                        loss_mean.append(diceloss(pred_bx1KxHxW, label_bxHxW).item())

                        # mcmiou = compute_multiclass_iou(pred_bxHxW.to(device=device), label_bxHxW.to(device=device), class_num, ignore_index=0)
                        mcmiou = compute_multiclass_iou(pred_bx1KxHxW, label_bxHxW)

                        ks.extend(Y_cls_bx1[:, 0].detach().tolist())
                        seg_miou_s.extend(seg_miou)
                        seg_fg_iou_s.extend(seg_fg_iou)
                        seg_bg_iou_s.extend(seg_bg_iou)
                        cls_overall_miou_bsize_s.append([mcmiou, b])

                    seg_miou_s = np.array(seg_miou_s)
                    seg_fg_iou_s = np.array(seg_fg_iou_s)
                    seg_bg_iou_s = np.array(seg_bg_iou_s)

                    unique_ks = set(ks)
                    ks = np.array(ks)

                    mean_seg_miou = seg_miou_s.mean()
                    mean_seg_fg_iou = seg_fg_iou_s.mean()
                    mean_seg_bg_iou = seg_bg_iou_s.mean()
                    mean_cls_foreground_miou = sum([miou * bsize for miou, bsize in cls_overall_miou_bsize_s]) / sum([bsize for miou, bsize in cls_overall_miou_bsize_s])

                    #scheduler.step(mean_cls_foreground_miou.item())
                    scheduler.step()

                    writer.add_scalar('Loss_cls_val', np.array(loss_mean).mean(), global_step)
                    writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)

                    predictions = torch.argmax(torch.cat(cls_predictions_bx1xK_s, dim=0).squeeze(1), dim=1).detach().to('cuda') 
                    targets = torch.cat(Y_cls_bx1_s, dim=0).squeeze(1).detach().T.to('cuda') 
                    mask = (predictions != 0).to('cuda') 

                    filtered_predictions = predictions[mask]
                    filtered_targets = targets[mask]

                    accuracy = np.sum(filtered_predictions.cpu().numpy() == filtered_targets.cpu().numpy()) / len(filtered_predictions)

                    print(f"Filtered Accuracy Val: {accuracy}")
                    writer.add_scalar('Filtered Accuracy Val', accuracy, global_step)


                    print("lvis")
                    print("mean_seg_miou")
                    print(mean_seg_miou.item())
                    print("mean_seg_fg_iou")
                    print(mean_seg_fg_iou.item())
                    print("mean_seg_bg_iou")
                    print(mean_seg_bg_iou.item())
                    print("mean_cls_foreground_miou")
                    print(mean_cls_foreground_miou.cpu().numpy().item())
                    print("cls_mean")
                    print(0.5 * mean_cls_foreground_miou.cpu().numpy().item() + 0.5 * mean_seg_bg_iou.item())
                    # print(df[df['label'] == 'MEAN'])

                    with open('learning_to_downsample_ziqi.txt', 'a', encoding='utf-8') as f:
                        # print(targets.tolist(), file=f)
                        print("lvis")
                        print("mean_seg_miou", file=f)
                        print(mean_seg_miou.item(), file=f)
                        print("mean_seg_fg_iou", file=f)
                        print(mean_seg_fg_iou.item(), file=f)
                        print("mean_seg_bg_iou", file=f)
                        print(mean_seg_bg_iou.item(), file=f)
                        print("mean_cls_foreground_miou", file=f)
                        print(mean_cls_foreground_miou.cpu().numpy().item(), file=f)
                        print("cls_mean", file=f)
                        print(0.5 * mean_cls_foreground_miou.cpu().numpy().item() + 0.5 * mean_seg_bg_iou.item(), file=f)
