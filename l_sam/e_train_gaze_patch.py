import os
import sys
import pdb
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from datetime import datetime

from torch import nn

from d_model.nn_A1_tools import merge_seg_cls, merge_seg_label
from d_model.nn_A1_tools import merge_seg_cls_new, merge_seg_label_new
from l_sam.forveated_sam.foveated_sam_gaze_patch import build_foveated_sam
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

task_name = 'T4:10vitgaussian+cls+temperature0.5+weight1,0,50,1000,50+sigma_div100+relpos0+DW0.005(min30)'

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
    device = torch.device("cuda")
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
sigmoid = nn.Sigmoid()
focaloss = FocalLoss()


class EmbeddingClassificationLoss(nn.Module):
    """
    基于分割 mask 选择 embedding 并计算分类损失
    """
    def __init__(self, epsilon=1e-6):
        super(EmbeddingClassificationLoss, self).__init__()
        self.epsilon = epsilon
        self.debug_count = 0
        self.criterion = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()  # 添加 BCE loss
        self.temperature = 0.5  # 添加温度参数

    def forward(self, model, mask, class_label, images=None):
        """
        Args:
            model: DeformSegmentationModule 模型，包含 embeddings
            mask: 分割掩码 [B, 1, H, W]
            class_label: 类别标签 [B]
            images: 原始图像 [B, 3, H, W]，可选，用于可视化
        Returns:
            loss: 分类损失
        """
        B, _, H, W = mask.shape
        
        # 获取模型中的 embeddings，形状为 [B, H, W, C]
        embeddings = model.image_encoder.segmentation_module.embeddings
        
        # 对 mask 进行下采样以匹配 embedding map 的大小
        ds_factor_h = H // embeddings.shape[1]
        ds_factor_w = W // embeddings.shape[2]
        
        # 使用平均池化来下采样 mask
        downsampled_mask = F.avg_pool2d(mask, kernel_size=(ds_factor_h, ds_factor_w), 
                                       stride=(ds_factor_h, ds_factor_w))  # [B, 1, H_emb, W_emb]
        
        # 将 downsampled_mask 转换为与 embeddings 相同的格式 [B, H_emb, W_emb, 1]
        downsampled_mask = downsampled_mask.permute(0, 2, 3, 1)  # [B, H_emb, W_emb, 1]

        # 使用模型的 make_prediction 方法基于 mask 进行分类预测
        predictions, neg_similarity, pos_loss, sigma_loss, rho_loss = model.image_encoder.segmentation_module.make_prediction(downsampled_mask)
        
        # 获取每个样本中所有embedding的预测结果和数量
        all_fg_predictions = model.image_encoder.segmentation_module.all_fg_embeddings_predictions
        #all_bg_predictions = model.image_encoder.segmentation_module.all_bg_embeddings_predictions
        all_embeddings_predictions = model.image_encoder.segmentation_module.all_embeddings_predictions
        
        # 计算每个样本内所有embedding的分类损失平均值，然后对所有样本取平均
        sample_losses = []
        for b in range(B):
            # 获取当前样本中所有embedding的预测结果
            sample_preds = all_fg_predictions[b]  # 形状为[N, 51]的张量
            #sample_bg_preds = all_bg_predictions[b]  # 形状为[N, 51]的张量
            
            # 扩展类别标签以匹配每个embedding [N, 52]
            expanded_label = class_label[b].expand(sample_preds.size(0))
            #expanded_label_bg = torch.tensor([51]).repeat(sample_bg_preds.size(0)).to(device)

            # 一次性计算所有embedding的损失
            sample_loss = self.criterion(sample_preds / self.temperature, expanded_label)
            #sample_loss_bg = self.criterion(sample_bg_preds / self.temperature, expanded_label_bg)
            #sample_losses.append(sample_loss + sample_loss_bg)
            sample_losses.append(sample_loss)
        
        # 计算批次内所有样本的平均损失
        loss = torch.stack(sample_losses).mean()
        
        # 可视化 (随机进行)
        if random.randint(0, 400) == 1 and images is not None:
            import matplotlib.pyplot as plt
            import os
            import numpy as np
            
            # 确保保存目录存在
            os.makedirs('../l_sam_experiment', exist_ok=True)
            
            self.debug_count += 1
            
            for b in range(min(B, 4)):  # 只可视化前4个样本
                # 保存原图
                img = images[b].permute(1, 2, 0).float().detach().cpu().numpy()
                if img.max() > 1.0:
                    img = img / 255.0
                plt.imsave(f'../l_sam_experiment/{task_name}_embedding_cls_{self.debug_count}_{b}_1_original.png', img)
                
                # 保存 mask
                plt.imsave(f'../l_sam_experiment/{task_name}_embedding_cls_{self.debug_count}_{b}_2_mask.png', 
                          mask[b, 0].float().detach().cpu().numpy(), cmap='gray')
                
                # 保存下采样后的 mask
                plt.imsave(f'../l_sam_experiment/{task_name}_embedding_cls_{self.debug_count}_{b}_3_downsampled_mask.png', 
                          downsampled_mask[b, :, :, 0].float().detach().cpu().numpy(), cmap='gray')
                
                # 获取预测和真实类别
                pred_class = torch.argmax(predictions[b]).item()
                true_class = class_label[b].item()
                all_score4true = F.softmax(all_embeddings_predictions[b], dim=1)[:, true_class].view(40, 40)

                # 可视化预测的 saliency map
                if hasattr(model.image_encoder.segmentation_module, 'xs') and model.image_encoder.segmentation_module.xs is not None:
                    saliency_map = model.image_encoder.segmentation_module.xs[b].float().detach().cpu().numpy()
                    allscore4true = all_score4true.float().detach().cpu().numpy()
                    
                    # 显示 saliency map
                    fig, ax = plt.subplots(figsize=(8, 8))
                    im = ax.imshow(allscore4true, cmap='hot')
                    ax.set_title(f"allscore4true (predicted: {pred_class}, GT: {true_class})")
                    ax.axis('off')
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    plt.tight_layout()
                    plt.savefig(f'../l_sam_experiment/{task_name}_embedding_cls_{self.debug_count}_{b}_4_allscore4true.png',
                              bbox_inches='tight', dpi=150)
                    plt.close(fig)

                    # 显示 saliency map
                    fig, ax = plt.subplots(figsize=(8, 8))
                    im = ax.imshow(saliency_map, cmap='hot')
                    ax.set_title(f"Saliency Map (predicted: {pred_class}, GT: {true_class})")
                    ax.axis('off')
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    plt.tight_layout()
                    plt.savefig(f'../l_sam_experiment/{task_name}_embedding_cls_{self.debug_count}_{b}_5_saliency_map.png',
                              bbox_inches='tight', dpi=150)
                    plt.close(fig)
                    
                    # 叠加 saliency map 和原图
                    if images is not None:
                        img = images[b].permute(1, 2, 0).float().detach().cpu().numpy()
                        if img.max() > 1.0:
                            img = img / 255.0
                            
                        # 对 saliency map 进行上采样以匹配原图大小
                        from scipy.ndimage import zoom
                        zoom_factor = img.shape[0] / saliency_map.shape[0]
                        saliency_map_resized = zoom(saliency_map, zoom_factor, order=1)
                        
                        # 可视化叠加效果
                        fig, ax = plt.subplots(figsize=(8, 8))
                        ax.imshow(img)
                        overlay = ax.imshow(saliency_map_resized, cmap='hot', alpha=0.5)
                        ax.set_title(f"Saliency Map (pred: {pred_class}, GT: {true_class})")
                        ax.axis('off')
                        plt.colorbar(overlay, ax=ax, fraction=0.046, pad=0.04)
                        plt.tight_layout()
                        plt.savefig(f'../l_sam_experiment/{task_name}_embedding_cls_{self.debug_count}_{b}_6_saliency_overlay.png',
                                  bbox_inches='tight', dpi=150)
                        plt.close(fig)
        margin = 0.0 
        loss_regularization = torch.clamp(neg_similarity - margin, min=0.0)
        loss_regularization = loss_regularization.mean()
        
        loss_pos = pos_loss.mean()
        loss_sigma = sigma_loss.mean()
        loss_rho = rho_loss.mean()
        
        return loss, loss_regularization, loss_pos, loss_sigma, loss_rho

# 初始化 embedding 分类损失函数
embedding_cls_loss = EmbeddingClassificationLoss().to(device)

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
            #dataset_train = DatasetLVIS('sp640_60000_sam_vit_b', dataset_partition='train')
            dataset_train = DatasetLVIS('sp640_60000', dataset_partition='train')
            dataloader_train = CustomDataLoader(dataset_train, xrange=trange, cache=False)
            #dataset_valid = DatasetLVIS('sp640_12000_sam_vit_b', dataset_partition='valid')
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
        #lr1 = 1e-3
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

        # 加载模型
        #print('\n加载模型!\n')
        #checkpoint_path = os.path.join("/home/external/DynamicFocus_new_ziqi/l_sam_experiment/T3:10vitgaussian+cls+temperature0.5+weight1,0,50,1000,50+sigma_div100+relpos0+DWcheckpoint.pt")
        #efficientsam_ti_custom.load_state_dict(torch.load(checkpoint_path))

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
                    #for name, param in efficientsam_ti_custom.image_encoder.segmentation_module.wang_net_compress.named_parameters():
                    #    param.requires_grad_(True)
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

                    embedding_classification_loss, loss_regularization, pos_loss, sigma_loss, rho_loss = embedding_cls_loss(efficientsam_ti_custom, cur_Y_bx1xHxW, Y_cls_b, image_bx3xHxW)
                    
                    seg_loss = diceloss(pred_bx1KxHxW, label_bxHxW)
                    
                    lambda_embedding = 1  # 可以根据需要调整权重
                    lambda_regularization = 0 # 可以根据需要调整权重
                    lambda_pos = 50
                    lambda_sigma = 1000
                    lambda_rho = 50

                    loss = (seg_loss +
                            lambda_embedding * embedding_classification_loss +
                            lambda_regularization * loss_regularization +
                            lambda_pos * pos_loss +
                            lambda_sigma * sigma_loss +
                            lambda_rho * rho_loss)
                    if torch.isnan(loss):
                        print(f"\n警告：loss为NaN，已将其设置为0")
                        loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
                    loss_mean.append(loss.item())
                    
                    # 记录各部分损失，便于监控
                    writer.add_scalar('Loss/seg_loss', seg_loss.item(), global_step)
                    writer.add_scalar('Loss/embedding_cls_loss', embedding_classification_loss.item(), global_step)
                    writer.add_scalar('Loss/loss_regularization', loss_regularization.item(), global_step)
                    writer.add_scalar('Loss/loss_pos', pos_loss.item(), global_step)
                    writer.add_scalar('Loss/loss_sigma', sigma_loss.item(), global_step)
                    writer.add_scalar('Loss/loss_rho', rho_loss.item(), global_step)
                    
                    if bidx % 50 == 0:  # 每50个批次打印一次损失值
                        print(f"\nBatch {bidx}: Seg Loss: {seg_loss.item():.4f}, Embedding Cls Loss: {embedding_classification_loss.item():.4f}, Loss Regularization: {loss_regularization.item():.4f}, Loss Pos: {pos_loss.item()*lambda_pos:.4f}, Loss Sigma: {sigma_loss.item()*lambda_sigma:.4f}, Loss Rho: {rho_loss.item()*lambda_rho:.4f}")
                        token_count_list = efficientsam_ti_custom.image_encoder.segmentation_module.token_count_list
                        if len(token_count_list) > 0:
                            print('\ntoken_mean', np.mean(np.concatenate(token_count_list)), 'token_max', np.max(np.concatenate(token_count_list)), 'token_min', np.min(np.concatenate(token_count_list)))
                            efficientsam_ti_custom.image_encoder.segmentation_module.token_count_list = []

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

            if epoch > 0:
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
                        if bidx % 100 == 0:
                            for i in range(8):
                                import matplotlib.pyplot as plt
                                arr = image_bx3xHxW[i,0,:,:].cpu().numpy()
                                plt.imsave('../l_sam_experiment/{}_{}_img.png'.format(task_name,i), arr, cmap='gray')
                                arr = Y_bx1xHxW[i,0,:,:].cpu().numpy()
                                plt.imsave('../l_sam_experiment/{}_{}_mask.png'.format(task_name,i), arr, cmap='gray')
                                arr = torch.zeros(40, 40)
                                arr[input_points_bx1xNx2[i,0,0,1].cpu().numpy()//16, input_points_bx1xNx2[i,0,0,0].cpu().numpy()//16] = 1
                                plt.imsave('../l_sam_experiment/{}_{}_point.png'.format(task_name,i), arr, cmap='gray')
                        # wang ----------------

                        output_masks_bx1x1xHxW, cls_predictions_bx1xK, iou_predictions_bx1x1 = efficientsam_ti_custom(
                            image_bx3xHxW,
                            input_points_bx1xNx2,
                            input_labels_bx1xN,
                        )

                        # wang ----------------
                        if bidx % 100 == 0:
                            for i in range(8):
                                arr = output_masks_bx1x1xHxW[i,0,0,:,:].cpu().numpy()
                                plt.imsave('../l_sam_experiment/{}_{}_outputmask.png'.format(task_name,i), arr, cmap='gray')
                        # wang ----------------

                        
                        mask_bx3xHxW = torch.ge(output_masks_bx1x1xHxW[:, 0, :, :, :], 0)
                        mask_bx1xHxW = mask_bx3xHxW[:, :1, :, :].to(device=device)
                        Y_bx1xHxW = Y_bx1xHxW[:, :, :, :].to(device=device)

                        # wang ----------------
                        if bidx % 100 == 0:
                            for i in range(8):
                                arr = mask_bx1xHxW[i,0,:,:].cpu().numpy()
                                plt.imsave('../l_sam_experiment/{}_{}_outputmask_t.png'.format(task_name,i), arr, cmap='gray')
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
                    
                    # 保存模型
                    checkpoint_path = os.path.join('/home/external/DynamicFocus_new_ziqi/l_sam_experiment', task_name + "checkpoint.pt")
                    torch.save(efficientsam_ti_custom.state_dict(), checkpoint_path)
                    print(f"model saved to: {checkpoint_path}")
                    
                    optimizer.step()