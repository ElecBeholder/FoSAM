import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from datetime import datetime

from torch import nn

from d_model.nn_A1_tools import merge_seg_cls, merge_seg_label
from d_model.nn_A1_tools import merge_seg_cls_new, merge_seg_label_new
from l_sam.forveated_sam.foveated_sam_implicit_backclass_cls_useimage_embed_crop_prune_token_advance_res101_saliency_joint import build_foveated_sam
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
    matplotlib.use('TkAgg')  # 有图形界面，使用 TkAgg

os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# 建议在脚本开头就创建
writer = SummaryWriter(log_dir='/root/autodl-tmp/DynamicFocus/l_sam_experiment')

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
        # 确保 pred 在 (0,1) 范围内并进行 clamp
        pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)

        # 计算 BCE loss
        bce_loss = - (target * torch.log(pred) + (1 - target) * torch.log(1 - pred))

        # 使用一个公式避免条件判断，元素级计算modulating_factor
        # 当 target=1 时，(1-pred) 有效；当 target=0 时，pred 有效。
        modulating_factor = (target * (1 - pred) + (1 - target) * pred) ** self.gamma

        # 计算alpha_factor
        alpha_factor = self.alpha * target + (1 - self.alpha) * (1 - target)

        # 计算focal loss
        focal_loss = alpha_factor * modulating_factor * bce_loss

        # 对所有维度求均值
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
        # 确保 pred 和 target 的形状是 [B, 1, H, W]
        pred = pred.view(pred.size(0), -1)  # [B, 1*H*W]
        target = target.view(target.size(0), -1)  # [B, 1*H*W]

        # 计算交集 (element-wise multiplication)
        intersection = (pred * target).sum(dim=1)  # 按每个样本计算 sum
        union = pred.sum(dim=1) + target.sum(dim=1)
        # 计算 Dice 系数
        dice_coeff = (2 * intersection + self.smooth) / (union + self.smooth)

        # Dice Loss
        dice_loss = 1.0 - dice_coeff

        # 取 batch 平均值
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

if __name__ == '__main__':

    old_state_dict = torch.load("/root/autodl-tmp/DynamicFocus/l_sam/efficient_sam_vits.singlemask.pt")['model']
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
            dataset_train = DatasetLVIS('sp20000', dataset_partition='train')
            dataloader_train = CustomDataLoader(dataset_train, xrange=trange, cache=False)
            dataset_valid = DatasetLVIS('sp4000', dataset_partition='valid')
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
        # freeze_parameters(efficientsam_ti_custom.mask_decoder.classify_model)

        # efficientsam_ti_custom.classify_model.apply(init_weights_random)

        # https://github.com/ziqi-jin/finetune-anything/blob/main/config/semantic_seg.yaml
        # lr1 = 5e-4
        # # lr2 = 3e-4
        # lr2 = 1e-3
        # num_epochs = 10


        # optimizer = torch.optim.Adam(
        #     [
        #         {"params": efficientsam_ti_custom.mask_decoder.parameters(), "lr": lr1},
        #         # {"params": efficientsam_ti_custom.mask_decoder.classify_model.parameters(), "lr": lr2},
        #     ]
        # )

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.90, patience=1, mode='max', min_lr=1e-6)

        # dpath = os.path.join(preset.dpath_training_records, f'{efficientsam_ti_custom.__class__.__name__}_{mode}_{HW_RAW_SIZE}x{HW_RAW_SIZE}_{HW_image_input_size}x{HW_image_input_size}_{datetime.now().strftime("%Y%m%d_%H%M")}')

        # os.makedirs(dpath, exist_ok=True)
        # # multi_class_miou_evaluator = MulticlassJaccardIndex(num_classes=class_num + 1).to(device=device)
        # global_step = 0
        # for epoch in trange(num_epochs):
        #     batch_size = 20

        #     optimizer.zero_grad()

        #     if epoch > 0:
        #         loss_diceloss_s = []
        #         loss_cme_s = []
        #         loss_s = []
        #         efficientsam_ti_custom.train()

        #         for bidx, bparts in enumerate(dataloader_train.get_iterator(batch_size=batch_size, device=torch.device("cpu"), shuffle=True, xrange=trange)):
        #             X_bx4xHxW, F_bx2, Y_bx1xHxW, Y_cls_bx1 = bparts
        #             X_bx4xHxW = avg_pool(X_bx4xHxW)
        #             Y_bx1xHxW = max_pool(Y_bx1xHxW)
        #             """
        #             image_1x3xHxW.shape torch.Size([1, 3, 512, 512])
        #             input_points_1x1xNx2.shape torch.Size([1, 1, 1, 2])
        #             input_labels_1x1xN.shape torch.Size([1, 1, 1])
        #             """

        #             image_bx3xHxW = X_bx4xHxW[:, :3, :, :].to(device=device)

        #             input_points_bx1xNx2 = (F_bx2[:, None, None, [1, 0]] * HW_image_input_size).to(device=device, dtype=torch.int64)
        #             b, _ = F_bx2.shape
        #             input_labels_bx1xN = torch.ones(b, 1, 1).to(device=device, dtype=torch.int64)

        #             #add 
        #             s_BxHTxWTx1 = (average_pool(Y_bx1xHxW, 16).permute(0, 2, 3, 1))
        #             # s_bin_selected_BxHMxWMx1 = (get_merge_map_object(s_BxHTxWTx1).to(device=device) > 0.5)
        #             s_bin_selected_BxHMxWMx1 = None

        #             output_masks_bx1x1xHxW, cls_predictions_bx1xK, iou_predictions_bx1x1 = efficientsam_ti_custom(
        #                 image_bx3xHxW,
        #                 input_points_bx1xNx2,
        #                 input_labels_bx1xN,
        #                 s_bin_selected_BxHMxWMx1=s_bin_selected_BxHMxWMx1,
        #             )
        #             seg_pred_bx1xHxW = nn.functional.sigmoid(output_masks_bx1x1xHxW[:, 0, :, :, :]).float() - 0.5
        #             cur_Y_bx1xHxW = Y_bx1xHxW[:, :, :, :].to(device=device)

        #             Y_cls_b = Y_cls_bx1.squeeze(1).to(device=device)

        #             cls_pred_bxK = cls_predictions_bx1xK.squeeze(1)
        #             # print(cls_pred_bxK)

        #             pred_bx1KxHxW = merge_seg_cls_new(seg_pred_bx1xHxW, cls_pred_bxK)
        #             label_bxHxW = merge_seg_label_new(cur_Y_bx1xHxW, Y_cls_b)
        #             # with open('output_lvis_640_cls_implicit_backclass_3e4_12000.txt', 'a', encoding='utf-8') as f:
        #             #     print(pred_bx1KxHxW.size(), file=f)
        #             #     print(Y_cls_b, file=f)
        #             # print(label_bxHxW)


        #             # loss = diceloss(seg_pred_bx1xHxW, cur_Y_bx1xHxW) + celoss(pred_bx1KxHxW, label_bxHxW)
        #             loss = diceloss_1(seg_pred_bx1xHxW + 0.5, cur_Y_bx1xHxW)

        #             # loss_s.append(loss.detach())

        #             optimizer.zero_grad()
        #             loss.backward()
        #             optimizer.step()

        #             global_step = global_step + 1
        #             writer.add_scalar('Loss_binary', diceloss_1(seg_pred_bx1xHxW + 0.5, cur_Y_bx1xHxW).item(), global_step)

        #         # print(f'\nEpoch [{epoch + 1}/{num_epochs}], Loss[Dice]: {np.array(loss_s).mean():.4f} kr={scheduler.get_last_lr()}', end='')

        #     if True:
        #         efficientsam_ti_custom.eval()
        #         with torch.no_grad():
        #             col2elems = {
        #                 'label': [],
        #                 'seg_miou': [],
        #                 'seg_fg_iou': [],
        #                 'seg_bg_iou': [],
        #                 'cls_fg_iou': [],
        #                 'cls_miou': [],

        #             }

        #             seg_miou_s, seg_fg_iou_s, seg_bg_iou_s = [], [], []
        #             cls_overall_miou_bsize_s = []
        #             cls_predictions_bx1xK_s = []
        #             Y_cls_bx1_s = []

        #             cls_fs_s = []
        #             ks = []

        #             for bidx, bparts in enumerate(dataloader_valid.get_iterator(batch_size=5, device=torch.device("cpu"), shuffle=True, xrange=range)):
        #                 X_bx4xHxW, F_bx2, Y_bx1xHxW, Y_cls_bx1 = bparts
        #                 X_bx4xHxW = avg_pool(X_bx4xHxW)
        #                 Y_bx1xHxW = max_pool(Y_bx1xHxW)

        #                 image_bx3xHxW = X_bx4xHxW[:, :3, :, :].to(device=device)
        #                 input_points_bx1xNx2 = (F_bx2[:, None, None, [1, 0]] * HW_image_input_size).to(device=device, dtype=torch.int64)
        #                 b, _ = F_bx2.shape
        #                 input_labels_bx1xN = torch.ones(b, 1, 1).to(device=device, dtype=torch.int64)

        #                 #add 
        #                 s_BxHTxWTx1 = (average_pool(Y_bx1xHxW, 16).permute(0, 2, 3, 1))
        #                 # s_bin_selected_BxHMxWMx1 = (get_merge_map_object(s_BxHTxWTx1).to(device=device) > 0.5)
        #                 s_bin_selected_BxHMxWMx1 = None

        #                 output_masks_bx1x1xHxW, cls_predictions_bx1xK, iou_predictions_bx1x1 = efficientsam_ti_custom(
        #                     image_bx3xHxW,
        #                     input_points_bx1xNx2,
        #                     input_labels_bx1xN,
        #                     s_bin_selected_BxHMxWMx1=s_bin_selected_BxHMxWMx1,
        #                 )

                        
        #                 mask_bx3xHxW = torch.ge(output_masks_bx1x1xHxW[:, 0, :, :, :], 0)
        #                 mask_bx1xHxW = mask_bx3xHxW[:, :1, :, :].to(device=device)
        #                 Y_bx1xHxW = Y_bx1xHxW[:, :, :, :].to(device=device)

        #                 # Evaluate segmentation with updated mIoU calculation
        #                 seg_miou, seg_fg_iou, seg_bg_iou = evaluate_segmentation(mask_bx1xHxW, Y_bx1xHxW)

        #                 cls_predictions_bx1xK_s.append(cls_predictions_bx1xK.squeeze(1))
        #                 Y_cls_bx1_s.append(Y_cls_bx1)

        #                 Y_cls_b = Y_cls_bx1.squeeze(1).to(device=device)

        #                 seg_pred_bx1xHxW = nn.functional.sigmoid(output_masks_bx1x1xHxW[:, 0, :, :, :]).float() - 0.5
        #                 cls_pred_bxK = cls_predictions_bx1xK.squeeze(1)
        #                 pred_bx1KxHxW = merge_seg_cls_new(seg_pred_bx1xHxW, cls_pred_bxK)

        #                 # pred_bxHxW = merge_seg_label(mask_bx1xHxW, torch.argmax(cls_predictions_bx1xK.squeeze(1), dim=1))

        #                 # pred_bxHxW = torch.argmax(pred_bx1KxHxW, dim=1).squeeze(1)
        #                 label_bxHxW = merge_seg_label(Y_bx1xHxW, Y_cls_b)
        #                 # mcmiou = compute_multiclass_iou(pred_bxHxW.to(device=device), label_bxHxW.to(device=device), class_num, ignore_index=0)
        #                 mcmiou = compute_multiclass_iou(pred_bx1KxHxW, label_bxHxW)

        #                 ks.extend(Y_cls_bx1[:, 0].detach().tolist())
        #                 seg_miou_s.extend(seg_miou)
        #                 seg_fg_iou_s.extend(seg_fg_iou)
        #                 seg_bg_iou_s.extend(seg_bg_iou)
        #                 cls_overall_miou_bsize_s.append([mcmiou, b])

        #             seg_miou_s = np.array(seg_miou_s)
        #             seg_fg_iou_s = np.array(seg_fg_iou_s)
        #             seg_bg_iou_s = np.array(seg_bg_iou_s)

        #             unique_ks = set(ks)
        #             ks = np.array(ks)

        #             mean_seg_miou = seg_miou_s.mean()
        #             mean_seg_fg_iou = seg_fg_iou_s.mean()
        #             mean_seg_bg_iou = seg_bg_iou_s.mean()
        #             mean_cls_foreground_miou = sum([miou * bsize for miou, bsize in cls_overall_miou_bsize_s]) / sum([bsize for miou, bsize in cls_overall_miou_bsize_s])

        #             scheduler.step(mean_cls_foreground_miou)
        #             # predictions = torch.argmax(torch.cat(cls_predictions_bx1xK_s, dim=0).squeeze(1), dim=1).detach()
        #             # targets = torch.cat(Y_cls_bx1_s, dim=0).squeeze(1).detach().T + 1
        #             # accuracy = np.sum(predictions == targets) / len(predictions)
        #             # print(predictions.tolist())
        #             # print(targets.tolist())
        #             # print(f"accuracy: {accuracy:.4f}")

        #             # 调用评估函数并打印结果

        #             # col2elems['label'].append('MEAN')
        #             # col2elems['seg_miou'].append(mean_seg_miou)
        #             # col2elems['seg_fg_iou'].append(mean_seg_fg_iou)
        #             # col2elems['seg_bg_iou'].append(mean_seg_bg_iou)
        #             # col2elems['cls_fg_iou'].append(mean_cls_foreground_miou)
        #             # col2elems['cls_miou'].append(0.5 * mean_cls_foreground_miou + 0.5 * mean_seg_bg_iou)

        #             # df = pd.DataFrame(col2elems)
        #             # print()
        #             # print(df[df['label'] == 'MEAN'])

        #             print("lvis")
        #             print("mean_seg_miou")
        #             print(mean_seg_miou.item())
        #             print("mean_seg_fg_iou")
        #             print(mean_seg_fg_iou.item())
        #             print("mean_seg_bg_iou")
        #             print(mean_seg_bg_iou.item())
        #             print("mean_cls_foreground_miou")
        #             print(mean_cls_foreground_miou.cpu().numpy().item())
        #             print("cls_mean")
        #             print(0.5 * mean_cls_foreground_miou.cpu().numpy().item() + 0.5 * mean_seg_bg_iou.item())
        #             # print(df[df['label'] == 'MEAN'])

        #             with open('output_lvis_640_new_two_branch_window_20_res101_small.txt', 'a', encoding='utf-8') as f:
        #                 # print(targets.tolist(), file=f)
        #                 print("lvis")
        #                 print("mean_seg_miou", file=f)
        #                 print(mean_seg_miou.item(), file=f)
        #                 print("mean_seg_fg_iou", file=f)
        #                 print(mean_seg_fg_iou.item(), file=f)
        #                 print("mean_seg_bg_iou", file=f)
        #                 print(mean_seg_bg_iou.item(), file=f)
        #                 print("mean_cls_foreground_miou", file=f)
        #                 print(mean_cls_foreground_miou.cpu().numpy().item(), file=f)
        #                 print("cls_mean", file=f)
        #                 print(0.5 * mean_cls_foreground_miou.cpu().numpy().item() + 0.5 * mean_seg_bg_iou.item(), file=f)
                        


        lr1 = 5e-4
        # lr2 = 3e-4
        # lr2 = 5e-4 # setting 1 slow 70 epoch 0.027
        lr2 = 1e-2
        num_epochs = 200


        optimizer = torch.optim.Adam(
            [
                {"params": efficientsam_ti_custom.mask_decoder.parameters(), "lr": lr1},
                {"params": efficientsam_ti_custom.classifier.parameters(), "lr": lr2},
            ]
        )

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=2, mode='max', min_lr=1e-6)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, mode='max', min_lr=1e-6)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,30,45,60,75,90,105,130,145,160,175], gamma=0.5)


        dpath = os.path.join(preset.dpath_training_records, f'{efficientsam_ti_custom.__class__.__name__}_{mode}_{HW_RAW_SIZE}x{HW_RAW_SIZE}_{HW_image_input_size}x{HW_image_input_size}_{datetime.now().strftime("%Y%m%d_%H%M")}')

        os.makedirs(dpath, exist_ok=True)
        # multi_class_miou_evaluator = MulticlassJaccardIndex(num_classes=class_num + 1).to(device=device)
        global_step = 0
        for epoch in trange(num_epochs):
            batch_size = 16

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
                    s_BxHTxWTx1 = (average_pool(Y_bx1xHxW, 16).permute(0, 2, 3, 1))
                    s_bin_selected_BxHMxWMx1 = (get_merge_map_object(s_BxHTxWTx1).to(device=device) > 0.5)
                    # s_bin_selected_BxHMxWMx1 = None

                    output_masks_bx1x1xHxW, cls_predictions_bx1xK, iou_predictions_bx1x1 = efficientsam_ti_custom(
                        image_bx3xHxW,
                        input_points_bx1xNx2,
                        input_labels_bx1xN,
                        s_bin_selected_BxHMxWMx1=s_bin_selected_BxHMxWMx1,
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
                    loss = diceloss(pred_bx1KxHxW, label_bxHxW)
                    loss_mean.append(loss.item())

                    # loss_s.append(loss.detach())

                    optimizer.zero_grad()
                    loss.backward()
                    # if epoch == 1:
                    #     for name, param in efficientsam_ti_custom.named_parameters():
                    #         print(f"{name} grad: {param.grad.norm()}")
                    optimizer.step()

                writer.add_scalar('Loss_cls_train', np.array(loss_mean).mean(), global_step)

                predictions = torch.argmax(torch.cat(cls_predictions_bx1xK_s, dim=0).squeeze(1), dim=1).detach().to('cuda') 
                targets = torch.cat(Y_cls_bx1_s, dim=0).squeeze(1).detach().T.to('cuda')

                # 创建布尔掩码，筛选 predictions 不为 0 的索引
                mask = (predictions != 0).to('cuda') 

                # 使用掩码筛选 predictions 和 targets
                filtered_predictions = predictions[mask]
                filtered_targets = targets[mask]

                # 计算过滤后的准确率
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

                    for bidx, bparts in enumerate(dataloader_valid.get_iterator(batch_size=16, device=torch.device("cpu"), shuffle=True, xrange=range)):
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

                        #add 
                        s_BxHTxWTx1 = (average_pool(Y_bx1xHxW, 16).permute(0, 2, 3, 1))
                        s_bin_selected_BxHMxWMx1 = (get_merge_map_object(s_BxHTxWTx1).to(device=device) > 0.5)
                        # s_bin_selected_BxHMxWMx1 = None

                        output_masks_bx1x1xHxW, cls_predictions_bx1xK, iou_predictions_bx1x1 = efficientsam_ti_custom(
                            image_bx3xHxW,
                            input_points_bx1xNx2,
                            input_labels_bx1xN,
                            s_bin_selected_BxHMxWMx1=s_bin_selected_BxHMxWMx1,
                        )

                        
                        mask_bx3xHxW = torch.ge(output_masks_bx1x1xHxW[:, 0, :, :, :], 0)
                        mask_bx1xHxW = mask_bx3xHxW[:, :1, :, :].to(device=device)
                        Y_bx1xHxW = Y_bx1xHxW[:, :, :, :].to(device=device)

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

                    scheduler.step(mean_cls_foreground_miou.item())
                    # predictions = torch.argmax(torch.cat(cls_predictions_bx1xK_s, dim=0).squeeze(1), dim=1).detach()
                    # targets = torch.cat(Y_cls_bx1_s + 1, dim=0).squeeze(1).detach().T + 1
                    # accuracy = np.sum(predictions == targets) / len(predictions)
                    # # print(predictions.tolist())
                    # # print(targets.tolist())
                    # print(f"accuracy: {accuracy:.4f}")

                    writer.add_scalar('Loss_cls_val', np.array(loss_mean).mean(), global_step)
                    writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)

                    predictions = torch.argmax(torch.cat(cls_predictions_bx1xK_s, dim=0).squeeze(1), dim=1).detach().to('cuda') 
                    targets = torch.cat(Y_cls_bx1_s, dim=0).squeeze(1).detach().T.to('cuda') 

                    # 创建布尔掩码，筛选 predictions 不为 0 的索引
                    mask = (predictions != 0).to('cuda') 

                    # 使用掩码筛选 predictions 和 targets
                    filtered_predictions = predictions[mask]
                    filtered_targets = targets[mask]

                    # 计算过滤后的准确率
                    accuracy = np.sum(filtered_predictions.cpu().numpy() == filtered_targets.cpu().numpy()) / len(filtered_predictions)

                    print(f"Filtered Accuracy Val: {accuracy}")
                    writer.add_scalar('Filtered Accuracy Val', accuracy, global_step)

                    # 调用评估函数并打印结果

                    # col2elems['label'].append('MEAN')
                    # col2elems['seg_miou'].append(mean_seg_miou)
                    # col2elems['seg_fg_iou'].append(mean_seg_fg_iou)
                    # col2elems['seg_bg_iou'].append(mean_seg_bg_iou)
                    # col2elems['cls_fg_iou'].append(mean_cls_foreground_miou)
                    # col2elems['cls_miou'].append(0.5 * mean_cls_foreground_miou + 0.5 * mean_seg_bg_iou)

                    # df = pd.DataFrame(col2elems)
                    # print()
                    # print(df[df['label'] == 'MEAN'])

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

                    with open('output_lvis_640_new_two_branch_window_20_res101_small_padding_0.txt', 'a', encoding='utf-8') as f:
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