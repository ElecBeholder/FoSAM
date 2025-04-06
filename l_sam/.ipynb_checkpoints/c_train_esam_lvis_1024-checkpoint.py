import os
import sys

from torch import nn
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from d_model.nn_A2_loss import BMSELoss
import preset
from d_model.nn_A0_utils import init_weights_random, freeze_parameters, unfreeze_parameters
from l_sam.efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
import os
import platform
import torch.nn.functional as F
import matplotlib
import pandas as pd
from d_model.nn_A3_metrics import evaluate_segmentation
from l_sam.efficient_sam.efficient_sam import build_efficient_sam
from utility.plot_tools import plt_imgshow, plt_multi_imgshow, plt_show
from utility.torch_tools import str_tensor_shape
from utility.watch import Watch

system = platform.system()
if system == "Windows":
    matplotlib.use('TkAgg')  # 有图形界面，使用 TkAgg

os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"

import torch
import numpy as np

from tqdm import trange, tqdm

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
        dice_coeff = (2*intersection + self.smooth) / (union + self.smooth)

        # Dice Loss
        dice_loss = 1.0 - dice_coeff

        # 取 batch 平均值
        mean_dice_loss = dice_loss.mean()
        return mean_dice_loss

BFLoss = BinaryFocalLoss()
BMSE = BMSELoss()
diceloss = BatchDiceLoss()
sigmoid = nn.Sigmoid()



if __name__ == '__main__':
    old_state_dict = torch.load("/root/autodl-tmp/DynamicFocus/l_sam/efficient_sam_vitt.pt")

    # copy mask 1 to 0
    keys_copy = ['mask_decoder.output_hypernetworks_mlps.1.layers.0.0.weight',
                 'mask_decoder.output_hypernetworks_mlps.1.layers.0.0.bias',
                 'mask_decoder.output_hypernetworks_mlps.1.layers.1.0.weight',
                 'mask_decoder.output_hypernetworks_mlps.1.layers.1.0.bias',
                 'mask_decoder.output_hypernetworks_mlps.1.fc.weight',
                 'mask_decoder.output_hypernetworks_mlps.1.fc.bias']
    
    for key in keys_copy:
        old_state_dict['model'][key.replace('_mlps.1.', '_mlps.0.')] = old_state_dict['model'][key]
    
    old_state_dict['model']['mask_decoder.mask_tokens.weight'][0] = old_state_dict['model']['mask_decoder.mask_tokens.weight'][1]
    old_state_dict['model']['mask_decoder.iou_prediction_head.fc.weight'][0, :] = old_state_dict['model']['mask_decoder.iou_prediction_head.fc.weight'][1, :]
    old_state_dict['model']['mask_decoder.iou_prediction_head.fc.bias'][0] = old_state_dict['model']['mask_decoder.iou_prediction_head.fc.bias'][1]
    
    # delete key
    
    keys_del = ['mask_decoder.output_hypernetworks_mlps.1.layers.0.0.weight',
                'mask_decoder.output_hypernetworks_mlps.1.layers.0.0.bias',
                'mask_decoder.output_hypernetworks_mlps.1.layers.1.0.weight',
                'mask_decoder.output_hypernetworks_mlps.1.layers.1.0.bias',
                'mask_decoder.output_hypernetworks_mlps.1.fc.weight',
                'mask_decoder.output_hypernetworks_mlps.1.fc.bias',
                'mask_decoder.output_hypernetworks_mlps.2.layers.0.0.weight',
                'mask_decoder.output_hypernetworks_mlps.2.layers.0.0.bias',
                'mask_decoder.output_hypernetworks_mlps.2.layers.1.0.weight',
                'mask_decoder.output_hypernetworks_mlps.2.layers.1.0.bias',
                'mask_decoder.output_hypernetworks_mlps.2.fc.weight',
                'mask_decoder.output_hypernetworks_mlps.2.fc.bias',
                'mask_decoder.output_hypernetworks_mlps.3.layers.0.0.weight',
                'mask_decoder.output_hypernetworks_mlps.3.layers.0.0.bias',
                'mask_decoder.output_hypernetworks_mlps.3.layers.1.0.weight',
                'mask_decoder.output_hypernetworks_mlps.3.layers.1.0.bias',
                'mask_decoder.output_hypernetworks_mlps.3.fc.weight',
                'mask_decoder.output_hypernetworks_mlps.3.fc.bias']
    
    old_state_dict['model']['mask_decoder.mask_tokens.weight'] = old_state_dict['model']['mask_decoder.mask_tokens.weight'][:1]
    old_state_dict['model']['mask_decoder.iou_prediction_head.fc.weight'] = old_state_dict['model']['mask_decoder.iou_prediction_head.fc.weight'][:1, :]
    old_state_dict['model']['mask_decoder.iou_prediction_head.fc.bias'] = old_state_dict['model']['mask_decoder.iou_prediction_head.fc.bias'][:1]
    
    for key,item in old_state_dict['model'].items():
        print(key,item.shape)
    
    for key_del in keys_del:
        del old_state_dict['model'][key_del]
    torch.save(old_state_dict, os.path.join(preset.dpath_esam_weights, 'efficient_sam_vits.singlemask_train_lvis.pt'))


    if True:
        mode = 'lvis'

        avg_pool = nn.Identity()
        max_pool = nn.Identity()
        HW_RAW_SIZE = 0
        HW_image_input_size = 0

        if mode == 'lvis':
            upsample_factor = 1.6
            HW_RAW_SIZE = 640
            HW_image_input_size = 1024
            # avg_pool = nn.AvgPool2d(kernel_size=downsample_factor, stride=downsample_factor) if downsample_factor > 1 else nn.Identity()
            # max_pool = nn.MaxPool2d(kernel_size=downsample_factor, stride=downsample_factor) if downsample_factor > 1 else nn.Identity()
            avg_pool = nn.Upsample(scale_factor=upsample_factor, mode='bilinear', align_corners=False) if upsample_factor > 1 else nn.Identity()
            max_pool = nn.Upsample(scale_factor=upsample_factor, mode='nearest') if upsample_factor > 1 else nn.Identity()

            # if preset.pc_name != 'XPS':
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
            cids_monitored = pplv_train.get_cids_monitored(take_num_class=50)

            get_name = lambda k: wrap_name(pplv_train.id2catyinfo[cids_monitored[k]]['name'])

        elif mode == 'cityscapes':

            downsample_factor = 1
            HW_RAW_SIZE = 1024
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
        efficientsam_ti = build_efficient_sam(img_size=HW_image_input_size,
                                              encoder_patch_embed_dim=192,
                                              encoder_num_heads=3,
                                              num_multimask_outputs=1,
                                              checkpoint_state_dict=old_state_dict,
                                              ).to(device)

        efficientsam_ti_custom = efficientsam_ti
        efficientsam_ti_custom.to(device)

        freeze_parameters(efficientsam_ti_custom)
        unfreeze_parameters(efficientsam_ti_custom.mask_decoder)

        # https://github.com/ziqi-jin/finetune-anything/blob/main/config/semantic_seg.yaml
        lr = 5e-4
        num_epochs = 500

        optimizer = torch.optim.NAdam(efficientsam_ti_custom.mask_decoder.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=1, factor=0.95)
        for epoch in trange(num_epochs):
            batch_size = 25

            optimizer.zero_grad()

            if epoch > 0:
                loss_s = []

                efficientsam_ti_custom.train()

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

                    predicted_logits_bx1x3xHxW, predicted_iou_bx1x3 = efficientsam_ti_custom(
                        image_bx3xHxW,
                        input_points_bx1xNx2,
                        input_labels_bx1xN,
                    )
                    cur_X_bx1xHxW = sigmoid(predicted_logits_bx1x3xHxW[:,0,:,:,:])
                    cur_Y_bx1xHxW = Y_bx1xHxW[:, :, :, :].to(device=device)
                    loss = diceloss(cur_X_bx1xHxW,cur_Y_bx1xHxW)
                    loss_s.append(loss.detach().cpu().numpy())

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                print(f'\nEpoch [{epoch + 1}/{num_epochs}], Loss: {np.array(loss_s).mean():.4f} ', end='')

            if True:
                efficientsam_ti_custom.eval()
                with torch.no_grad():
                    col2elems = {
                        'label': [],
                        'seg_miou': [],
                        'seg_fg_iou': [],
                        'seg_bg_iou': []
                    }

                    seg_miou_s, seg_fg_iou_s, seg_bg_iou_s = [], [], []
                    ks = []

                    for bidx, bparts in enumerate(dataloader_valid.get_iterator(batch_size=25, device=torch.device("cpu"), shuffle=True, xrange=trange)):
                        X_bx4xHxW, F_bx2, Y_bx1xHxW, Y_cls_bx1 = bparts
                        X_bx4xHxW = avg_pool(X_bx4xHxW)
                        Y_bx1xHxW = max_pool(Y_bx1xHxW)

                        image_bx3xHxW = X_bx4xHxW[:, :3, :, :].to(device=device)
                        input_points_bx1xNx2 = (F_bx2[:, None, None, [1, 0]] * HW_image_input_size).to(device=device, dtype=torch.int64)
                        b, _ = F_bx2.shape
                        input_labels_bx1xN = torch.ones(b, 1, 1).to(device=device, dtype=torch.int64)

                        predicted_logits_bx1x3xHxW, predicted_iou_bx1x3 = efficientsam_ti_custom(
                            image_bx3xHxW,
                            input_points_bx1xNx2,
                            input_labels_bx1xN,
                        )

                        mask_bx3xHxW = torch.ge(predicted_logits_bx1x3xHxW[:, 0, :, :, :], 0)
                        mask_bx1xHxW = mask_bx3xHxW[:, :1, :, :].to(device=device)
                        Y_bx1xHxW = Y_bx1xHxW[:, :, :, :].to(device=device)

                        # Evaluate segmentation with updated mIoU calculation
                        seg_miou, seg_fg_iou, seg_bg_iou = evaluate_segmentation(mask_bx1xHxW, Y_bx1xHxW)

                        ks.extend(Y_cls_bx1[:, 0].detach().cpu().numpy().tolist())
                        seg_miou_s.extend(seg_miou)
                        seg_fg_iou_s.extend(seg_fg_iou)
                        seg_bg_iou_s.extend(seg_bg_iou)

                    seg_miou_s = np.array(seg_miou_s)
                    seg_fg_iou_s = np.array(seg_fg_iou_s)
                    seg_bg_iou_s = np.array(seg_bg_iou_s)

                    unique_ks = set(ks)
                    ks = np.array(ks)

                    mean_seg_miou = seg_miou_s.mean()
                    mean_seg_fg_iou = seg_fg_iou_s.mean()
                    mean_seg_bg_iou = seg_bg_iou_s.mean()

                    col2elems['label'].append('MEAN')
                    col2elems['seg_miou'].append(mean_seg_miou)
                    col2elems['seg_fg_iou'].append(mean_seg_fg_iou)
                    col2elems['seg_bg_iou'].append(mean_seg_bg_iou)

                    df = pd.DataFrame(col2elems)
                    print("lvis")
                    print(df[df['label'] == 'MEAN'])
                    with open('output_lvis_1024.txt', 'a', encoding='utf-8') as f:
                        print("lvis", file=f)
                        print(df[df['label'] == 'MEAN'], file=f)

                    scheduler.step(mean_seg_miou)


"""
# Server
CUDA_VISIBLE_DEVICES=1 python l_sam/c_train_esam.py

1: only mask decoder
"""