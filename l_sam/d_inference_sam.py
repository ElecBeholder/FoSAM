import os

from segment_anything import SamPredictor, sam_model_registry

import os
import sys

from torch import nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import preset
import os
import platform
import torch.nn.functional as F
import matplotlib
import pandas as pd
from d_model.nn_A3_metrics import evaluate_segmentation

system = platform.system()
if system == "Windows":
    matplotlib.use('TkAgg')  # 有图形界面，使用 TkAgg

os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"

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


sam = sam_model_registry["vit_b"](checkpoint="/root/autodl-tmp/DynamicFocus/l_sam/sam_vit_b_01ec64.pth").to(device=device)
predictor = SamPredictor(sam)

if __name__ == '__main__':

    if True:
        mode = 'cityscapes'
        
        avg_pool = nn.Identity()
        max_pool = nn.Identity()
        HW_RAW_SIZE = 0
        HW_image_input_size = 0

        if mode == 'lvis':
            downsample_factor = 1
            HW_RAW_SIZE = 640
            HW_image_input_size = HW_RAW_SIZE // downsample_factor
            avg_pool = nn.AvgPool2d(kernel_size=downsample_factor, stride=downsample_factor) if downsample_factor > 1 else nn.Identity()
            max_pool = nn.MaxPool2d(kernel_size=downsample_factor, stride=downsample_factor) if downsample_factor > 1 else nn.Identity()

            dataset_train = DatasetLVIS('sp20000', dataset_partition='train')
            dataloader_train = CustomDataLoader(dataset_train, xrange=trange, cache=False)
            dataset_valid = DatasetLVIS('sp4000', dataset_partition='valid')
            dataloader_valid = CustomDataLoader(dataset_valid, xrange=trange, cache=False)

            pplv_train = PreprocessLVIS(dataset_partition='train')
            cids_monitored = pplv_train.get_cids_monitored(take_num_class=50)

            get_name = lambda k: wrap_name(pplv_train.id2catyinfo[cids_monitored[k]]['name'])

        elif mode == 'cityscapes':

            downsample_factor = 1
            HW_RAW_SIZE = 1024
            HW_image_input_size = 1024

            avg_pool = nn.AvgPool2d(kernel_size=downsample_factor, stride=downsample_factor) if downsample_factor > 1 else nn.Identity()
            max_pool = nn.MaxPool2d(kernel_size=downsample_factor, stride=downsample_factor) if downsample_factor > 1 else nn.Identity()

            interp_pool = lambda x: F.interpolate(x, size=(HW_image_input_size, HW_image_input_size), mode='bilinear', align_corners=False)

            # if preset.pc_name != 'XPS':
            dataset_train = DatasetCityScapes('sp1024_10000', dataset_partition='train', cropH=HW_RAW_SIZE, cropW=HW_RAW_SIZE)
            dataloader_train = CustomDataLoader(dataset_train, xrange=trange, cache=False)

            dataset_valid = DatasetCityScapes('sp1024_2000', dataset_partition='valid', cropH=HW_RAW_SIZE, cropW=HW_RAW_SIZE)
            dataloader_valid = CustomDataLoader(dataset_valid, xrange=trange, cache=False)
            # else:
            #     # dataset_train = DatasetCityScapes('sp1024_100', dataset_partition='train', cropH=HW_RAW_SIZE, cropW=HW_RAW_SIZE)
            #     # dataloader_train = CustomDataLoader(dataset_train, xrange=trange, cache=False)

            #     dataset_valid = DatasetCityScapes('sp1024_20', dataset_partition='valid', cropH=HW_RAW_SIZE, cropW=HW_RAW_SIZE)
            #     dataloader_valid = CustomDataLoader(dataset_valid, xrange=trange, cache=False)

            ppcs_train = PreprocessCityscapes(dataset_partition='train')

            get_name = lambda k: wrap_name(ppcs_train.idx2label[k])

        lr = 1e-5
        num_epochs = 1

        for epoch in trange(num_epochs):

            if True:

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
                        """
                        image_1x3xHxW.shape torch.Size([1, 3, 512, 512])
                        input_points_1x1xNx2.shape torch.Size([1, 1, 1, 2])
                        input_labels_1x1xN.shape torch.Size([1, 1, 1])
                        """

                        image_bx3xHxW = X_bx4xHxW[:, :3, :, :]

                        b = F_bx2.shape[0]
                        seg_miou_s_, seg_fg_iou_s_, seg_bg_iou_s_= [], [], []

                        for i in range(b):
                            image_HxWx3 = (image_bx3xHxW[i] * 255).to(dtype=torch.uint8).permute(1, 2, 0).numpy()

                            input_points_Nx2 = (F_bx2[i, None, [1, 0]] * HW_image_input_size).to(dtype=torch.int64).numpy()

                            predictor.set_image(image_HxWx3)

                            mask_1xHxW, _, _ = predictor.predict(point_coords=input_points_Nx2, point_labels=np.ones((1)), multimask_output=False)

                            # print(mask_1xHxW.shape)
                            mask_1x1xHxW = torch.tensor(mask_1xHxW)[None, :, :, :].to(device=device)
                            Y_1x1xHxW = Y_bx1xHxW[i:i + 1, :, :, :].to(device=device)

                            # b, _, _, _ = mask_bx3xHxW.shape
                            # imgs = []
                            # N = 5
                            # for i in range(N):
                            #     pltimg = image_bx3xHxW[i]
                            #
                            #     fy, fx = F_bx2[i]
                            #
                            #     pltimg[:, int(fy * HW_RAW_SIZE), :] = torch.Tensor([1.0, 0, 1.0])[:, None]
                            #     pltimg[:, :, int(fx * HW_RAW_SIZE)] = torch.Tensor([1.0, 0, 1.0])[:, None]
                            #
                            #     pltmask = mask_bx1xHxW[i]
                            #     pltY = Y_bx1xHxW[i]
                            #
                            #     imgs.extend([pltimg, pltmask, pltY])
                            # plt_multi_imgshow(imgs, row_col=(N, 3))
                            #
                            # plt_show()

                            seg_miou_, seg_fg_iou_, seg_bg_iou_ = evaluate_segmentation(mask_1x1xHxW, Y_1x1xHxW)
                            # print(iou_B)
                            seg_miou_s_.extend(seg_miou_)
                            seg_fg_iou_s_.extend(seg_fg_iou_)
                            seg_bg_iou_s_.extend(seg_bg_iou_)

                        ks.extend(Y_cls_bx1[:, 0].detach().cpu().numpy().tolist())
                        seg_miou_s.extend(seg_miou_s_)
                        seg_fg_iou_s.extend(seg_fg_iou_s_)
                        seg_bg_iou_s.extend(seg_bg_iou_s_)

                    seg_miou_s = np.array(seg_miou_s)
                    seg_fg_iou_s = np.array(seg_fg_iou_s)
                    seg_bg_iou_s = np.array(seg_bg_iou_s)

                    mean_seg_miou = seg_miou_s.mean()
                    mean_seg_fg_iou = seg_fg_iou_s.mean()
                    mean_seg_bg_iou = seg_bg_iou_s.mean()

                    col2elems['label'].append('MEAN')
                    col2elems['seg_miou'].append(mean_seg_miou)
                    col2elems['seg_fg_iou'].append(mean_seg_fg_iou)
                    col2elems['seg_bg_iou'].append(mean_seg_bg_iou)

                    df = pd.DataFrame(col2elems)
                    print(df[df['label'] == 'MEAN'])
