from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
import os
import platform
import torch.nn.functional as F
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from graphviz import Digraph
from torch.fx import symbolic_trace

from d_model.nn_A3_metrics import evaluate_segmentation
from utility.plot_tools import plt_imgshow, plt_multi_imgshow, plt_show
from utility.torch_tools import str_tensor_shape
from utility.watch import Watch

system = platform.system()
if system == "Windows":
    matplotlib.use('TkAgg')  # 有图形界面，使用 TkAgg

os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"

import torch
import numpy as np

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


fpath_model = r'D:/h_project/EfficientSAM/weights/efficient_sam_vits.pt.zip'

# Build the EfficientSAM-Ti model.
efficientsam_ti = build_efficient_sam_vitt().to(device)




class ESAMWrap:

    def __init__(self):
        pass

    def predict(self, image_3xHxW: torch.Tensor, point_coords: list):
        image_3xHxW = image_3xHxW.to(device=device)
        input_points = torch.tensor([[[[point_coords[0], point_coords[1]]]]]).to(device=device)
        input_labels = torch.tensor([[[1]]]).to(device=device)

        predicted_logits, predicted_iou = efficientsam_ti(
            image_3xHxW[None, ...],
            input_points,
            input_labels,
        )
        sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
        predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
        predicted_logits = torch.take_along_dim(
            predicted_logits, sorted_ids[..., None, None], dim=2
        )

        mask_3xHxW = torch.ge(predicted_logits[0, 0, :, :, :], 0)
        iou_3 = predicted_iou.flatten()
        logits_3xHxW = predicted_logits.squeeze(dim=[0, 1])
        return mask_3xHxW, iou_3, logits_3xHxW


esw = ESAMWrap()
if __name__ == '__main__':
    pass

    mode = 'cityscapes_sam2'

    downsample_factors = [1, 2, 4, 8, 16, 32]
    if mode == 'lvis':
        dataset = DatasetLVIS('sp100', dataset_partition='valid')
        dataloader = CustomDataLoader(dataset, xrange=trange, cache=False)

        pplv_train = PreprocessLVIS(dataset_partition='train')
        cids_monitored = pplv_train.get_cids_monitored(take_num_class=50)

        get_name = lambda k: wrap_name(pplv_train.id2catyinfo[cids_monitored[k]]['name'])
        HW_RAW_SIZE = 640

    elif mode == 'cityscapes_sam2':
        dataset = DatasetCityScapes('sp40', dataset_partition='valid')
        dataloader = CustomDataLoader(dataset, xrange=trange, cache=False)
        ppcs_train = PreprocessCityscapes(dataset_partition='train')

        get_name = lambda k: wrap_name(ppcs_train.idx2label[k])
        HW_RAW_SIZE = 512

    for downsample_factor in downsample_factors:
        HW_size = HW_RAW_SIZE / downsample_factor

        col2elems = {'label': [],
                     'seg_iou': [],
                     'seg_f1': [],
                     'seg_accuracy': [],
                     'seg_precision': [],
                     'seg_recall': []
                     }

        seg_iou_s, seg_f1_s, seg_accuracy_s, seg_precision_s, seg_recall_s = [], [], [], [], []
        ks = []

        for bidx, bparts in enumerate(dataloader.get_iterator(batch_size=1, device=torch.device("cpu"), shuffle=False, xrange=trange)):
            # print([(bpart.shape, bpart.dtype) for bpart in bparts])
            X_bx4xHxW, F_bx2, Y_bx1xHxW, Y_cls_bx1 = bparts

            X_bx4xHxW = pooling_function(X_bx4xHxW, downsample_factor=downsample_factor, pool_type='avg')
            Y_bx1xHxW = pooling_function(Y_bx1xHxW, downsample_factor=downsample_factor, pool_type='max')

            X_bx3xHxW = X_bx4xHxW[:, :3, :, :]
            image_3xHxW = (X_bx3xHxW[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
            point_1x2 = (F_bx2.detach().cpu().numpy() * HW_size).astype(np.int32)
            label_1 = np.array([1]).astype(np.int32)

            category = Y_cls_bx1[0, 0].item()

            masks, scores, logits = esw.predict(X_bx3xHxW[0], [point_1x2[0, 1], point_1x2[0, 0]])
            # print(masks.shape, scores.shape, logits.shape)
            # print(f"masks : {np.min(masks)}~{np.max(masks)}")
            # print(f"scores : {np.min(scores)}~{np.max(scores)}")
            # print(f"logits : {np.min(logits)}~{np.max(logits)}")

            mask_HxW = masks[0]
            mask_1x1xHxW = torch.tensor(mask_HxW[None, None, :, :]).to(device=device)
            Y_1x1xHxW = Y_bx1xHxW.to(device=device)

            # draw_X_3xHxW = X_bx3xHxW[0]
            # draw_X_3xHxW[:, point_1x2[0, 0], :] = torch.tensor([1., 0., 1.])[:, None]
            # draw_X_3xHxW[:, :, point_1x2[0, 1]] = torch.tensor([1., 0., 1.])[:, None]
            #
            # axes = plt_multi_imgshow([X_bx3xHxW[0], Y_1x1xHxW[0], mask_1x1xHxW[0], logits[0]], row_col=(1, 4))
            #
            # plt.savefig(f"""C:/Users/harry/Dropbox/AGI/CS-GY-997X/DynamicFocus/l_sam/cityscapes_esam/{get_name(category)}_bid{bidx}_{str_tensor_shape(X_bx3xHxW[0])}_{downsample_factor}_{point_1x2[0, 0]}x{point_1x2[0, 1]}.png""")
            # plt.close('all')

            iou_B, f1_B, accuracy_B, precision_B, recall_B = evaluate_segmentation(mask_1x1xHxW, Y_1x1xHxW)

            ks.append(category)
            seg_iou_s.append(iou_B)
            seg_f1_s.append(f1_B)
            seg_accuracy_s.append(accuracy_B)
            seg_precision_s.append(precision_B)
            seg_recall_s.append(recall_B)

            # if bidx > 5:
            #     break

        seg_iou_s = np.array(seg_iou_s)
        seg_f1_s = np.array(seg_f1_s)
        seg_accuracy_s = np.array(seg_accuracy_s)
        seg_precision_s = np.array(seg_precision_s)
        seg_recall_s = np.array(seg_recall_s)

        unique_ks = set(ks)

        ks = np.array(ks)
        col2elems = {'label': [],
                     'seg_iou': [],
                     'seg_f1': [],
                     'seg_accuracy': [],
                     'seg_precision': [],
                     'seg_recall': []
                     }

        for k in unique_ks:
            col2elems['label'].append(get_name(k))
            col2elems['seg_iou'].append(np.mean(seg_iou_s[ks == k]))
            col2elems['seg_f1'].append(np.mean(seg_f1_s[ks == k]))
            col2elems['seg_accuracy'].append(np.mean(seg_accuracy_s[ks == k]))
            col2elems['seg_precision'].append(np.mean(seg_precision_s[ks == k]))
            col2elems['seg_recall'].append(np.mean(seg_recall_s[ks == k]))

        mean_seg_iou = np.array(col2elems['seg_iou']).mean()
        mean_seg_f1 = np.array(col2elems['seg_f1']).mean()
        mean_seg_accuracy = np.array(col2elems['seg_accuracy']).mean()
        mean_seg_precision = np.array(col2elems['seg_precision']).mean()
        mean_seg_recall = np.array(col2elems['seg_recall']).mean()

        col2elems['label'].append('MEAN')
        col2elems['seg_iou'].append(mean_seg_iou)
        col2elems['seg_f1'].append(mean_seg_f1)
        col2elems['seg_accuracy'].append(mean_seg_accuracy)
        col2elems['seg_precision'].append(mean_seg_precision)
        col2elems['seg_recall'].append(mean_seg_recall)

        df = pd.DataFrame(col2elems)
        print(df[df['label'] == 'MEAN'])
        print(f"image size {HW_size * downsample_factor}-{downsample_factor}->{HW_size}")

        #
        #
        # fpath_image = r'D:\b_data_train\data_a_raw\leftImg8bit_demoVideo\leftImg8bit\demoVideo\stuttgart_00\stuttgart_00_000000_000205_leftImg8bit.png'
        # fpath_mask = r'mask.png'
        #
        # # load an image
        # sample_image_np = np.array(Image.open(fpath_image))
        # sample_image_tensor = transforms.ToTensor()(sample_image_np)
        # # Feed a few (x,y) points in the mask as input.
        #
        #
        # # Run inference for both EfficientSAM-Ti and EfficientSAM-S models.
        #
        #
        # mask_3xHxW, iou_3, logits_3xHxW = esw.predict(sample_image_tensor, [1024 + 512, 512])
        # [print(t.shape) for t in [mask_3xHxW, iou_3, logits_3xHxW]]
        # #
        # # masked_image_np = sample_image_np.copy().astype(np.uint8) * mask[:, :, None]
        # # Image.fromarray(masked_image_np).save(fpath_mask)

"""

# Cityscapes

   label   seg_iou    seg_f1  seg_accuracy  seg_precision  seg_recall
34  MEAN  0.335092  0.442878      0.909145       0.845836    0.356151
image size 512.0-1->512.0
   label   seg_iou    seg_f1  seg_accuracy  seg_precision  seg_recall
34  MEAN  0.360061  0.467227      0.931966       0.823555    0.395445
image size 512.0-2->256.0
   label   seg_iou    seg_f1  seg_accuracy  seg_precision  seg_recall
34  MEAN  0.338701  0.442527      0.929469        0.76503    0.364685
image size 512.0-4->128.0
   label   seg_iou    seg_f1  seg_accuracy  seg_precision  seg_recall
34  MEAN  0.354598  0.461133      0.890611       0.749232    0.469546
image size 512.0-8->64.0
   label  seg_iou    seg_f1  seg_accuracy  seg_precision  seg_recall
34  MEAN  0.27072  0.378422        0.7609       0.528706    0.571534
image size 512.0-16->32.0
   label   seg_iou    seg_f1  seg_accuracy  seg_precision  seg_recall
34  MEAN  0.221493  0.310455      0.507123       0.404748    0.754591
image size 512.0-32->16.0
"""

"""
  label   seg_iou    seg_f1  seg_accuracy  seg_precision  seg_recall
5  MEAN  0.407451  0.499082      0.947469       0.645923    0.703654
image size 640.0-1->640.0
  label   seg_iou    seg_f1  seg_accuracy  seg_precision  seg_recall
5  MEAN  0.267427  0.345589       0.93959        0.51599    0.568465
image size 640.0-2->320.0
  label   seg_iou    seg_f1  seg_accuracy  seg_precision  seg_recall
5  MEAN  0.249843  0.327106      0.952512       0.479049    0.548898
image size 640.0-4->160.0
  label   seg_iou    seg_f1  seg_accuracy  seg_precision  seg_recall
5  MEAN  0.181681  0.261002       0.92468       0.388209    0.549369
image size 640.0-8->80.0
  label   seg_iou    seg_f1  seg_accuracy  seg_precision  seg_recall
5  MEAN  0.159025  0.226828      0.839031       0.288422    0.675734
image size 640.0-16->40.0
  label   seg_iou    seg_f1  seg_accuracy  seg_precision  seg_recall
5  MEAN  0.105981  0.159666       0.60025       0.167241     0.74343
image size 640.0-32->20.0

"""
