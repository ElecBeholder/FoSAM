import os
import platform
import torch.nn.functional as F
import matplotlib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from pandas.core.indexes.base import str_t
from tqdm import trange

from d_model.nn_A3_metrics import evaluate_segmentation
from e_preprocess_scripts.a_preprocess_tools import CustomDataLoader
from e_preprocess_scripts.b2_preprocess_lvis import DatasetLVIS, PreprocessLVIS
from e_preprocess_scripts.b3_preprocess_cityscapes import DatasetCityScapes, wrap_name, PreprocessCityscapes
from utility.plot_tools import plt_imgshow, plt_multi_imgshow, plt_show
from utility.torch_tools import str_tensor_shape
from utility.watch import Watch

system = platform.system()
if system == "Windows":
    matplotlib.use('TkAgg')  # 有图形界面，使用 TkAgg

os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"

from utility.xprint import *
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

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
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show(block=True)


sam2_checkpoint = "D:/h_project/sam2/checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "D:/h_project/sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

predictor = SAM2ImagePredictor(sam2_model)


class SAM2Wrap:

    def __init__(self):
        pass

    def predict(self, image: np.ndarray, point_coords: np.ndarray, point_labels: np.ndarray):
        w = Watch()
        predictor.set_image(image)
        # print(f"set_image {w.see_seconds()}")
        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        # print(f"predict {w.see_seconds()}")
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]
        return masks, scores, logits


sw = SAM2Wrap()


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


if __name__ == '__main__':
    pass

    mode = 'lvis'

    downsample_factors = [1, 2, 4, 8, 16, 32]
    if mode == 'lvis':
        dataset = DatasetLVIS('sp20', dataset_partition='valid')
        dataloader = CustomDataLoader(dataset, xrange=trange, cache=False)

        pplv_train = PreprocessLVIS(dataset_partition='train')
        cids_monitored = pplv_train.get_cids_monitored(take_num_class=5)

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

            # print(image_3xHxW.shape, image_3xHxW.dtype)
            # print(point_1x2.shape, point_1x2.dtype)
            # print(label_1.shape, label_1.dtype)

            """
            image.shape,image.dtype
            Out[4]: ((176, 176, 3), dtype('uint8'))
            input_point.shape,input_point.dtype
            Out[5]: ((1, 2), dtype('int32'))
            input_label.shape,input_label.dtype
            Out[6]: ((1,), dtype('int32'))
            """
            masks, scores, logits = sw.predict(image_3xHxW, np.flip(point_1x2, axis=-1).copy(), label_1)
            # print(masks.shape, scores.shape, logits.shape)
            # print(f"masks : {np.min(masks)}~{np.max(masks)}")
            # print(f"scores : {np.min(scores)}~{np.max(scores)}")
            # print(f"logits : {np.min(logits)}~{np.max(logits)}")

            score = scores[0]
            mask_HxW = masks[0]
            mask_1x1xHxW = torch.tensor(mask_HxW[None, None, :, :]).to(device=device)
            Y_1x1xHxW = Y_bx1xHxW.to(device=device)

            # draw_X_3xHxW = X_bx3xHxW[0]
            # draw_X_3xHxW[:, point_1x2[0, 0], :] = torch.tensor([1., 0., 1.])[:, None]
            # draw_X_3xHxW[:, :, point_1x2[0, 1]] = torch.tensor([1., 0., 1.])[:, None]
            #
            # axes = plt_multi_imgshow([X_bx3xHxW[0], Y_1x1xHxW[0], mask_1x1xHxW[0], logits[0]], row_col=(1, 4))
            #
            # plt.savefig(f"""C:/Users/harry/Dropbox/AGI/CS-GY-997X/DynamicFocus/l_sam/lvis/{get_name(category)}_bid{bidx}_{str_tensor_shape(X_bx3xHxW[0])}_{point_1x2[0, 0]}x{point_1x2[0, 1]}.png""")
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
        # image = Image.open(r'D:\b_data_train\data_a_raw\leftImg8bit_demoVideo\leftImg8bit\demoVideo\stuttgart_00\stuttgart_00_000000_000205_leftImg8bit.png')
        # image = Image.open(r'D:\b_data_train\data_a_raw\projectaria_tools_adt_data\Apartment_release_clean_seq138_M1292\imgs\fidx_0033.X.png')
        # image = np.array(image.convert("RGB"))
        #
        # input_point = np.array([[52, 122]])
        # input_label = np.array([1])
        #
        # masks, scores, logits = sw.predict(image, input_point, input_label)
        #
        # img_CxHxW = torch.tensor(image).permute(2, 0, 1).to(dtype=torch.float32, device=device) / 255.0
        # mask_CxHxW = torch.tensor(masks)
        # score_C = torch.tensor(scores)
        # logit_CxHxW = torch.tensor(logits)
        #
        # print(score_C)
        # plt_imgshow(img_CxHxW)
        # plt_multi_imgshow([mask_CxHxW[0], mask_CxHxW[1], mask_CxHxW[2], logit_CxHxW[0], logit_CxHxW[1], logit_CxHxW[2]], row_col=(2, 3))
        # plt_show()
