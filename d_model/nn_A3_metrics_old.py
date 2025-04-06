import torch
from tensorboard.compat.proto.histogram_pb2 import HistogramProto

from d_model.nn_A0_utils import RAM
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report


def calc_confusion_matrix(preds, targets, num_classes):
    """
    计算每个类别的 TP、FP、TN 和 FN
    :param preds: 模型的预测输出
    :param targets: 真实标签
    :param num_classes: 类别总数
    :return: 每个类别的 TP、FP、TN 和 FN
    """
    confusion_matrix = torch.zeros((num_classes, 4))  # 每个类别的 [TP, FP, FN, TN]

    for cls in range(num_classes):
        # 当前类别的布尔掩码
        pred_class = preds == cls
        true_class = targets == cls

        # 计算 TP, FP, FN, TN
        TP = (pred_class & true_class).sum().item()
        FP = (pred_class & ~true_class).sum().item()
        FN = (~pred_class & true_class).sum().item()
        TN = (~pred_class & ~true_class).sum().item()

        confusion_matrix[cls] = torch.tensor([TP, FP, FN, TN])

    return confusion_matrix


def calc_metrics(confusion_matrix):
    """
    根据混淆矩阵计算 IoU、Precision、Recall、F1 和 Accuracy
    :param confusion_matrix: 每个类别的 [TP, FP, FN, TN]
    :return: 每个类别的 IoU、Precision、Recall、F1 和 Accuracy
    """
    TP = confusion_matrix[:, 0]
    FP = confusion_matrix[:, 1]
    FN = confusion_matrix[:, 2]
    TN = confusion_matrix[:, 3]
    eps = 1e-7
    # 计算 IoU
    iou = TP / (TP + FP + FN + eps)

    # 计算 Precision
    precision = TP / (TP + FP + eps)

    # 计算 Recall
    recall = TP / (TP + FN + eps)

    # 计算 Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN + eps)

    # 计算 F1-score
    f1 = 2 * (precision * recall) / (precision + recall + eps)

    return iou, f1, accuracy, precision, recall


def get_report(y_pred_N: torch.Tensor, y_trgt_N: torch.Tensor, num_classes: int):
    confusion_matrix = calc_confusion_matrix(y_pred_N, y_trgt_N, num_classes)

    iou, precision, recall, f1, accuracy = calc_metrics(confusion_matrix)

    miou = iou.mean().item()
    mean_f1 = f1.mean().item()
    mean_precision = precision.mean().item()
    mean_recall = recall.mean().item()
    mean_accuracy = accuracy.mean().item()

    print(f"Mean IoU (mIoU): {miou:.4f}")
    print(f"Mean F1-score: {mean_f1:.4f}")
    print(f"Mean Precision: {mean_precision:.4f}")
    print(f"Mean Recall: {mean_recall:.4f}")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")


def calc_cfmtx(pred_Bx1xHxW: torch.Tensor, target_Bx1xHxW: torch.Tensor):
    # 将 BxHxW 展平为 B x (H * W)
    B, _, H, W = pred_Bx1xHxW.shape
    threshold = 0.5
    mgpu = RAM()

    mgpu.pred_BxHW = pred_Bx1xHxW.flatten(start_dim=1) >= threshold  # B x (H * W)
    mgpu.target_BxHW = target_Bx1xHxW.flatten(start_dim=1) >= threshold  # B x (H * W)

    # 计算 True Positive, True Negative, False Positive, False Negative
    mgpu.TP_B = torch.sum((mgpu.pred_BxHW) & (mgpu.target_BxHW), dim=1).float()
    mgpu.TN_B = torch.sum((~mgpu.pred_BxHW) & (~mgpu.target_BxHW), dim=1).float()
    mgpu.FP_B = torch.sum((mgpu.pred_BxHW) & (~mgpu.target_BxHW), dim=1).float()
    mgpu.FN_B = torch.sum((~mgpu.pred_BxHW) & (mgpu.target_BxHW), dim=1).float()

    del mgpu.pred_BxHW, mgpu.target_BxHW

    return mgpu.TP_B, mgpu.TN_B, mgpu.FP_B, mgpu.FN_B


def calc_cfmtx_conditions(pred_Bx1xHxW: torch.Tensor, target_Bx1xHxW: torch.Tensor):
    # 将 BxHxW 展平为 B x (H * W)
    B, _, H, W = pred_Bx1xHxW.shape
    threshold = 0.5

    # Flatten and threshold
    pred_BxHW = pred_Bx1xHxW.flatten(start_dim=1) >= threshold  # B x (H * W)
    target_BxHW = target_Bx1xHxW.flatten(start_dim=1) >= threshold  # B x (H * W)

    # Masks for target = 1 and target = 0
    mask_target_1 = target_BxHW
    mask_target_0 = ~target_BxHW

    # Case 1: Only consider target = 1
    TP_target_1 = torch.sum((pred_BxHW & target_BxHW), dim=1).float()
    TN_target_1 = torch.sum((~pred_BxHW & ~target_BxHW), dim=1).float()
    FP_target_1 = torch.sum((pred_BxHW & ~target_BxHW), dim=1).float()
    FN_target_1 = torch.sum((~pred_BxHW & target_BxHW), dim=1).float()

    # Case 2: Only consider target = 0
    TP_target_0 = torch.sum((~pred_BxHW & ~target_BxHW), dim=1).float()
    TN_target_0 = torch.sum((pred_BxHW & target_BxHW), dim=1).float()
    FP_target_0 = torch.sum((~pred_BxHW & target_BxHW), dim=1).float()
    FN_target_0 = torch.sum((pred_BxHW & ~target_BxHW), dim=1).float()

    return {
        "target_1": (TP_target_1, TN_target_1, FP_target_1, FN_target_1),
        "target_0": (TP_target_0, TN_target_0, FP_target_0, FN_target_0),
    }


def evaluate_segmentation(pred_Bx1xHxW: torch.Tensor, target_Bx1xHxW: torch.Tensor):
    """
    Evaluate segmentation and calculate mIoU, foreground IoU, background IoU, and overall IoU.
    """
    # Get shape and compute confusion matrix for different conditions
    cond2tpfn = calc_cfmtx_conditions(pred_Bx1xHxW, target_Bx1xHxW)

    # Extract TP, TN, FP, FN for each condition
    TP_fg, TN_fg, FP_fg, FN_fg = cond2tpfn["target_1"]  # Only target = 1 (foreground)
    TP_bg, TN_bg, FP_bg, FN_bg = cond2tpfn["target_0"]  # Only target = 0 (background)

    B, _, H, W = pred_Bx1xHxW.shape

    eps = 1e-7  # To avoid division by zero

    # Calculate IoUs for each condition
    # Foreground IoU (target = 1)
    iou_fg = TP_fg / (TP_fg + FP_fg + FN_fg + eps)

    # Background IoU (target = 0)
    iou_bg = TP_bg / (TP_bg + FP_bg + FN_bg + eps)

    # Mean IoU (average of foreground and background IoU)
    #
    # threshold = 0.5
    # # Flatten and threshold
    # pred_BxHW = pred_Bx1xHxW.flatten(start_dim=1) >= threshold  # B x (H * W)
    # target_BxHW = target_Bx1xHxW.flatten(start_dim=1) >= threshold  # B x (H * W)

    miou = (iou_fg + iou_bg) / 2

    # Return all results
    return miou.tolist(), iou_fg.tolist(), iou_bg.tolist()


def evaluate_classification(predict_BxK: torch.Tensor, target_Bx1: torch.Tensor, class_num: int):
    # Convert predictions to class indices
    predict_B = torch.argmax(predict_BxK, dim=1).cpu().numpy()  # Convert to numpy array for sklearn
    target_B = target_Bx1.squeeze(dim=1).cpu().numpy()  # Convert to numpy array for sklearn

    # Initialize dictionaries to hold precision, recall, F1, accuracy per class
    precision_per_class = {}
    recall_per_class = {}
    f1_per_class = {}
    accuracy_per_class = {}

    # For each class, calculate precision, recall, F1-score, and accuracy
    for k in range(class_num):
        # For binary classification of class k, treat class k as the positive class
        # and all other classes as negative.
        binary_target = (target_B == k).astype(int)  # Convert to binary labels for class k
        binary_predict = (predict_B == k).astype(int)  # Convert to binary predictions for class k

        # Precision, recall, F1 for class k
        precision_per_class[k] = precision_score(binary_target, binary_predict, zero_division=0)
        recall_per_class[k] = recall_score(binary_target, binary_predict, zero_division=0)
        f1_per_class[k] = f1_score(binary_target, binary_predict, zero_division=0)

        # Accuracy for class k (proportion of correct predictions for both class k and non-k)
        accuracy_per_class[k] = accuracy_score(binary_target, binary_predict)

    return f1_per_class, accuracy_per_class, precision_per_class, recall_per_class


def evaluate_classification_overall(predict_BxK: torch.Tensor, target_Bx1: torch.Tensor, class_num: int):
    predict_B = torch.argmax(predict_BxK, dim=1).cpu().numpy()  # 预测类别
    target_B = target_Bx1.squeeze(dim=1).cpu().numpy()  # 真实标签

    # Calculate overall accuracy
    overall_accuracy = accuracy_score(target_B, predict_B)

    # Generate classification report (contains precision, recall, F1)
    report = classification_report(target_B, predict_B, output_dict=True, zero_division=1)

    # Extract weighted average metrics for overall evaluation
    overall_precision = report['weighted avg']['precision']
    overall_recall = report['weighted avg']['recall']
    overall_f1 = report['weighted avg']['f1-score']

    return overall_f1, overall_accuracy, overall_precision, overall_recall


def evaluate_iou(pred, target, ignore_empty=False, threshold=0.5):
    """
    计算批次中的 mIoU，包括所有像素或大于 0 区域的 IoU。

    Args:
        pred (Tensor): 预测的概率图，形状为 [B, 1, H, W] 或 [B, H, W]。
        target (Tensor): 真实标签，形状为 [B, 1, H, W] 或 [B, H, W]。
        ignore_empty (bool): 是否忽略没有目标的样本，只计算有前景区域的 mIoU。
        threshold (float): 用于将概率图转换为二值图的阈值，默认为 0.5。

    Returns:
        mean_iou (float): 计算得到的平均 IoU。
    """
    # 确保维度一致
    if pred.dim() == 4 and pred.size(1) == 1:
        pred = pred.squeeze(1)  # [B, H, W]
    if target.dim() == 4 and target.size(1) == 1:
        target = target.squeeze(1)  # [B, H, W]

    # 将预测概率转换为二值 mask
    pred = (pred > threshold).float()  # 二值化

    batch_size = pred.size(0)
    ious = []

    for i in range(batch_size):
        pred_i = pred[i]
        target_i = target[i]

        # 计算交集和并集
        intersection = torch.logical_and(pred_i, target_i).sum().float()
        union = torch.logical_or(pred_i, target_i).sum().float()

        # IoU 计算
        if union == 0:
            iou = torch.tensor(1.0 if intersection == 0 else 0.0)
        else:
            iou = intersection / union

        # 如果忽略空样本，跳过没有目标的情况
        if ignore_empty:
            if target_i.sum() > 0:  # 只计算有前景区域的 IoU
                ious.append(iou)
        else:
            ious.append(iou)

    # 计算平均 IoU
    if len(ious) > 0:
        mean_iou = torch.stack(ious).mean().item()
    else:
        mean_iou = 0.0  # 如果所有样本都被忽略

    return mean_iou