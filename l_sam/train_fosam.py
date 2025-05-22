import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
import random
import pdb
import argparse
from datetime import datetime
import torch
from torch import nn
from d_model.nn_A1_tools import merge_seg_cls, merge_seg_label
from d_model.nn_A1_tools import merge_seg_cls_new, merge_seg_label_new
from l_sam.forveated_sam.foveated_sam_gaze_patch import build_foveated_sam
from d_model.nn_A2_loss import dice_loss
import preset
from d_model.nn_A0_utils import freeze_parameters, unfreeze_parameters
import platform
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from d_model.nn_A3_metrics import evaluate_segmentation, compute_multiclass_iou
from pytorch_toolbelt.losses.dice import DiceLoss
from l_sam.forveated_sam.efficient_sam_encoder_saliency import average_pool, get_merge_map_edge, get_merge_map_object
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from e_preprocess_scripts.a_preprocess_tools import CustomDataLoader
from e_preprocess_scripts.b2_preprocess_lvis import DatasetLVIS, PreprocessLVIS
from e_preprocess_scripts.b3_preprocess_cityscapes import wrap_name, DatasetCityScapes, PreprocessCityscapes
from e_preprocess_scripts.b4_preprocess_ade20k import DatasetADE, PreprocessADE

torch.set_num_threads(5)

system = platform.system()
if system == "Windows":
    matplotlib.use('TkAgg') 

def parse_args():
    parser = argparse.ArgumentParser(description='Train EfficientSAM')
    parser.add_argument('--model_path', type=str, default='efficient_sam_vits.singlemask.pt')
    parser.add_argument('--mode', type=str, default='lvis', choices=['lvis', 'ade', 'cityscapes'])
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of trainingepochs')
    parser.add_argument('--lr1', type=float, default=1e-4, help='learning rate for mask decoder')
    parser.add_argument('--lr2', type=float, default=1e-2, help='learning rate for classifier')
    parser.add_argument('--lr3', type=float, default=1e-2, help='learning rate for segmentation module')
    parser.add_argument('--lambda1', type=float, default=1, help='weight for negtive log likelihood loss')
    parser.add_argument('--lambda2', type=float, default=1, help='weight for classification loss')
    parser.add_argument('--lambda3', type=float, default=1, help='weight for dice loss')
    parser.add_argument('--mintokens', type=int, default=0, help='minimum number of tokens')
    parser.add_argument('--cut_ratio', type=float, default=0.5, help='cut ratio')
    parser.add_argument('--min_tokens', type=int, default=50, help='minimum number of tokens')
    parser.add_argument('--task_name', type=str, default='lvis_default', help='task name')
    return parser.parse_args()
args = parse_args()

writer = SummaryWriter(log_dir=f'../l_sam_experiment/{args.task_name}')

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

class CalculateLoss(nn.Module):
    """
    Calculate overall loss by selecting embeddings based on segmentation mask
    """
    def __init__(self, epsilon=1e-6):
        super(CalculateLoss, self).__init__()
        self.epsilon = epsilon
        self.debug_count = 0
        self.criterion = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.temperature = 0.5

    def forward(self, model, mask, class_label, images=None):
        """
        Args:
            model: DeformSegmentationModule model containing embeddings
            mask: Segmentation mask [B, 1, H, W]
            class_label: Class labels [B]
            images: Original images [B, 3, H, W], optional, for visualization
        Returns:
            loss: Classification loss
        """
        B, _, H, W = mask.shape
        
        embeddings = model.image_encoder.segmentation_module.embeddings
        ds_factor_h = H // embeddings.shape[1]
        ds_factor_w = W // embeddings.shape[2]
        ts = embeddings.shape[1]
        
        downsampled_mask = F.avg_pool2d(mask, kernel_size=(ds_factor_h, ds_factor_w), 
                                       stride=(ds_factor_h, ds_factor_w))  # [B, 1, H_emb, W_emb]
        
        downsampled_mask = downsampled_mask.permute(0, 2, 3, 1)  # [B, H_emb, W_emb, 1]

        predictions, neg_similarity, nll_loss = model.image_encoder.segmentation_module.make_prediction(downsampled_mask)
        
        all_fg_predictions = model.image_encoder.segmentation_module.all_fg_embeddings_predictions
        all_embeddings_predictions = model.image_encoder.segmentation_module.all_embeddings_predictions
        
        sample_losses = []
        for b in range(B):
            sample_preds = all_fg_predictions[b]  # Tensor of shape [N, 51]
            
            expanded_label = class_label[b].expand(sample_preds.size(0))

            sample_loss = self.criterion(sample_preds / self.temperature, expanded_label)
            sample_losses.append(sample_loss)
        
        loss = torch.stack(sample_losses).mean()
        
        loss_nll = nll_loss.mean()
        
        return loss, loss_nll

if __name__ == '__main__':
    mode = args.mode
    calculate_loss = CalculateLoss().to(device)
    diceloss = DiceLoss(mode='multiclass', from_logits=True)

    old_state_dict = torch.load(args.model_path)['model']
    mode = args.mode

    avg_pool = nn.Identity()
    max_pool = nn.Identity()
    HW_RAW_SIZE = 0
    HW_image_input_size = 0
    class_num = 1

    if mode == 'lvis':
        downsample_factor = 1
        HW_RAW_SIZE = 640
        class_num = 51
        HW_image_input_size = 640
        avg_pool = nn.AvgPool2d(kernel_size=downsample_factor, stride=downsample_factor) if downsample_factor > 1 else nn.Identity()
        max_pool = nn.MaxPool2d(kernel_size=downsample_factor, stride=downsample_factor) if downsample_factor > 1 else nn.Identity()
        dataset_train = DatasetLVIS('sp640_60000', dataset_partition='train')
        dataloader_train = CustomDataLoader(dataset_train, xrange=trange, cache=False)

        dataset_valid = DatasetLVIS('sp640_12000', dataset_partition='valid')
        dataloader_valid = CustomDataLoader(dataset_valid, xrange=trange, cache=False)

    elif mode == 'ade':
        downsample_factor = 1
        HW_RAW_SIZE = 640
        class_num = 51
        HW_image_input_size = 640
        avg_pool = nn.AvgPool2d(kernel_size=downsample_factor, stride=downsample_factor) if downsample_factor > 1 else nn.Identity()
        max_pool = nn.MaxPool2d(kernel_size=downsample_factor, stride=downsample_factor) if downsample_factor > 1 else nn.Identity()

        dataset_train = DatasetADE('sp640_60000', dataset_partition='train')
        dataloader_train = CustomDataLoader(dataset_train, xrange=trange, cache=False)

        dataset_valid = DatasetADE('sp640_12000', dataset_partition='valid')
        dataloader_valid = CustomDataLoader(dataset_valid, xrange=trange, cache=False)

    elif mode == 'cityscapes':

        downsample_factor = 1
        HW_RAW_SIZE = 1024
        class_num = 42
        HW_image_input_size = HW_RAW_SIZE // downsample_factor

        avg_pool = nn.AvgPool2d(kernel_size=downsample_factor, stride=downsample_factor) if downsample_factor > 1 else nn.Identity()
        max_pool = nn.MaxPool2d(kernel_size=downsample_factor, stride=downsample_factor) if downsample_factor > 1 else nn.Identity()

        dataset_train = DatasetCityScapes('sp1024_20000', dataset_partition='train', cropH=HW_RAW_SIZE, cropW=HW_RAW_SIZE)
        dataloader_train = CustomDataLoader(dataset_train, xrange=trange, cache=False)

        dataset_valid = DatasetCityScapes('sp1024_4000', dataset_partition='valid', cropH=HW_RAW_SIZE, cropW=HW_RAW_SIZE)
        dataloader_valid = CustomDataLoader(dataset_valid, xrange=trange, cache=False)

    efficientsam_ti = build_foveated_sam(img_size=HW_image_input_size,
                                         encoder_patch_embed_dim=384,
                                         encoder_num_heads=6,
                                         num_multimask_outputs=1,
                                         class_num=class_num,
                                         checkpoint_state_dict=old_state_dict,
                                         cut_ratio=args.cut_ratio,
                                         min_tokens=args.min_tokens).to(device)

    efficientsam_ti_custom = efficientsam_ti
    efficientsam_ti_custom.to(device)

    freeze_parameters(efficientsam_ti_custom)

    unfreeze_parameters(efficientsam_ti_custom.mask_decoder)
    unfreeze_parameters(efficientsam_ti_custom.classifier)
    unfreeze_parameters(efficientsam_ti_custom.image_encoder.segmentation_module)

    lr1 = args.lr1
    lr2 = args.lr2
    lr3 = args.lr3
    num_epochs = args.num_epochs

    for name, param in efficientsam_ti_custom.image_encoder.segmentation_module.named_parameters():
        param.requires_grad_(True)

    optimizer = torch.optim.Adam(
        [
            {"params": efficientsam_ti_custom.mask_decoder.parameters(), "lr": lr1},
            {"params": efficientsam_ti_custom.classifier.parameters(), "lr": lr2},
            {"params": efficientsam_ti_custom.image_encoder.segmentation_module.parameters(), "lr": lr3},
        ]
    )

    batch_size = 8
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30*len(dataloader_train.dataset)//batch_size, T_mult=2, eta_min=5e-5)

    dpath = os.path.join(preset.dpath_training_records, f'{efficientsam_ti_custom.__class__.__name__}_{mode}_{HW_RAW_SIZE}x{HW_RAW_SIZE}_{HW_image_input_size}x{HW_image_input_size}_{datetime.now().strftime("%Y%m%d_%H%M")}')

    os.makedirs(dpath, exist_ok=True)
    global_step = 0
    log_idx = 0
    for epoch in trange(num_epochs):

        optimizer.zero_grad()
        global_step = global_step + 1
        loss_mean = []

        if epoch >= 0:
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

                pred_bx1KxHxW = merge_seg_cls_new(seg_pred_bx1xHxW, cls_pred_bxK)
                label_bxHxW = merge_seg_label_new(cur_Y_bx1xHxW, Y_cls_b)
                mask_bx3xHxW = torch.ge(output_masks_bx1x1xHxW[:, 0, :, :, :], 0)
                mask_bx1xHxW = mask_bx3xHxW[:, :1, :, :].to(device=device)
                Y_bx1xHxW = Y_bx1xHxW[:, :, :, :].to(device=device)
                seg_miou, seg_fg_iou, seg_bg_iou = evaluate_segmentation(mask_bx1xHxW, Y_bx1xHxW)

                embedding_classification_loss, loss_nll = calculate_loss(efficientsam_ti_custom, cur_Y_bx1xHxW, Y_cls_b, image_bx3xHxW)
                
                seg_loss = diceloss(pred_bx1KxHxW, label_bxHxW)
                
                loss = (args.lambda3 * seg_loss +
                        args.lambda2 * embedding_classification_loss +
                        args.lambda1 * loss_nll)

                if torch.isnan(loss):
                    print(f"\nWarning: loss is NaN, setting it to 0")
                    loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
                loss_mean.append(loss.item())
                
                
                if bidx % 50 == 0: 
                    writer.add_scalar('Loss/seg_loss', seg_loss.item(), log_idx)
                    writer.add_scalar('Loss/embedding_cls_loss', embedding_classification_loss.item(), log_idx)
                    writer.add_scalar('Loss/loss_nll', loss_nll.item(), log_idx)
                    writer.add_scalar('lr', scheduler.get_last_lr()[2], log_idx)
                    log_idx += 1
                    print(f"\nBatch {bidx}: Seg Loss: {seg_loss.item():.4f}, Embedding Cls Loss: {embedding_classification_loss.item():.4f}, Loss NLL: {loss_nll.item():.4f}")
                    token_count_list = efficientsam_ti_custom.image_encoder.segmentation_module.token_count_list
                    if len(token_count_list) > 0:
                        print('\ntoken_mean', np.mean(np.concatenate(token_count_list)), 'token_max', np.max(np.concatenate(token_count_list)), 'token_min', np.min(np.concatenate(token_count_list)))
                        efficientsam_ti_custom.image_encoder.segmentation_module.token_count_list = []
                    writer.add_scalar('DynamicWindow/token_mean', np.mean(np.concatenate(token_count_list)), log_idx)
                    writer.add_scalar('training_seg_miou', np.array(seg_fg_iou).mean(), log_idx)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            writer.add_scalar('Loss_cls_train', np.array(loss_mean).mean(), global_step)

            predictions = torch.argmax(torch.cat(cls_predictions_bx1xK_s, dim=0).squeeze(1), dim=1).detach().to('cuda') 
            targets = torch.cat(Y_cls_bx1_s, dim=0).squeeze(1).detach().T.to('cuda')

            mask = (predictions != 0).to('cuda') 

            filtered_predictions = predictions[mask]
            filtered_targets = targets[mask]

            accuracy = np.sum(filtered_predictions.cpu().numpy() == filtered_targets.cpu().numpy()) / len(filtered_predictions)

            print(f"Filtered Accuracy Train: {accuracy}")
            writer.add_scalar('Filtered Accuracy Train', accuracy, global_step)

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
                loss_cls_list = []
                loss_nll_list = []

                for bidx, bparts in enumerate(dataloader_valid.get_iterator(batch_size=8, device=torch.device("cpu"), shuffle=False, xrange=range)):
                    X_bx4xHxW, F_bx2, Y_bx1xHxW, Y_cls_bx1 = bparts
                    X_bx4xHxW = avg_pool(X_bx4xHxW)
                    Y_bx1xHxW = max_pool(Y_bx1xHxW)

                    image_bx3xHxW = X_bx4xHxW[:, :3, :, :].to(device=device)
                    input_points_bx1xNx2 = (F_bx2[:, None, None, [1, 0]] * HW_image_input_size).to(device=device, dtype=torch.int64)
                    b, _ = F_bx2.shape
                    input_labels_bx1xN = torch.ones(b, 1, 1).to(device=device, dtype=torch.int64)

                    output_masks_bx1x1xHxW, cls_predictions_bx1xK, iou_predictions_bx1x1 = efficientsam_ti_custom(
                        image_bx3xHxW,
                        input_points_bx1xNx2,
                        input_labels_bx1xN,
                    )
                    Y_cls_b = Y_cls_bx1.squeeze(1).to(device=device)
                    cur_Y_bx1xHxW = Y_bx1xHxW[:, :, :, :].to(device=device)
                    embedding_classification_loss, loss_nll = calculate_loss(efficientsam_ti_custom, cur_Y_bx1xHxW, Y_cls_b, image_bx3xHxW)
                    loss_cls_list.append(embedding_classification_loss.item())
                    loss_nll_list.append(loss_nll.item())
                    
                    mask_bx3xHxW = torch.ge(output_masks_bx1x1xHxW[:, 0, :, :, :], 0)
                    mask_bx1xHxW = mask_bx3xHxW[:, :1, :, :].to(device=device)
                    Y_bx1xHxW = Y_bx1xHxW[:, :, :, :].to(device=device)

                    seg_miou, seg_fg_iou, seg_bg_iou = evaluate_segmentation(mask_bx1xHxW, Y_bx1xHxW)

                    cls_predictions_bx1xK_s.append(cls_predictions_bx1xK.squeeze(1))
                    Y_cls_bx1_s.append(Y_cls_bx1+1)

                    Y_cls_b = Y_cls_bx1.squeeze(1).to(device=device)

                    seg_pred_bx1xHxW = nn.functional.sigmoid(output_masks_bx1x1xHxW[:, 0, :, :, :]).float() - 0.5
                    cls_pred_bxK = cls_predictions_bx1xK.squeeze(1)
                    pred_bx1KxHxW = merge_seg_cls_new(seg_pred_bx1xHxW, cls_pred_bxK)

                    label_bxHxW = merge_seg_label(Y_bx1xHxW, Y_cls_b)

                    loss_mean.append(diceloss(pred_bx1KxHxW, label_bxHxW).item())

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

                writer.add_scalar('Loss_cls_val', np.array(loss_mean).mean(), global_step)
                writer.add_scalar('fg_miou', mean_seg_fg_iou.item(), global_step)
                writer.add_scalar('Loss/loss_cls_val', np.array(loss_cls_list).mean(), global_step)
                writer.add_scalar('Loss/loss_nll_val', np.array(loss_nll_list).mean(), global_step)

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

                with open(f'{args.task_name}.txt', 'a', encoding='utf-8') as f:
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
                
                #save model
                checkpoint_path = os.path.join('./l_sam/l_sam_experiment', args.task_name + "checkpoint.pt")
                torch.save(efficientsam_ti_custom.state_dict(), checkpoint_path)
                print(f"model saved to: {checkpoint_path}")