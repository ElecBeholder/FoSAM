import os
import sys

from torch.backends import cudnn

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from typing import List

from utility.fctn import save_text

from e_preprocess_scripts.a_preprocess_tools import CustomDataLoader

from d_model.nn_A4_earlystop import EarlyStopMin

from e_preprocess_scripts.b2_preprocess_lvis import DatasetLVIS
import argparse
import shutil
from pprint import pprint

from d_model.nn_B4_Unet6 import UNet6
from d_model.nn_B6_SegNet import SegNet
from d_model.nn_D1_seger_zoom import SegerZoom
from d_model.nn_D2_seger_zoomcropcat import SegerZoomCropCat
from d_model.nn_D3_seger_average import SegerAverage
from d_model.nn_D4_seger_uniform import SegerUniform
from d_model.nn_D5_seger_crop import SegerCrop

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

import preset
from d_model.nn_A0_utils import init_weights_random, calc_model_memsize, RAM, try_gpu
from d_model.nn_A2_loss import DBMSELoss, BMSELoss
from d_model.nn_A3_metrics import evaluate_segmentation
from utility.plot_tools import plt_multi_imgshow


def delete_subfolders_without_all_required_files(root_folder, required_files):
    # Iterate over all items in the root folder
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)

        # Check if the item is a directory (subfolder)
        if os.path.isdir(subfolder_path):
            # Get a list of all files in the current subfolder
            files_in_subfolder = []
            for root, dirs, files in os.walk(subfolder_path):
                files_in_subfolder.extend(files)
                # No need to go deeper into subfolders
                break

                # Check if all required files are present in the current subfolder
            if all(req_file in files_in_subfolder for req_file in required_files):
                contains_all_required_files = True
            else:
                contains_all_required_files = False

            # If the subfolder doesn't contain all of the required files, delete it
            if not contains_all_required_files:
                print(f"Deleting subfolder: {subfolder_path}")
                shutil.rmtree(subfolder_path)


class ModelManager:

    def __init__(self, model, model_name, device, refresh=False):
        self.device = device
        self.model: nn.Module = model.to(device=self.device)
        self.model_name = model_name

        self.fpath_training_records = os.path.join(preset.dpath_training_records, f"{self.model_name}")
        os.makedirs(self.fpath_training_records, exist_ok=True)

        self.recorder = SummaryWriter(self.fpath_training_records)

        self.fpath_work_pt = os.path.join(self.fpath_training_records, f'params.pt')
        self.fpath_view_pth = os.path.join(self.fpath_training_records, f'params.pth')
        self.fpath_msg_json = os.path.join(self.fpath_training_records, f'msg.json')

        loaded = False

        if not refresh:
            print(f'\nload NN d_model state from {self.fpath_work_pt}')
            loaded = self.load_model(self.fpath_work_pt)
        if not loaded:
            print('\nload NN d_model state fail ; init state')
            self.model.apply(init_weights_random)

    def init_model(self):
        self.model.apply(init_weights_random)

    def load_model(self, fpath_model):
        loaded = False
        try:
            self.model.load_state_dict(torch.load(fpath_model))
            self.model.eval()
            loaded = True
        except Exception as err:
            # print(traceback.format_exc())
            pass
        return loaded

    def save_model_state(self, fpath_model, msg=''):
        training = self.model.training
        self.model.train(False)

        # back up in case the model is not fully saved
        fpath_bak = f"{fpath_model}.bak"

        if os.path.exists(fpath_model): os.rename(fpath_model, fpath_bak)
        torch.save(self.model.state_dict(), fpath_model)
        if os.path.exists(fpath_bak): os.remove(fpath_bak)

        self.model.train(training)

        if msg:
            save_text(msg, fpath_model + '.msg.txt')

        print(f'save to {self.fpath_view_pth}')

    def train(self, take_model, target_device, loss_fctn: nn.modules.loss._Loss, optimizer: torch.optim.Optimizer, datasetloader_train: CustomDataLoader, datasetloader_valid: CustomDataLoader, batch_size=50):

        mgpu = RAM()

        optimizer.zero_grad()

        loss_weight_s_train = []
        loss_weight_s_valid = []
        for datasetloader, loss_weight_s, istraining, in [(datasetloader_train, loss_weight_s_train, True), (datasetloader_valid, loss_weight_s_valid, False)]:
            take_model.train(istraining)

            for i, data in enumerate(datasetloader_train.get_iterator(batch_size=batch_size, device=target_device, shuffle=True)):

                mgpu.x_bx4xHxW, mgpu.x_bx2, mgpu.y_bx1xHxW, mgpu.y_bx1 = data
                b, _, H, W = mgpu.x_bx4xHxW.shape

                tensor_train_loss = None

                check_model = take_model
                if isinstance(take_model, nn.DataParallel):
                    check_model = take_model.module

                if isinstance(check_model, SegerZoom) or isinstance(check_model, SegerZoomCropCat):
                    mgpu.y_pred_gs_bx1xHSxWS, mgpu.grid_pred_bxHSxWSx2, mgpu.dmap_pred_ds_bx1xHSxWS, mgpu.x_ds_rgbaf_bxC1xHSxWS, mgpu.x_gs_rgbaf_bxC1xHSxWS = check_model(mgpu.x_bx4xHxW, mgpu.x_bx2)

                    mgpu.y_real_gs_bx1xHSxWS = check_model.downsample_y_gridsp_real_Bx1xHSxWS(mgpu.y_bx1xHxW, mgpu.grid_pred_bxHSxWSx2)
                    mgpu.y_real_ds_bx1xHSxWS = check_model.downsample_y_maxpool_real_Bx1xHSxWS(mgpu.y_bx1xHxW)

                    tensor_train_loss = loss_fctn(mgpu.y_pred_gs_bx1xHSxWS,
                                                  mgpu.y_real_gs_bx1xHSxWS,
                                                  mgpu.dmap_pred_ds_bx1xHSxWS,
                                                  mgpu.y_real_ds_bx1xHSxWS)

                elif isinstance(check_model, SegerAverage) or isinstance(check_model, SegerUniform):

                    mgpu.y_pred_gs_bx1xHSxWS, mgpu.x_ds_rgbaf_bxC1xHSxWS = check_model(mgpu.x_bx4xHxW, mgpu.x_bx2)
                    mgpu.y_real_ds_bx1xHSxWS = check_model.downsample_y_maxpool_real_Bx1xHSxWS(mgpu.y_bx1xHxW)
                    tensor_train_loss = loss_fctn(mgpu.y_pred_gs_bx1xHSxWS,
                                                  mgpu.y_real_ds_bx1xHSxWS)

                elif isinstance(check_model, SegerCrop):
                    mgpu.y_pred_as_BxAx1xHSxWS, mgpu.idxs_crop_BxAx4, mgpu.x_as_rgbaf_BxAxC1xHSxWS = check_model(mgpu.x_bx4xHxW, mgpu.x_bx2)

                    mgpu.y_real_as_BxAx1xHSxWS = check_model.gen_alias(mgpu.y_bx1xHxW, mgpu.idxs_crop_BxAx4, check_model.max_pool_d)
                    tensor_train_loss = loss_fctn(mgpu.y_pred_as_BxAx1xHSxWS,
                                                  mgpu.y_real_as_BxAx1xHSxWS)

                cur_train_loss_item = tensor_train_loss.item()
                tensor_train_loss.backward()
                loss_weight_s.append([cur_train_loss_item, b])

                mgpu.delete_all()

        final_train_loss_item = sum([ls * wt for ls, wt in loss_weight_s_train]) / sum([wt for ls, wt in loss_weight_s_train])
        final_valid_loss_item = sum([ls * wt for ls, wt in loss_weight_s_valid]) / sum([wt for ls, wt in loss_weight_s_valid])

        if np.isnan(final_train_loss_item):
            raise Exception('shut down for nan loss')

        optimizer.step()
        lr_scheduler.step(final_train_loss_item)
        cur_lr = optimizer.param_groups[0]['lr']

        mgpu.gc()

        return cur_lr, final_train_loss_item, final_valid_loss_item

    
    def predict(self, *args):
        mgpu = RAM()

        mgpu.x_bx4xHxW, mgpu.x_bx2 = args
        b, _, H, W = mgpu.x_bx4xHxW.shape
        y_pred_bx1xHxW = None

        if isinstance(self.model, SegerZoom) or isinstance(self.model, SegerZoomCropCat):

            mgpu.y_pred_gs_bx1xHSxWS, mgpu.grid_pred_bxHSxWSx2, mgpu.dmap_pred_ds_bx1xHSxWS, mgpu.x_ds_rgbaf_bxC1xHSxWS, mgpu.x_gs_rgbaf_bxC1xHSxWS = self.model(mgpu.x_bx4xHxW, mgpu.x_bx2)

            del mgpu.dmap_pred_ds_bx1xHSxWS
            del mgpu.x_ds_rgbaf_bxC1xHSxWS
            del mgpu.x_gs_rgbaf_bxC1xHSxWS

            y_pred_bx1xHxW = self.model.output_y_pred_Bx1xHxW(mgpu.y_pred_gs_bx1xHSxWS, mgpu.grid_pred_bxHSxWSx2)

            del mgpu.y_pred_gs_bx1xHSxWS
            del mgpu.grid_pred_bxHSxWSx2
        elif isinstance(self.model, SegerAverage) or isinstance(self.model, SegerUniform):
            mgpu.y_pred_gs_bx1xHSxWS, mgpu.x_ds_rgbaf_bxC1xHSxWS = self.model(mgpu.x_bx4xHxW, mgpu.x_bx2)

            del mgpu.x_ds_rgbaf_bxC1xHSxWS
            y_pred_bx1xHxW = self.model.output_y_pred_Bx1xHxW(mgpu.y_pred_gs_bx1xHSxWS)
            del mgpu.y_pred_gs_bx1xHSxWS

        elif isinstance(self.model, SegerCrop):
            mgpu.y_pred_as_BxAx1xHSxWS, mgpu.idxs_crop_BxAx4, mgpu.x_as_rgbaf_BxAxC1xHSxWS = self.model(mgpu.x_bx4xHxW, mgpu.x_bx2)
            del mgpu.x_as_rgbaf_BxAxC1xHSxWS

            y_pred_bx1xHxW = self.model.gen_unalias(mgpu.y_pred_as_BxAx1xHSxWS, mgpu.idxs_crop_BxAx4)

            del mgpu.y_pred_as_BxAx1xHSxWS
            del mgpu.idxs_crop_BxAx4

        mgpu.gc()
        # print(self.model)

        return y_pred_bx1xHxW

    def get_metrics(self, datasetloader: CustomDataLoader, namekeys, dataset_partition, batch_size=5):

        target_device = self.device

        mgpu = RAM()

        iou_s, f1_s, accuracy_s, precision_s, recall_s = [], [], [], [], []
        for i, data in enumerate(datasetloader.get_iterator(batch_size=batch_size, device=target_device, shuffle=False, xrange=trange)):
            mgpu.x_bx3xHxW, mgpu.x_bx2, mgpu.y_bx1xHxW, mgpu.y_bx1 = data
            mgpu.y_pred_bx1xHxW = self.predict(mgpu.x_bx3xHxW, mgpu.x_bx2)

            iou_B, f1_B, accuracy_B, precision_B, recall_B = evaluate_segmentation(mgpu.y_pred_bx1xHxW, mgpu.y_bx1xHxW)
            del data

            iou_s.extend(iou_B)
            f1_s.extend(f1_B)
            accuracy_s.extend(accuracy_B)
            precision_s.extend(precision_B)
            recall_s.extend(recall_B)

            mgpu.delete_all()
        mgpu.gc()

        labelkey2bidxs = {}
        for bidx, namekey in enumerate(namekeys):
            labelkey = namekey.split('_')[0]
            if labelkey not in labelkey2bidxs:
                labelkey2bidxs[labelkey] = []
            labelkey2bidxs[labelkey].append(bidx)

        col2elems = {'label': [],
                     'iou': [],
                     'f1': [],
                     'accuracy': [],
                     'precision': [],
                     'recall': []
                     }

        for labelkey, bidxs in labelkey2bidxs.items():
            iou = np.mean(np.array(iou_s)[bidxs])
            f1 = np.mean(np.array(f1_s)[bidxs])
            accuracy = np.mean(np.array(accuracy_s)[bidxs])
            precision = np.mean(np.array(precision_s)[bidxs])
            recall = np.mean(np.array(recall_s)[bidxs])

            col2elems['label'].append(labelkey)
            col2elems['iou'].append(iou)
            col2elems['f1'].append(f1)
            col2elems['accuracy'].append(accuracy)
            col2elems['precision'].append(precision)
            col2elems['recall'].append(recall)

        col2elems['label'].append('MEAN')
        col2elems['iou'].append(np.array(col2elems['iou']).mean())
        col2elems['f1'].append(np.array(col2elems['f1']).mean())
        col2elems['accuracy'].append(np.array(col2elems['accuracy']).mean())
        col2elems['precision'].append(np.array(col2elems['precision']).mean())
        col2elems['recall'].append(np.array(col2elems['recall']).mean())

        df = pd.DataFrame(col2elems)

        fpath_metrics = os.path.join(preset.dpath_training_records, f"{self.model_name}", f'{dataset_partition}.metrics.csv')

        df.to_csv(fpath_metrics, index=False)
        print(f"SAVE metrics to {fpath_metrics}")

    def plot_figure(self, datasetloader: CustomDataLoader, fname=f"plot_figure.png"):
        is_training_model = self.model.training
        self.model.eval()
        mgpu = RAM()

        self.model.eval()
        if isinstance(self.model, SegerZoom):
            rows = 6
            mgpu.x_bx4xHxW, mgpu.x_bx2, mgpu.y_bx1xHxW, mgpu.y_bx1 = next(iter(datasetloader.get_iterator(batch_size=rows, device=target_device, shuffle=True)))

            B, _, H, W = mgpu.x_bx4xHxW.shape

            mgpu.y_real_ds_bx1xHSxWS = self.model.downsample_y_avepool_real_Bx1xHDxWD(mgpu.y_bx1xHxW)

            mgpu.y_pred_gs_bx1xHSxWS, mgpu.grid_pred_bxHSxWSx2, mgpu.dmap_pred_ds_bx1xHSxWS, mgpu.x_ds_rgbaf_bxC1xHSxWS, mgpu.x_gs_rgbaf_bxC1xHSxWS = self.model(mgpu.x_bx4xHxW, mgpu.x_bx2)
            mgpu.y_real_gs_bx1xHSxWS = self.model.downsample_y_gridsp_real_Bx1xHSxWS(mgpu.y_bx1xHxW, mgpu.grid_pred_bxHSxWSx2)

            imgs = []
            titles = []

            for b in range(B):
                imgs.extend([
                    torch.cat([mgpu.x_ds_rgbaf_bxC1xHSxWS[b, :3], mgpu.dmap_pred_ds_bx1xHSxWS[b]]),
                    torch.cat([mgpu.x_ds_rgbaf_bxC1xHSxWS[b, :3], mgpu.y_real_ds_bx1xHSxWS[b]]),

                    torch.cat([mgpu.x_gs_rgbaf_bxC1xHSxWS[b, :3], mgpu.y_pred_gs_bx1xHSxWS[b]]),
                    torch.cat([mgpu.x_gs_rgbaf_bxC1xHSxWS[b, :3], mgpu.y_real_gs_bx1xHSxWS[b]])

                ])
                titles.extend([
                    f"dmap_pred_aver_rgba_C1xHSxWS_{b}",
                    f"dmap_real_aver_rgba_C1xHSxWS_{b}",
                    f"sgmt_pred_grid_rgba_C1xHSxWS_{b}",
                    f"sgmt_real_grid_rgba_C1xHSxWS_{b}"
                ])

            plt_multi_imgshow(imgs, titles, row_col=(B, 4))
            fpath_plot = os.path.join(self.fpath_training_records, fname)
            plt.savefig(fpath_plot)
            plt.close('all')
            print(f"save to {fpath_plot}")
        mgpu.delete_all()
        mgpu.gc()
        self.model.train(is_training_model)


def get_sys_kwargs():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Preprocess the Cityscape dataset with specified parameters.")
    parser.add_argument(
        '--Kblur', type=int, default=20, required=False,
        help="One or more Kblur to use (e.g., 2 4 8 ,16,32)."
    )

    parser.add_argument(
        '--Kgrid', type=int, default=20, required=False,
        help="One or more Kgrid to use (e.g., 2 4 8 ,16,32)."
    )

    parser.add_argument(
        '--Module_Loss', type=str, default='SegerZoom_DBMSELoss', required=False,
        help="One or more Module_Loss to use [Module]_[Loss]"
    )

    cbase_seg0 = 4
    nlayer_seg0 = 3
    cbase_seg1 = 16
    nlayer_seg1 = 4
    default_downsample_factor = 4

    if preset.pc_name == preset.PC_NAME_server_H100:
        cbase_seg0 = 4
        nlayer_seg0 = 3
        cbase_seg1 = 64
        nlayer_seg1 = 5
        default_downsample_factor = 4

    parser.add_argument(
        '--downsample_factor', type=int, default=default_downsample_factor, required=False,
        help="The factor by which to downsample the data during processing. This affects the final resolution of the output."

    )

    parser.add_argument(
        '--cbase_seg0', type=int, default=cbase_seg0, required=False,
        help="Base channel count for segment 0."
    )

    parser.add_argument(
        '--cbase_seg1', type=int, default=cbase_seg1, required=False,
        help="Base channel count for segment 1."
    )

    parser.add_argument(
        '--nlayer_seg0', type=int, default=nlayer_seg0, required=False,
        help="Number of layers for segment 0."
    )

    parser.add_argument(
        '--nlayer_seg1', type=int, default=nlayer_seg1, required=False,
        help="Number of layers for segment 1."
    )

    parser.add_argument(
        '--modelname', type=str, default="", required=False,
        help="Name of the model."
    )

    # 添加 train 和 metrics 参数，作为独立的布尔选项
    parser.add_argument(
        '--train', action='store_true', help="Enable training mode."
    )

    parser.add_argument(
        '--metrics', action='store_true', help="Enable metrics mode."
    )

    parser.add_argument(
        '--dataset_marker_train', type=str, default="", required=False, help="marker folder of dataset"
    )

    parser.add_argument(
        '--dataset_marker_valid', type=str, default="", required=False, help="marker folder of dataset"
    )

    # 新增的 batch size 参数
    parser.add_argument(
        '--batch_size_train', type=int, default=10, required=False, help="batch size for training"
    )

    parser.add_argument(
        '--batch_size_valid', type=int, default=1, required=False, help="batch size for validation"
    )

    parser.add_argument(
        '--dataset_class', type=str, default="", required=False, help="dataset name"
    )

    parser.add_argument(
        '--gpu_idxs', type=int, nargs='+', required=True, help="gpu indices, e.g., --gpu_idxs 0 1 2"
    )

    kwargs = parser.parse_args()

    return kwargs


if preset.pc_name == preset.PC_NAME_harry_local:
    delete_subfolders_without_all_required_files(preset.dpath_training_records, ['train.metrics.csv', 'valid.metrics.csv'])

if __name__ == '__main__':
    dp_train = 'train'
    dp_valid = 'valid'

    in_channels = 4
    out_channels = 1
    w1w2 = [1, 1]

    name2loss_cls = {'DBMSELoss': DBMSELoss, 'BMSELoss': BMSELoss}

    base_module = UNet6
    base_module_deformation = UNet6

    Skwargs = get_sys_kwargs()

    batch_size_train = Skwargs.batch_size_train
    batch_size_val = Skwargs.batch_size_valid

    pprint(Skwargs._get_kwargs())

    # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(idx) for idx in Skwargs.gpu_idxs])

    # target_device = try_gpu(gpu_index=Skwargs.gpu_idxs[0])

    target_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not Skwargs.dataset_class:
        Skwargs.dataset_class = 'DatasetLVIS'

    datasetloader_train = None
    datasetloader_valid = None

    if Skwargs.dataset_class == 'DatasetLVIS':
        datasetloader_train = CustomDataLoader(DatasetLVIS(marker=Skwargs.dataset_marker_train, dataset_partition=dp_train), cache=True, xrange=trange)
        datasetloader_valid = CustomDataLoader(DatasetLVIS(marker=Skwargs.dataset_marker_valid, dataset_partition=dp_valid), cache=True, xrange=trange)

    B = len(datasetloader_train.dataset)

    module_name, lossname = Skwargs.Module_Loss.split('_')

    nn_loss_fctn = name2loss_cls[lossname](w1=w1w2[0], w2=w1w2[1])

    nn_module = None
    if module_name == 'SegerZoom':
        nn_module = SegerZoom(base_module_deformation=base_module_deformation, base_module=base_module, in_channels=in_channels,
                              out_channels=out_channels,
                              downsample_factor=Skwargs.downsample_factor,
                              kernel_gridsp=Skwargs.Kblur + 1,
                              kernel_gblur=Skwargs.Kgrid + 1,
                              cbase_seg0=Skwargs.cbase_seg0,
                              cbase_seg=Skwargs.cbase_seg1,
                              nlayer_seg0=Skwargs.nlayer_seg0,
                              nlayer_seg=Skwargs.nlayer_seg1
                              )
        calc_model_memsize(nn_module.gen_seg_0)
    if module_name == 'SegerZoomCropCat':
        nn_module = SegerZoomCropCat(base_module_deformation=base_module_deformation, base_module=base_module, in_channels=in_channels,
                                     out_channels=out_channels,
                                     downsample_factor=Skwargs.downsample_factor,
                                     kernel_gridsp=Skwargs.Kblur + 1,
                                     kernel_gblur=Skwargs.Kgrid + 1,
                                     cbase_seg0=Skwargs.cbase_seg0,
                                     cbase_seg=Skwargs.cbase_seg1,
                                     nlayer_seg0=Skwargs.nlayer_seg0,
                                     nlayer_seg=Skwargs.nlayer_seg1
                                     )
        calc_model_memsize(nn_module.gen_seg_0)

    elif module_name == 'SegerAverage':
        nn_module = SegerAverage(base_module=base_module, in_channels=in_channels,
                                 out_channels=out_channels,
                                 downsample_factor=Skwargs.downsample_factor,
                                 cbase_seg=Skwargs.cbase_seg1,
                                 nlayer_seg=Skwargs.nlayer_seg1)
    elif module_name == 'SegerUniform':
        nn_module = SegerUniform(base_module=base_module, in_channels=in_channels,
                                 out_channels=out_channels,
                                 downsample_factor=Skwargs.downsample_factor,
                                 cbase_seg=Skwargs.cbase_seg1,
                                 nlayer_seg=Skwargs.nlayer_seg1)
    elif module_name == 'SegerCrop':
        nn_module = SegerCrop(base_module=base_module, in_channels=in_channels,
                              out_channels=out_channels,
                              downsample_factor=Skwargs.downsample_factor,
                              cbase_seg=Skwargs.cbase_seg1,
                              nlayer_seg=Skwargs.nlayer_seg1)

    calc_model_memsize(nn_module.gen_seg)
    calc_model_memsize(nn_module)

    modelname = Skwargs.modelname

    if not modelname:
        modelname = ""
        modelname += datetime.now().strftime('D%y%m%d_T%H%M%S')
        modelname += f"_{datasetloader_train.dataset.__class__.__name__.replace('Dataset', '')}"
        modelname += f"_{len(datasetloader_train.dataset)}x{datasetloader_train.dataset.HC}x{datasetloader_train.dataset.WC}"
        modelname += f"_{datasetloader_train.dataset.HC // Skwargs.downsample_factor}x{datasetloader_train.dataset.WC // Skwargs.downsample_factor}"

        if module_name in ['SegerZoom', 'SegerZoomCropCat']:
            modelname += f"_{base_module_deformation.__class__.__name__}"

        modelname += f"_{base_module.__class__.__name__}"

        modelname += f"_{nn_module.__class__.__name__}"

        modelname += f"_{nn_loss_fctn.__class__.__name__}"

        if module_name in ['SegerZoom', 'SegerZoomCropCat']:
            modelname += f"_cbase-{Skwargs.cbase_seg0}-{Skwargs.cbase_seg1}"
            modelname += f"_nlayer-{Skwargs.nlayer_seg0}-{Skwargs.nlayer_seg1}"
            modelname += f"_wloss-{w1w2[0]}-{w1w2[1]}"
        else:
            modelname += f"_cbase-{Skwargs.cbase_seg1}"
            modelname += f"_nlayer-{Skwargs.nlayer_seg1}"

    print(modelname)

    print(target_device)

    nn_module = nn_module.to(target_device)
    if target_device == 'cuda':
        nn_module = nn.DataParallel(nn_module)
        cudnn.benchmark = True

    mm = ModelManager(nn_module, modelname, target_device, refresh=False)

    epoch = 500
    save_per_N_epoch = 10
    plot_per_N_epoch = 100
    eval_per_N_epoch = 100

    if preset.pc_name == preset.PC_NAME_server_H100:
        epoch = 1000
        save_per_N_epoch = 10
        plot_per_N_epoch = 100
        eval_per_N_epoch = 100

    print_per_N_epoch = 5

    init_lr = 0.01
    decay = 0.95

    optimizer = torch.optim.NAdam(mm.model.parameters(), lr=init_lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=decay, patience=5)

    earlystop = EarlyStopMin()
    save_trigger = False

    if Skwargs.train:

        for ep in trange(1, epoch + 1):

            mm.model.train(True)
            cur_lr, cur_loss_train, cur_loss_valid = mm.train(nn_module, target_device, loss_fctn=nn_loss_fctn, optimizer=optimizer, datasetloader_train=datasetloader_train, datasetloader_valid=datasetloader_valid,
                                                              batch_size=batch_size_train)
            mm.model.train(False)

            lr_scheduler.step(cur_loss_train)
            mm.recorder.add_scalar('train_loss', cur_loss_train, ep)

            if ep % save_per_N_epoch == 0:
                save_trigger = True

            if ep % print_per_N_epoch == 0:
                print(f"ep={ep} loss={cur_loss_train:.6f}/{cur_loss_valid:.6f} lr={cur_lr:.6f} trig={int(save_trigger)}")

            if save_trigger:
                if earlystop.check(cur_loss_valid):
                    mm.save_model_state(mm.fpath_work_pt, msg=f'Best Epoch : {ep} loss={cur_loss_train:.6f}/{cur_loss_valid:.6f}')
                    save_trigger = False

            if ep % plot_per_N_epoch == 0:
                mm.plot_figure(datasetloader_train, fname=f"plot_ep{ep}.png")

            if ep % eval_per_N_epoch == 0:
                mm.load_model(fpath_model=mm.fpath_work_pt)
                mm.get_metrics(datasetloader=datasetloader_train, namekeys=datasetloader_train.dataset.get_namekeys(), dataset_partition=datasetloader_train.dataset.dataset_partition, batch_size=batch_size_val)
                mm.get_metrics(datasetloader=datasetloader_valid, namekeys=datasetloader_valid.dataset.get_namekeys(), dataset_partition=datasetloader_valid.dataset.dataset_partition, batch_size=batch_size_val)

        mm.recorder.flush()
        mm.recorder.close()

    elif Skwargs.metrics:
        mm.model.train(False)
        mm.get_metrics(datasetloader=datasetloader_train, namekeys=datasetloader_train.dataset.get_namekeys(), dataset_partition=datasetloader_train.dataset.dataset_partition, batch_size=batch_size_val)
        mm.get_metrics(datasetloader=datasetloader_valid, namekeys=datasetloader_valid.dataset.get_namekeys(), dataset_partition=datasetloader_valid.dataset.dataset_partition, batch_size=batch_size_val)

"""
TODO add GPU manually assign

# on laptop
python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerZoom_DBMSELoss --dataset_marker_train sp50 --dataset_marker_valid sp50 --gpu_idxs 0
python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerZoomCropCat_DBMSELoss --dataset_marker_train sp50 --dataset_marker_valid sp50 --gpu_idxs 0
python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerAverage_BMSELoss --dataset_marker_train sp50 --dataset_marker_valid sp50 --gpu_idxs 0
python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerUniform_BMSELoss --dataset_marker_train sp50 --dataset_marker_valid sp50 --gpu_idxs 0
python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerCrop_BMSELoss --dataset_marker_train sp50 --dataset_marker_valid sp50 --gpu_idxs 0


# on server
python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerZoom_DBMSELoss --dataset_marker_train sp2500 --dataset_marker_valid sp100 --gpu_idxs 3 2 0 --batch_size_train 500
python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerZoomCropCat_DBMSELoss --dataset_marker_train sp2500 --dataset_marker_valid sp100 --gpu_idxs 3 2 0 --batch_size_train 500
python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerAverage_BMSELoss --dataset_marker_train sp2500 --dataset_marker_valid sp100 --gpu_idxs 3 2 0 --batch_size_train 500
python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerUniform_BMSELoss --dataset_marker_train sp2500 --dataset_marker_valid sp100 --gpu_idxs 3 2 0 --batch_size_train 500
python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerCrop_BMSELoss --dataset_marker_train sp2500 --dataset_marker_valid sp100 --gpu_idxs 3 2 0 --batch_size_train 500

"""
