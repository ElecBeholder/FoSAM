import os
os.environ["NCCL_BLOCKING_WAIT"] = "1"  


import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from d_model.nn_C1_mobilenetv2 import MobileNetV2
from d_model.nn_C2_efficientnet import CustomShuffleNet
from typing import List, Optional
from utility.fctn import save_text

from e_preprocess_scripts.a_preprocess_tools_parallel import CustomDataLoader

from d_model.nn_A4_earlystop import EarlyStopMin, EarlyStopMax

from e_preprocess_scripts.b2_preprocess_lvis import DatasetLVIS
# from e_preprocess_scripts.b3_preprocess_cityscapes import DatasetCityScapes
from e_preprocess_scripts.b4 import DatasetCityScapes
import argparse
import shutil
from pprint import pprint

from d_model.nn_B4_Unet6 import UNet6
from d_model.nn_B7_FovSimModule import FovSimModule
from d_model.nn_B6_SegNet import SegNet
from d_model.nn_D1_seger_zoom import SegerZoom
from d_model.nn_D2_seger_zoomcropcat import SegerZoomCropCat
from d_model.nn_D3_seger_average import SegerAverage
from d_model.nn_D4_seger_uniform import SegerUniform
from d_model.nn_D5_seger_crop import SegerCrop
from d_model.nn_D6_serger_zoom_withembed import SegerZoomEmbed
# from d_model.nn_D8_seger_segvit import SegerZoomSeg
from d_model.nn_D8_seger_segvit_3 import SegerZoomSeg
# from d_model.nn_D9_seger_deeplab import SegerZoomDeep
from d_model.nn_D1_seger_original import SegerZoomDeep
from d_model.nn_D7_serger_zoom_sam import SegerZoomSAM
from d_model.nn_D10_seger_psp import SegerZoomPSP

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

import preset
from d_model.nn_A0_utils import init_weights_random, calc_model_memsize, RAM, try_gpu, init_weights_zero
from d_model.nn_A2_loss import BMSELoss, BCOSIMLoss, WCELoss, WCE_TVLoss
from d_model.nn_A3_metrics import evaluate_segmentation, evaluate_classification
from utility.plot_tools import plt_multi_imgshow, plt_show

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = '1254'

    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

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

        self.model = model
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
            if 'SegerZoomSeg' not in model_name and 'SegerZoomDeep' not in model_name:
                self.init_model()

    def init_model(self):
        self.model.module.gen_seg.apply(init_weights_random)
        # self.model.module.gen_cls.apply(init_weights_random)

        take_model = self.model
        check_model = take_model
        if isinstance(take_model, nn.DataParallel):
            check_model = take_model.module

        if isinstance(check_model, SegerZoom) or isinstance(check_model, SegerZoomCropCat):
            self.model.gen_seg_0.apply(init_weights_zero)

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
        torch.save(self.model.module.state_dict(), fpath_model)
        if os.path.exists(fpath_bak): os.remove(fpath_bak)

        self.model.train(training)

        if msg:
            save_text(msg, fpath_model + '.msg.txt')

        print(f'save to {self.fpath_view_pth}')


    def train(self, take_model: nn.Module, target_device: str, loss_fctn_seg: Optional[nn.modules.loss._Loss], loss_fctn_cls: Optional[nn.modules.loss._Loss], wloss: List, optimizer: torch.optim.Optimizer,
              datasetloader_train: CustomDataLoader,
              batch_size=50,
              lr_scheduler = None):
        optimizer.zero_grad()
        loss_weight_s_train = []
        take_model = self.model

        for i, data in enumerate(datasetloader_train.get_iterator(batch_size=batch_size, device=target_device, shuffle=True)):
        
            x_bx4xHxW, y_bx1xHxW= data
            b, _, H, W = x_bx4xHxW.shape
            tensor_train_loss = None

            check_model = take_model
            while isinstance(check_model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                check_model = check_model.module

            if isinstance(check_model, SegerZoom) or isinstance(check_model, SegerZoomSeg) or isinstance(check_model, SegerZoomPSP) or isinstance(check_model, SegerZoomDeep) or isinstance(check_model, SegerZoomSAM):
                y_pred_gs_bx1xHSxWS, grid_pred_bxHSxWSx2, out0_Bx1xHDxWD, x_ds_rgbaff_BxC2xHDxWD, x_gs_rgbaff_BxC2xHSxWS = take_model(x_bx4xHxW)

                y_real_gs_bx1xHSxWS = check_model.downsample_y_gridsp_real_Bx1xHSxWS(y_bx1xHxW, grid_pred_bxHSxWSx2)
                y_real_ds_bx1xHDxWD = check_model.downsample_y_avepool_real_Bx1xHDxWD(y_bx1xHxW)[check_model.downsample_y_avepool_real_Bx1xHDxWD(y_bx1xHxW)>1] = 1

                w1, w2, w3 = wloss

                tensor_train_loss = (
                        w1 * loss_fctn_seg(out0_Bx1xHDxWD, y_real_ds_bx1xHDxWD.squeeze(1).to(torch.int64)) 
                        + w2 * loss_fctn_seg(y_pred_gs_bx1xHSxWS, y_real_gs_bx1xHSxWS.squeeze(1).to(torch.int64))
                )

            if isinstance(check_model, SegerUniform):
                y_pred_ds_bx1xHSxWS, x_ds_rgbf_BxC1xHSxWS = take_model(x_bx4xHxW)

                y_real_gs_bx1xHSxWS = check_model.downsample_y_maxpool_real_Bx1xHSxWS(y_bx1xHxW)

                w1, w2, w3 = wloss
                tensor_train_loss = (
                        + w2 * loss_fctn_seg(y_pred_ds_bx1xHSxWS, y_real_gs_bx1xHSxWS.squeeze(1).to(torch.int64))
                )

            cur_train_loss_item = tensor_train_loss.item()
            tensor_train_loss.backward()

            loss_weight_s_train.append([cur_train_loss_item, b])


        final_train_loss_item = sum([ls * wt for ls, wt in loss_weight_s_train]) / sum([wt for ls, wt in loss_weight_s_train])

        if np.isnan(final_train_loss_item):
            raise Exception('shut down for nan loss')

        optimizer.step()
        lr_scheduler.step(final_train_loss_item)
        cur_lr = optimizer.param_groups[0]['lr']

        return cur_lr, final_train_loss_item

    def predict(self, *args):

        mgpu = RAM()

        mgpu.x_bx4xHxW, = args
        b, _, H, W = mgpu.x_bx4xHxW.shape
        mgpu.y_pred_bx1xHxW = None

        take_model = self.model
        check_model = take_model
        while isinstance(check_model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            check_model = check_model.module

        check_model.eval()
        take_model.eval()

        with torch.no_grad():
            if isinstance(check_model, SegerZoom) or isinstance(check_model, SegerZoomDeep) or isinstance(check_model, SegerZoomPSP) or isinstance(check_model, SegerZoomSeg) or isinstance(check_model, SegerZoomSAM):

                mgpu.y_pred_gs_bx1xHSxWS, mgpu.grid_pred_bxHSxWSx2, mgpu.out0_Bx1xHDxWD, mgpu.x_ds_rgbaff_bxC1xHDxWD, mgpu.x_gs_rgbaff_bxC1xHSxWS = take_model(mgpu.x_bx4xHxW)
                del mgpu.dmap_pred_ds_bx1xHDxWD
                del mgpu.x_ds_rgbaff_bxC1xHDxWD
                del mgpu.x_gs_rgbaff_bxC1xHSxWS

                mgpu.y_pred_bx1xHxW = check_model.output_y_pred_Bx1xHxW(mgpu.y_pred_gs_bx1xHSxWS, mgpu.grid_pred_bxHSxWSx2, mgpu.x_bx4xHxW)

                del mgpu.y_pred_gs_bx1xHSxWS
                del mgpu.grid_pred_bxHSxWSx2
            else:
                mgpu.y_pred_ds_bx1xHSxWS, _ = take_model(mgpu.x_bx4xHxW)
                mgpu.y_pred_bx1xHxW = check_model.output_y_pred_Bx1xHxW(mgpu.y_pred_ds_bx1xHSxWS)

        return mgpu.y_pred_bx1xHxW
    
    def calculate_iou_per_class(self, pred, target, num_classes=20):
        ious = []
        pred = pred.view(-1) 
        target = target.view(-1)  

        for cls in range(num_classes):
            pred_cls = (pred == cls)
            target_cls = (target == cls)

            intersection = torch.sum((pred_cls & target_cls).float())
            union = torch.sum((pred_cls | target_cls).float())

            if union == 0:
                ious.append(float('nan')) 
            else:
                ious.append((intersection / union).item())

        return ious
    def mean_iou(self, output, target, num_classes=20):
        pred_classes = torch.argmax(output, dim=1)  
        batch_ious = []
        for i in range(pred_classes.shape[0]):  # 遍历 batch
            ious = self.calculate_iou_per_class(pred_classes[i], target[i], num_classes)
            valid_ious = [iou for iou in ious if not np.isnan(iou)]  # 过滤掉 NaN
            if valid_ious:
                batch_ious.append(np.mean(valid_ious))

        return np.mean(batch_ious) if batch_ious else float('nan')


    def get_metrics(self, class_num, datasetloader: CustomDataLoader, namekeys, dataset_partition, batch_size=5, show=False, save=False,
                    rank=0, world_size=1,stage=0):

        target_device = self.device
        xrange = trange if show else range
        mgpu = RAM()
        namekeys = namekeys[rank::world_size]

        seg_iou_s= []
        y_pred_BxK = []
        y_Bx1 = []

        for i, data in enumerate(datasetloader.get_iterator(batch_size=batch_size, device=target_device, shuffle=False, xrange=xrange)):
            mgpu.x_bx4xHxW, mgpu.y_bx1xHxW = data

            mgpu.y_pred_bx1xHxW = self.predict(mgpu.x_bx4xHxW)
            batch_miou = self.mean_iou(mgpu.y_pred_bx1xHxW, mgpu.y_bx1xHxW)
            seg_iou_s.append(batch_miou)
        del mgpu.x_bx4xHxW
        del mgpu.y_bx1xHxW
        print(f'in rank {rank} miou is {seg_iou_s}')
        return np.array(seg_iou_s).mean()


    def plot_figure(self, datasetloader: CustomDataLoader, fname=f"plot_figure.png", rank=0):
        is_training_model = self.model.training
        self.model.eval()
        mgpu = RAM()

        self.model.eval()

        check_model = self.model
        while isinstance(check_model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            check_model = check_model.module

        target_device = self.model.device
        with torch.no_grad():
            if isinstance(check_model, SegerZoom) or isinstance(check_model, SegerZoomDeep) or isinstance(check_model, SegerZoomSeg) or isinstance(check_model, SegerZoomDeep) :
                rows = 5
                mgpu.x_bx4xHxW,  mgpu.y_bx1xHxW = next(iter(datasetloader.get_iterator(batch_size=rows, device=target_device, shuffle=True)))

                B, _, H, W = mgpu.x_bx4xHxW.shape
                rows = B

                mgpu.y_real_ds_bx1xHSxWS = check_model.downsample_y_avepool_real_Bx1xHDxWD(mgpu.y_bx1xHxW)

                mgpu.y_pred_gs_bx1xHSxWS, mgpu.grid_pred_bxHSxWSx2, mgpu.out0_Bx1xHDxWD, mgpu.x_ds_rgbaff_bxC1xHDxWD, mgpu.x_gs_rgbaff_bxC1xHSxWS = check_model(mgpu.x_bx4xHxW)

                mgpu.y_real_gs_bx1xHSxWS = check_model.downsample_y_gridsp_real_Bx1xHSxWS(mgpu.y_bx1xHxW, mgpu.grid_pred_bxHSxWSx2)

                mgpu.y_pred_bx1xHxW = (check_model.output_y_pred_Bx1xHxW(mgpu.y_pred_gs_bx1xHSxWS, mgpu.grid_pred_bxHSxWSx2, mgpu.x_bx4xHxW) >=0.5 ).float()

                imgs = []
                titles = []

                for b in range(B):
                    imgs.extend([

                        mgpu.x_bx4xHxW[b, :3],
                        torch.cat([mgpu.x_ds_rgbaff_bxC1xHDxWD[b, :3], mgpu.dmap_pred_ds_bx1xHDxWD[b]]),
                        torch.cat([mgpu.x_ds_rgbaff_bxC1xHDxWD[b, :3], mgpu.y_real_ds_bx1xHSxWS[b]]),

                        mgpu.x_gs_rgbaff_bxC1xHSxWS[b, :3],

                        torch.cat([mgpu.x_gs_rgbaff_bxC1xHSxWS[b, :3], mgpu.y_pred_gs_bx1xHSxWS[b]]),
                        torch.cat([mgpu.x_gs_rgbaff_bxC1xHSxWS[b, :3], mgpu.y_real_gs_bx1xHSxWS[b]]),
                        mgpu.y_pred_bx1xHxW[b],
                        mgpu.y_bx1xHxW[b]

                    ])
                    titles.extend([
                        f"orig_rgba_4xHxW_{b}",
                        f"dmap_aver_rgba_pred_4xHDxWD_{b}",
                        f"dmap_aver_rgba_trgt_4xHDxWD_{b}",

                        f"sgmt_grid_rgba_4xHSxWS_{b}",

                        f"sgmt_grid_rgba_pred_4xHSxWS_{b}",
                        f"sgmt_grid_rgba_trgt_4xHSxWS__{b}",

                        f"fina_pred_back_rgba_1xHxW_{b}",
                        f"fina_real_back_rgba_1xHxW_{b}"
                    ])

                plt_multi_imgshow(imgs, titles, row_col=(B, 8))
                fname = fname[:-4]+str(rank)+fname[-4:]
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

    cbase_seg0 = 16
    nlayer_seg0 = 4
    cbase_seg1 = 16
    nlayer_seg1 = 4

    if preset.pc_name == preset.PC_NAME_server_H100 or preset.pc_name == preset.PC_NAME_server_4090:
        cbase_seg0 = 16
        nlayer_seg0 = 4
        cbase_seg1 = 64
        nlayer_seg1 = 6

    parser.add_argument('--Kblur', type=int, default=32, required=False, help=" (e.g., 2 4 8 ,16,32).")
    parser.add_argument('--Kprio', type=int, default=32, required=False, help=" (e.g., 2 4 8 ,16,32).")
    parser.add_argument('--Kgrid', type=int, default=32, required=False, help=" (e.g., 2 4 8 ,16,32).")

    parser.add_argument('--downsample_factor', type=int, default=4, required=False, help="downsample factor")
    parser.add_argument('--downsample_factor_deformation', type=int, default=2, required=False, help="downsample factor deformation")

    parser.add_argument('--cbase_seg0', type=int, default=cbase_seg0, required=False, help="Base channel count for segment 0.")
    parser.add_argument('--cbase_seg1', type=int, default=cbase_seg1, required=False, help="Base channel count for segment 1.")
    parser.add_argument('--nlayer_seg0', type=int, default=nlayer_seg0, required=False, help="Number of layers for segment 0.")
    parser.add_argument('--nlayer_seg1', type=int, default=nlayer_seg1, required=False, help="Number of layers for segment 1.")
    parser.add_argument('--param_priori', type=bool, default=False, required=False, help="Enable priori")
    parser.add_argument('--param_square_focus', type=bool, default=False, required=False, help="Enable square_focus")

    parser.add_argument('--dataset_marker_train', type=str, default="", required=False, help="marker folder of dataset")
    parser.add_argument('--dataset_marker_valid', type=str, default="", required=False, help="marker folder of dataset")
    parser.add_argument('--batch_size_train', type=int, default=25, required=False, help="batch size for training")
    parser.add_argument('--batch_size_valid', type=int, default=10, required=False, help="batch size for validation")
    parser.add_argument('--class_num', type=int, default=5, required=False, help='Class Number')
    parser.add_argument('--finetune_module', type=str, default='seg_0', required=False, help='Finetune Module')
    parser.add_argument('--ckpt_path', type=str, default=None, required=False, help='Checkpoint Path')

    parser.add_argument('--Module_Loss', type=str, default='SegerZoom_DBMSELoss', required=False, help="One or more Module_Loss to use [Module]_[Loss]")
    parser.add_argument('--modelname', type=str, default="", required=False, help="Name of the model.")
    parser.add_argument('--dataset_class', type=str, default="", required=False, help="dataset name")

    parser.add_argument('--train', action='store_true', help="Enable training mode.")
    parser.add_argument('--metrics', action='store_true', help="Enable metrics mode.")
    parser.add_argument('--clean_records', action='store_true', help='delete subfolders without all required files')

    kwargs = parser.parse_args()

    return kwargs

def main(rank, world_size):
    ddp_setup(rank, world_size)

    dp_train = 'train'
    dp_valid = 'valid'

    in_channels = 3
    out_channels = 20
    wloss = [5, 0, 0]

    base_module_deformation = UNet6
    base_module = UNet6
    classify_module = MobileNetV2
    # classify_module = CustomShuffleNet

    Skwargs = get_sys_kwargs()
    class_num = Skwargs.class_num

    if Skwargs.clean_records:
        delete_subfolders_without_all_required_files(preset.dpath_training_records, ['train.metrics.csv', 'valid.metrics.csv'])

    batch_size_train = Skwargs.batch_size_train
    batch_size_val = Skwargs.batch_size_valid

    pprint(Skwargs._get_kwargs())

    # target_device = try_gpu(gpu_index=Skwargs.gpu_idxs[0])

    target_device = rank
    torch.cuda.set_device(rank)


    if not Skwargs.dataset_class:
        Skwargs.dataset_class = 'DatasetLVIS'

    datasetloader_train = None
    datasetloader_valid = None

    from torch.utils.data import DistributedSampler

    if Skwargs.dataset_class == 'DatasetLVIS':
        train_sampler = DistributedSampler(DatasetLVIS(marker=Skwargs.dataset_marker_train, dataset_partition=dp_train), num_replicas=world_size, rank=rank)
        valid_sampler = DistributedSampler(DatasetLVIS(marker=Skwargs.dataset_marker_valid, dataset_partition=dp_valid), num_replicas=world_size, rank=rank)


        datasetloader_train = CustomDataLoader(DatasetLVIS(marker=Skwargs.dataset_marker_train, dataset_partition=dp_train), sampler=train_sampler, cache=True, xrange=trange)
        datasetloader_valid = CustomDataLoader(DatasetLVIS(marker=Skwargs.dataset_marker_valid, dataset_partition=dp_valid), sampler=valid_sampler, cache=True, xrange=trange)
    else:
        train_sampler = DistributedSampler(DatasetCityScapes(marker=Skwargs.dataset_marker_train, dataset_partition=dp_train), num_replicas=world_size, rank=rank)
        valid_sampler = DistributedSampler(DatasetCityScapes(marker=Skwargs.dataset_marker_valid, dataset_partition=dp_valid), num_replicas=world_size, rank=rank)


        datasetloader_train = CustomDataLoader(DatasetCityScapes(marker=Skwargs.dataset_marker_train, dataset_partition=dp_train), sampler=train_sampler, cache=True, xrange=trange)
        datasetloader_valid = CustomDataLoader(DatasetCityScapes(marker=Skwargs.dataset_marker_valid, dataset_partition=dp_valid), sampler=valid_sampler, cache=True, xrange=trange)

    module_name, lossname = Skwargs.Module_Loss.split('_')

    if lossname == 'BMSELoss':
        nn_loss_fctn_seg = BMSELoss()
    elif lossname == 'WCELoss':
        nn_loss_fctn_seg = WCE_TVLoss(weight_tv=0.2, weight_ce=0.8)
    nn_loss_fctn_seg = nn.CrossEntropyLoss()
    # nn_loss_fctn_cls = BCOSIMLoss(class_num=class_num)
    nn_loss_fctn_cls = WCELoss(class_num=class_num)

    nn_module = None
    print(module_name)
    if module_name == 'SegerZoom':
        nn_module = SegerZoom(base_module_deformation=base_module_deformation,
                              base_module=base_module,
                              classify_module=classify_module,
                              in_channels=in_channels,
                              out_channels=out_channels,
                              class_num=class_num,
                              downsample_factor=Skwargs.downsample_factor,
                              downsample_factor_deformation=Skwargs.downsample_factor_deformation,
                              kernel_gridsp=Skwargs.Kblur + 1,
                              kernel_gblur=Skwargs.Kgrid + 1,
                              cbase_seg0=Skwargs.cbase_seg0,
                              cbase_seg=Skwargs.cbase_seg1,
                              nlayer_seg0=Skwargs.nlayer_seg0,
                              nlayer_seg=Skwargs.nlayer_seg1,
                              priori=Skwargs.param_priori,
                              kernel_priori=Skwargs.Kprio + 1,
                              square_focus=Skwargs.param_square_focus,
                              )
        calc_model_memsize(nn_module.gen_seg_0, label='gen_seg_0')
    elif module_name=='SegerZoomSeg':
        nn_module = SegerZoomSeg(base_module_deformation=base_module_deformation,
                        base_module=base_module,
                        classify_module=classify_module,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        class_num=class_num,
                        downsample_factor=Skwargs.downsample_factor,
                        downsample_factor_deformation=Skwargs.downsample_factor_deformation,
                        kernel_gridsp=Skwargs.Kblur + 1,
                        kernel_gblur=Skwargs.Kgrid + 1,
                        cbase_seg0=Skwargs.cbase_seg0,
                        cbase_seg=Skwargs.cbase_seg1,
                        nlayer_seg0=Skwargs.nlayer_seg0,
                        nlayer_seg=Skwargs.nlayer_seg1,
                        priori=Skwargs.param_priori,
                        kernel_priori=Skwargs.Kprio + 1,
                        square_focus=Skwargs.param_square_focus,
                        )
    elif module_name=='SegerZoomPSP':
        nn_module = SegerZoomSeg(base_module_deformation=base_module_deformation,
                        base_module=base_module,
                        classify_module=classify_module,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        class_num=class_num,
                        downsample_factor=Skwargs.downsample_factor,
                        downsample_factor_deformation=Skwargs.downsample_factor_deformation,
                        kernel_gridsp=Skwargs.Kblur + 1,
                        kernel_gblur=Skwargs.Kgrid + 1,
                        cbase_seg0=Skwargs.cbase_seg0,
                        cbase_seg=Skwargs.cbase_seg1,
                        nlayer_seg0=Skwargs.nlayer_seg0,
                        nlayer_seg=Skwargs.nlayer_seg1,
                        priori=Skwargs.param_priori,
                        kernel_priori=Skwargs.Kprio + 1,
                        square_focus=Skwargs.param_square_focus,
                        )
    elif module_name=='SegerZoomDeep':
        nn_module = SegerZoomDeep(base_module_deformation=base_module_deformation,
                        base_module=base_module,
                        classify_module=classify_module,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        class_num=class_num,
                        downsample_factor=Skwargs.downsample_factor,
                        downsample_factor_deformation=Skwargs.downsample_factor_deformation,
                        kernel_gridsp=Skwargs.Kblur + 1,
                        kernel_gblur=Skwargs.Kgrid + 1,
                        cbase_seg0=Skwargs.cbase_seg0,
                        cbase_seg=Skwargs.cbase_seg1,
                        nlayer_seg0=Skwargs.nlayer_seg0,
                        nlayer_seg=Skwargs.nlayer_seg1,
                        priori=Skwargs.param_priori,
                        kernel_priori=Skwargs.Kprio + 1,
                        square_focus=Skwargs.param_square_focus,
                        )

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

    elif module_name == 'SegerZoomSAM':
        nn_module = SegerZoomSAM(base_module_deformation=base_module_deformation,
                base_module=base_module,
                classify_module=classify_module,
                in_channels=in_channels,
                out_channels=out_channels,
                class_num=class_num,
                downsample_factor=Skwargs.downsample_factor,
                downsample_factor_deformation=Skwargs.downsample_factor_deformation,
                kernel_gridsp=Skwargs.Kblur + 1,
                kernel_gblur=Skwargs.Kgrid + 1,
                cbase_seg0=Skwargs.cbase_seg0,
                cbase_seg=Skwargs.cbase_seg1,
                nlayer_seg0=Skwargs.nlayer_seg0,
                nlayer_seg=Skwargs.nlayer_seg1,
                priori=Skwargs.param_priori,
                kernel_priori=Skwargs.Kprio + 1,
                square_focus=Skwargs.param_square_focus,
                )

    calc_model_memsize(nn_module.gen_seg, label='gen_seg')
    calc_model_memsize(nn_module, label='enire_model')

    modelname = Skwargs.modelname

    if not modelname:
        modelname = ""
        modelname += datetime.now().strftime('D%y%m%d_T%H%M')
        modelname += f"_{datasetloader_train.dataset.__class__.__name__.replace('Dataset', '')}"
        modelname += f"_{len(datasetloader_train.dataset)}x{datasetloader_train.dataset.HC}x{datasetloader_train.dataset.WC}"
        modelname += f"_{datasetloader_train.dataset.HC // Skwargs.downsample_factor}x{datasetloader_train.dataset.WC // Skwargs.downsample_factor}"
        modelname += f"_{datasetloader_train.dataset.HC // Skwargs.downsample_factor_deformation}x{datasetloader_train.dataset.WC // Skwargs.downsample_factor_deformation}"

        if module_name in ['SegerZoom', 'SegerZoomCropCat']:
            modelname += f"_{base_module_deformation.__name__}"

        modelname += f"_{base_module.__name__}"

        modelname += f"_{nn_module.__class__.__name__}"

        modelname += f"_{nn_loss_fctn_seg.__class__.__name__.replace('Loss', '')}"

        if module_name in ['SegerZoom', 'SegerZoomCropCat']:
            modelname += f"_cbase-{Skwargs.cbase_seg0}-{Skwargs.cbase_seg1}"
            modelname += f"_nlayer-{Skwargs.nlayer_seg0}-{Skwargs.nlayer_seg1}"
            modelname += f"_kblur{Skwargs.Kblur}"
            modelname += f"_kgrid{Skwargs.Kgrid}"
            modelname += f"_kprio{Skwargs.Kprio}"

            modelname += f"_wloss-{''.join([str(w) for w in wloss])}"
            modelname += f"_pri-{int(Skwargs.param_priori)}"
            modelname += f"_sqf-{int(Skwargs.param_square_focus)}"
        else:
            modelname += f"_cbase-{Skwargs.cbase_seg1}"
            modelname += f"_nlayer-{Skwargs.nlayer_seg1}"

    print(modelname)

    print(target_device)
    n_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {n_gpus}")
    torch.cuda.set_device(rank)  

    nn_module = nn_module.to(rank)
    if Skwargs.ckpt_path is not None:
        nn_module.load_state_dict(torch.load(Skwargs.ckpt_path, map_location=f'cuda:{rank}'), strict=False)
    nn_module = DDP(nn_module, device_ids=[rank], output_device=rank, find_unused_parameters = True)

    """
    -------------------------------------------------------------------------------
    Finetune Process of Seg_0
    """

    mm = ModelManager(nn_module, modelname, target_device, refresh=False)
    epoch = 800
    plot_per_N_epoch = 50
    eval_per_N_epoch = 50
    save_per_N_epoch = 50
    print_per_N_epoch = 10
    init_lr = 0.02
    decay = 0.9
    wloss = [0, 1, 0.]

    optimizer = torch.optim.NAdam(filter(lambda p: p.requires_grad, mm.model.parameters()), lr=init_lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=decay, patience=20)
    earlystop = EarlyStopMax()
    save_trigger = False
    mgpu = RAM()

    batch_size_train = Skwargs.batch_size_train // 3 // world_size 
    batch_size_val = Skwargs.batch_size_valid // 3 // world_size 

    if Skwargs.train:
        for ep in trange(epoch):
            train_sampler.set_epoch(ep)
            mm.model.train(True)
            cur_lr, cur_loss_train = mm.train(mm.model, target_device, loss_fctn_seg=nn_loss_fctn_seg, loss_fctn_cls=nn_loss_fctn_cls, wloss=wloss, optimizer=optimizer, datasetloader_train=datasetloader_train, batch_size=batch_size_train, lr_scheduler=lr_scheduler)

            mm.model.train(False)
            # mgpu.show_cuda_info()
            save_trigger = False

            lr_scheduler.step(cur_loss_train)
            mm.recorder.add_scalar('train_loss', cur_loss_train, ep)

            if (ep+1) % save_per_N_epoch == 0:
                print(ep, save_per_N_epoch)
                save_trigger = True

            if (ep+1) % print_per_N_epoch == 0:
                print(f"ep={ep} loss={cur_loss_train:.6f} lr={cur_lr:.6f} trig={int(save_trigger)}")

            # if (ep+1) % plot_per_N_epoch == 0:
            #     mm.plot_figure(datasetloader_train, fname=f"ep{ep}_stage2_train_plot.png", rank=rank)
            #     mm.plot_figure(datasetloader_valid, fname=f"ep{ep}_stage2_valid_plot.png", rank=rank)

            if save_trigger:
                res_bind = mm.get_metrics(class_num=class_num, datasetloader=datasetloader_valid,
                                        namekeys=datasetloader_valid.dataset.get_namekeys(),
                                        dataset_partition=datasetloader_valid.dataset.dataset_partition,
                                        batch_size=batch_size_val, show=False, save=False, rank=rank, world_size=world_size)

                mean_seg_iou = res_bind

                res_bind = mm.get_metrics(class_num=class_num, datasetloader=datasetloader_train,
                                        namekeys=datasetloader_train.dataset.get_namekeys(),
                                        dataset_partition=datasetloader_train.dataset.dataset_partition,
                                        batch_size=batch_size_train, show=False, save=False, rank=rank, world_size=world_size)

                mean_seg_iou = res_bind
            torch.distributed.barrier()

            mgpu = RAM()
            mgpu.delete_all()
            mgpu.gc()

        mm.recorder.flush()
        mm.recorder.close()


    """
    -------------------------------------------------------------------------------
    Finetune Process of Classification
    

    finetune_module_name = 'cls' 
    # nn_module.load_state_dict(torch.load(mm.fpath_work_pt, map_location=f'cuda:{rank}'), strict=False)

    for name, param in nn_module.named_parameters():
        if 'seg_0' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    mm = ModelManager(nn_module, modelname, target_device, refresh=False)


    epoch = 500
    plot_per_N_epoch = 50
    eval_per_N_epoch = 50
    save_per_N_epoch = 50
    print_per_N_epoch = 10
    init_lr = 0.000001
    decay = 0.9
    wloss = [0, 0.01, 1]

    optimizer = torch.optim.NAdam(filter(lambda p: p.requires_grad, mm.model.parameters()), lr=init_lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=decay, patience=20)
    save_trigger = False
    mgpu = RAM()

    batch_size_train = Skwargs.batch_size_train // 5 // world_size 
    batch_size_val = Skwargs.batch_size_valid // 5 // world_size 

    if Skwargs.train:
        for ep in trange(epoch):
            train_sampler.set_epoch(ep)
            mm.model.train(True)
            cur_lr, cur_loss_train = mm.train(mm.model, target_device, loss_fctn_seg=nn_loss_fctn_seg, loss_fctn_cls=nn_loss_fctn_cls, wloss=wloss, optimizer=optimizer, datasetloader_train=datasetloader_train, batch_size=batch_size_train, lr_scheduler=lr_scheduler)

            mm.model.train(False)
            # mgpu.show_cuda_info()
            save_trigger = False

            lr_scheduler.step(cur_loss_train)
            mm.recorder.add_scalar('train_loss', cur_loss_train, ep)

            if (ep+1) % save_per_N_epoch == 0:
                print(ep, save_per_N_epoch)
                save_trigger = True

            if (ep+1) % print_per_N_epoch == 0:
                print(f"ep={ep} loss={cur_loss_train:.6f} lr={cur_lr:.6f} trig={int(save_trigger)}")

            if (ep+1) % plot_per_N_epoch == 0:
                mm.plot_figure(datasetloader_train, fname=f"ep{ep}_stage3_train_plot.png", rank=rank)
                mm.plot_figure(datasetloader_valid, fname=f"ep{ep}_stage3_valid_plot.png", rank=rank)

            if save_trigger:
                res_bind = mm.get_metrics(class_num=class_num, datasetloader=datasetloader_valid,
                                        namekeys=datasetloader_valid.dataset.get_namekeys(),
                                        dataset_partition=datasetloader_valid.dataset.dataset_partition,
                                        batch_size=batch_size_val, show=False, save=False, rank=rank, world_size=world_size)

                mean_seg_iou, mean_seg_f1, mean_seg_accuracy, mean_seg_precision, mean_seg_recall, mean_cls_f1, mean_cls_accuracy, mean_cls_precision, mean_cls_recall = res_bind
                mean_score = 0.5 * mean_seg_f1 + 0.5 * mean_cls_f1
                if earlystop.check(mean_seg_iou):
                    if rank==0:
                        mm.save_model_state(mm.fpath_work_pt, msg=f'Best Epoch : {ep} loss={cur_loss_train:.6f}')
                        mm.get_metrics(class_num=class_num, datasetloader=datasetloader_train, namekeys=datasetloader_train.dataset.get_namekeys(), dataset_partition=datasetloader_train.dataset.dataset_partition, batch_size=batch_size_val,
                                    show=True, save=True, rank=rank, world_size=world_size,stage=3)
                        mm.get_metrics(class_num=class_num, datasetloader=datasetloader_valid, namekeys=datasetloader_valid.dataset.get_namekeys(), dataset_partition=datasetloader_valid.dataset.dataset_partition, batch_size=batch_size_val,
                                show=True, save=True, rank=rank, world_size=world_size, stage=3)
            torch.distributed.barrier()

            mgpu = RAM()
            mgpu.delete_all()
            mgpu.gc()

        mm.recorder.flush()
        mm.recorder.close()
"""
    destroy_process_group()


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed(0)

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, ), nprocs=world_size, start_method='spawn')
"""

+++
python d_model/nn_E_manager_parallel_stage.py --train --metrics --Module_Loss SegerZoomDeep_BMSELoss --dataset_marker_train sp50000 --dataset_marker_valid sp10000 --downsample_factor 4 --downsample_factor_deformation 8 --batch_size_train 1200 --batch_size_valid 600
CUDA_VISIBLE_DEVICES=0,1 python d_model/nn_E_manager_parallel_stage.py --train --metrics --Module_Loss SegerZoomDeep_BMSELoss --dataset_marker_train sp10000 --dataset_marker_valid sp2000 --downsample_factor 8 --downsample_factor_deformation 8 --batch_size_train 800 --batch_size_valid 400 --class_num 41 --dataset_class cityscapes
"""