import os
import random
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from d_model.nn_A0_utils import try_gpu, try_cpu
from utility.plot_tools import *
from utility.watch import watch_time

from e_preprocess_scripts.a_preprocess_tools import CustomDataLoader, AbstractDataset
from utility.watch import Watch

import argparse
import traceback
from shapely import Polygon, Point

import numpy as np
import skimage
import torch
from tqdm import tqdm, trange

import preset
from utility.fctn import read_json, load_image, save_tensor, save_image, load_tensor, save_jsonl, read_jsonl, save_json, read_pickle
from utility.torch_tools import str_tensor_shape, add_alpha

torch.manual_seed(0)
np.random.seed(0)

import torchvision.transforms as transforms

from PIL import Image


def get_marker(N, marker_prefix):
    return f'{marker_prefix}{N}'


def wrap_name(name):
    return name.replace(' ', '-')


def clean_sample_folder(dataset_partition: str, marker: str):
    fdatasetname = 'ade20k'
    dpath_data_cook_lvis_part_marker = os.path.join(preset.dpath_data_cook, fdatasetname, dataset_partition, marker)

    dpath = dpath_data_cook_lvis_part_marker
    if os.path.exists(dpath):
        for fname in tqdm(list(os.listdir(dpath))):
            os.remove(os.path.join(dpath, fname))
        os.rmdir(dpath)
    print(f"CLEAN {dpath}")


ade_class_ids_150 = [2978, 312, 2420, 976, 2855, 447, 2131, 165, 3055, 1125, 350, 2377, 1831, 838, 774, 783, 2684, 1610, 1910, 687, 471, 401, 2994, 1735, 2473, 2329, 1276, 2264, 1564, 2178, 913, 57, 2272, 907, 724, 2138, 2985, 533, 1395, 155, 2053, 689, 137, 266, 581, 2380, 491, 627, 2212, 2388, 2423, 943, 2096, 1121,
                     1788, 2530, 2185, 420, 1948, 1869, 2251, 2531, 2128, 294, 239, 212, 571, 2793, 978, 236, 1240, 181, 629, 2598, 1744, 1374, 591, 2679, 223, 123, 47, 1282, 327, 2821, 1451, 2880, 2828, 480, 77, 2616, 246, 247, 2733, 14, 738, 38, 1936, 1401, 120, 868, 1702, 249, 308, 1969, 2526, 2928, 2337, 1023, 609,
                     389, 2989, 1930, 2668, 2586, 131, 146, 3016, 2739, 95, 1563, 642, 1708, 103, 1002, 2569, 2704, 2833, 1551, 1981, 29, 187, 1393, 747, 2254, 206, 2262, 1260, 2243, 2932, 2836, 2850, 64, 894, 1858, 3109, 1919, 1583, 318, 2356, 2046, 1098, 530, 954]
ade_class_ids = ade_class_ids_150[:50]
ade_class_idxs = {}
idx = 0
for id in ade_class_ids:
    ade_class_idxs[id - 1] = idx
    idx += 1


class PreprocessADE:

    @watch_time
    def __init__(self, dataset_partition='train'):

        self.dataset_partition = dataset_partition

        self.fdatasetname = 'ade20k'
        self.crop_H = 640
        self.crop_W = 640
        self.HD = self.crop_H
        self.WD = self.crop_W
        self.K = len(ade_class_ids)
        self.LABEL_unlabeled = 'unlabeled'

        self.path_data_cache_ade20k = os.path.join(preset.dpath_data_cache, self.fdatasetname)
        self.path_data_cook_ade20k = os.path.join(preset.dpath_data_cook, self.fdatasetname)
        self.path_data_cache_ade20k_part = os.path.join(self.path_data_cache_ade20k, self.dataset_partition)
        self.path_data_cook_ade20k_part = os.path.join(self.path_data_cook_ade20k, self.dataset_partition)

        fpath_pkl = os.path.join(preset.dpath_data_raw_ade20k, r'index_ade20k.pkl')
        self.info = read_pickle(fpath_pkl)

        os.makedirs(self.path_data_cache_ade20k, exist_ok=True)
        os.makedirs(self.path_data_cook_ade20k, exist_ok=True)
        os.makedirs(self.path_data_cache_ade20k_part, exist_ok=True)
        os.makedirs(self.path_data_cook_ade20k_part, exist_ok=True)

    def pad_or_crop_image(self, image, mask, target_size, coord):
        H, W = target_size
        h, w = coord

        # Get original size
        original_width, original_height = image.size

        # Check if the coordinate is valid
        if not (0 <= w < original_width and 0 <= h < original_height):
            raise ValueError("Coordinate (h, w) must be within the bounds of the image.")

        gray_image = Image.new("L", (original_width, original_height), 255)
        if original_width < W or original_height < H:
            # Pad the image
            padding = (
                max(0, W - original_width),  # Right
                max(0, H - original_height)  # Bottom
            )
            # Create a new black image and paste the original image
            padded_image = Image.new('RGB', (original_width + padding[0], original_height + padding[1]), (0, 0, 0))
            padded_image.paste(image, (0, 0))
            image = padded_image  # Update image reference to the padded image

            padded_mask = Image.new('L', (original_width + padding[0], original_height + padding[1]), (0))
            padded_mask.paste(mask, (0, 0))
            mask = padded_mask  # Update image reference to the padded image

            # Original image part is 1, padded part is 0
            padded_gray_image = Image.new("L", (original_width + padding[0], original_height + padding[1]), 0)
            padded_gray_image.paste(gray_image, (0, 0))
            gray_image = padded_gray_image
            # Update new (x, y) coordinates in the padded image
            new_h = h
            new_w = w
        else:
            new_h, new_w = h, w  # No change in coordinates if no padding

        # Now randomly crop the padded or original image
        original_width, original_height = image.size  # Update sizes after padding if needed

        # Calculate valid ranges for the top-left corner of the crop
        half_h, half_w = H // 2, W // 2
        left_min = max(0, w - half_w)
        left_max = min(original_width - W, w)
        top_min = max(0, h - half_h)
        top_max = min(original_height - H, h)

        # Ensure the crop doesn't go out of bounds
        left = random.randint(left_min, left_max) if left_max >= left_min else left_min
        top = random.randint(top_min, top_max) if top_max >= top_min else top_min

        # Define the crop box
        crop_box = (left, top, left + W, top + H)

        # Crop the image
        cropped_image = image.crop(crop_box)
        cropped_mask = mask.crop(crop_box)
        cropped_gray_image = gray_image.crop(crop_box)
        # Calculate the new coordinates of (x, y) in the cropped image
        new_w_in_crop = new_w - left
        new_h_in_crop = new_h - top

        return cropped_image, cropped_mask, new_h_in_crop, new_w_in_crop, cropped_gray_image

    def prep_a_sample_by_class_id_image_id(self, target_class_id, img_id, mark: str = 'default'):
        class_name = self.info['objectnames'][target_class_id]
        img_folder = self.info['folder'][img_id]
        img_name = self.info['filename'][img_id]
        img_path = '{}/{}'.format(img_folder, img_name)
        full_img_path = os.path.join(preset.dpath_data_raw, img_path)
        annotation_path = full_img_path.replace(".jpg", ".json")
        try:
            annotation = read_json(annotation_path)
        except Exception as e:
            print(f"Error in read_json: {annotation_path}, {e}")
            return
        objects = annotation['annotation']['object']
        matching_masks = [(item["instance_mask"], item["id"]) for item in objects if item["name_ndx"] == target_class_id + 1]
        random_instance_mask, instance_id = random.choice(matching_masks) if matching_masks else None
        if not random_instance_mask:
            print(f"ERROR : No class: {class_name}, class id: {target_class_id} in image: {full_img_path}")
        # instance_mask = next((item["instance_mask"] for item in objects if item["name_ndx"] == target_class_id+1), None)
        full_instance_mask_path = os.path.join(preset.dpath_data_raw, img_folder, random_instance_mask)
        modified_full_instance_mask_path = full_instance_mask_path.replace('/', '_').replace('\\', '_')
        instance_mask_gray = Image.open(full_instance_mask_path).convert('L')
        instance_mask_array = np.array(instance_mask_gray)
        has_only_0_128_255 = np.all((instance_mask_array == 0) | (instance_mask_array == 255) | (instance_mask_array == 128))
        if not has_only_0_128_255:
            print(f"ERROR : more than three color in instance mask: {full_instance_mask_path}")
        coords = np.argwhere(instance_mask_array == 255)
        if coords.size > 0:
            idx_H, idx_W = coords[random.randint(0, len(coords) - 1)]
        else:
            print(f"ERROR : no instance in instance mask: {full_instance_mask_path}")
            return

        instance_mask_binary = instance_mask_gray.point(lambda p: 255 if p > 128 else 0)

        view_rgb_3xHxW = load_image(full_img_path)

        view_rgb_pil = transforms.ToPILImage()(view_rgb_3xHxW)
        labelidx_tracker_pil = instance_mask_binary
        target_size = (self.HD, self.WD)
        coord = (idx_H, idx_W)
        cropped_image_pil, cropped_labelidx_pil, idx_HS, idx_WS, alpha_mask_image = self.pad_or_crop_image(view_rgb_pil, labelidx_tracker_pil, target_size, coord)

        view_rgb_3xHPxWP = transforms.ToTensor()(cropped_image_pil)
        labelidx_1xHPxWP = transforms.ToTensor()(cropped_labelidx_pil)

        XY_rgba_4xHPxWP = add_alpha(view_rgb_3xHPxWP)
        XY_rgba_4xHPxWP[-1, :, :] = labelidx_1xHPxWP[0, :, :]

        target_class_idx = int(ade_class_idxs[int(target_class_id)])
        class_name = class_name.replace(', ', '-')

        if not (0 <= idx_HS < self.HD and 0 <= idx_WS < self.WD):
            raise ValueError(f"Adjusted index ({idx_HS}, {idx_WS},{self.HD},{self.WD}) is out of bounds after cropping.")

        save_image(XY_rgba_4xHPxWP, os.path.join(self.path_data_cook_ade20k_part, mark, f'{class_name}_c{target_class_idx}_k{target_class_idx}_{img_id}_{idx_HS}x{idx_WS}_{str_tensor_shape(XY_rgba_4xHPxWP)}.uint8.XY.png'))

    def prep_N_samples_per_class(self, class_id, samples_per_class, mark):
        if self.dataset_partition == 'train':
            part = 'train'
        else:
            part = 'val'
        img_ids = np.array([i for i in range(len(self.info['filename']))
                            if part in self.info['filename'][i] and
                            self.info['objectPresence'][class_id][i] > 0])
        if img_ids.size == 0:
            class_name = self.info['objectnames'][class_id]
            print('class ', class_name, ' not exist in any image in ', self.dataset_partition)
            return
        for _ in tqdm(range(samples_per_class)):
            img_id = np.random.choice(img_ids)
            for i in range(1):
                self.prep_a_sample_by_class_id_image_id(target_class_id=class_id, img_id=img_id, mark=mark)

    def make_N_samples(self, N, marker):
        dpath = os.path.join(self.path_data_cook_ade20k_part, marker)
        os.makedirs(dpath, exist_ok=True)

        number_of_class = self.K
        samples_per_class = int(max(N / number_of_class, 1))
        # sums = np.sum(self.info['objectPresence'], axis=1)
        # sorted_class_ids = np.argsort(sums)[::-1][:number_of_class]
        sorted_class_ids = ade_class_ids
        progress_bar = tqdm(sorted_class_ids)
        for class_id in progress_bar:
            progress_bar.set_description(f"Processing class_id: {class_id}")
            self.prep_N_samples_per_class(class_id - 1, samples_per_class, mark=marker)


class DatasetADE(AbstractDataset):

    def __init__(self, marker, dataset_partition='train'):
        super().__init__()
        self.HC = 640
        self.WC = 640
        self.K = len(ade_class_ids)

        self.fdatasetname = 'ade20k'

        self.marker = marker
        self.dataset_partition = dataset_partition

        self.path_data_cache_ade20k = os.path.join(preset.dpath_data_cache, self.fdatasetname)
        self.path_data_cook_ade20k = os.path.join(preset.dpath_data_cook, self.fdatasetname)
        self.path_data_cache_ade20k_part = os.path.join(self.path_data_cache_ade20k, self.dataset_partition)
        self.path_data_cook_ade20k_part = os.path.join(self.path_data_cook_ade20k, self.dataset_partition)

        fpath_pkl = os.path.join(preset.dpath_data_raw_ade20k, r'index_ade20k.pkl')
        self.info = read_pickle(fpath_pkl)

        os.makedirs(self.path_data_cache_ade20k, exist_ok=True)
        os.makedirs(self.path_data_cook_ade20k, exist_ok=True)
        os.makedirs(self.path_data_cache_ade20k_part, exist_ok=True)
        os.makedirs(self.path_data_cook_ade20k_part, exist_ok=True)

        self.path_data_cook_ade20k_part_mark = os.path.join(self.path_data_cook_ade20k_part, self.marker)

        self.fnames_XYpng = list(self.get_fnames_Ypt())

    def get_fnames_Ypt(self):
        for entry in os.scandir(self.path_data_cook_ade20k_part_mark):
            if entry.name.endswith('.XY.png') and entry.is_file():
                yield entry.name

    def get_namekeys(self):
        return self.fnames_XYpng

    def __len__(self) -> int:
        return len(self.fnames_XYpng)

    def __getitem__(self, index):

        fname_XY = self.fnames_XYpng[index]

        class_name, ctarget_class_idx, ktarget_class_idx, img_id, fpos, IxHxW_Y = fname_XY.split('.')[0].split('_')
        target_class_idx = ctarget_class_idx[1:]

        Y_cls_s = int(target_class_idx)

        fpath_XY = os.path.join(self.path_data_cook_ade20k_part_mark, fname_XY)

        XY_rgba_4xHxW = load_image(fpath_XY,mode='RGBA')

        X_3xHxW = XY_rgba_4xHxW[:-1]
        Y_1xHxW = XY_rgba_4xHxW[3:]

        idx_H, idx_W = [int(num) for num in fpos.split('x')]
        F_2 = torch.Tensor([idx_H / self.HC, idx_W / self.WC]).to(dtype=torch.float32)
        Y_cls_1 = torch.Tensor([Y_cls_s]).to(dtype=torch.int64)

        return X_3xHxW, F_2, Y_1xHxW, Y_cls_1


def get_Skwargs():
    parser = argparse.ArgumentParser(description="Script to process dataset with specified arguments.")

    # 添加参数
    parser.add_argument('--task', type=str, required=False, default='', help='Dataset partition to use (e.g., preprocess, speed_test )')

    # for preprocess
    parser.add_argument('--dataset_partition', nargs='+', type=str, required=False, help='Dataset partition to use (e.g., train, val, test)')
    parser.add_argument('--sample_num', type=int, nargs='+', required=False, help='Number of samples to process')
    parser.add_argument('--marker_prefix', type=str, required=False, default='sp', help='Marker to use for labeling or identification')

    # for speed_test
    parser.add_argument('--epoch', type=int, required=False, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, required=False, default=64, help='Size of each training batch')
    parser.add_argument('--HW_size', type=int, required=False, default=640, help='Size of each training batch')

    parser.add_argument('--target_device', type=str, required=False, default='gpu', help='Device to use for training (e.g., cpu, gpu)')
    parser.add_argument('--cache', action='store_true', help='Whether to cache the dataset in memory for faster access')
    parser.add_argument('--show', action='store_true', help='show sample number')

    parser.add_argument('--delete', action='store_true', help='delete cook data')
    parser.add_argument('--all_cidxs', action='store_true', help='delete cook data')

    # 解析参数
    return parser.parse_args()


def print_sample_count():
    dpath_train = os.path.join(preset.dpath_data_cook, 'ade20k', 'train')
    dpath_valid = os.path.join(preset.dpath_data_cook, 'ade20k', 'valid')

    for dpath in [dpath_train, dpath_valid]:
        print(dpath)
        for folder in os.listdir(dpath):
            dpath_sp = os.path.join(dpath, folder)
            print(folder, '\t', len(os.listdir(dpath_sp)))


if __name__ == '__main__':
    pass

    Skwargs = get_Skwargs()

    if Skwargs.task in ['preprocess', 'speed_test']:
        if Skwargs.dataset_partition is None or Skwargs.sample_num is None:
            raise ValueError("Please specify dataset_partition and sample_num.")

    if Skwargs.task == 'preprocess':

        w = Watch()
        pplv_train = PreprocessADE(dataset_partition='train')
        pplv_valid = PreprocessADE(dataset_partition='valid')
        for sp_train in Skwargs.sample_num:
            sp_valid = sp_train // 5
            marker_train = get_marker(sp_train, Skwargs.marker_prefix)
            marker_valid = get_marker(sp_valid, Skwargs.marker_prefix)
            print('--------------------')
            print(sp_train)
            clean_sample_folder(dataset_partition='train', marker=marker_train)
            clean_sample_folder(dataset_partition='valid', marker=marker_valid)

            if 'train' in Skwargs.dataset_partition:
                pplv_train.make_N_samples(sp_train, marker=marker_train)
            if 'valid' in Skwargs.dataset_partition:
                pplv_valid.make_N_samples(sp_valid, marker=marker_valid)

        print(f"preprocess done! total cost {w.see_timedelta()}")

    elif Skwargs.task == 'speed_test':

        epoch = Skwargs.epoch
        batch_size = Skwargs.batch_size
        target_device = try_gpu() if Skwargs.target_device == 'gpu' else try_cpu()

        for dp in Skwargs.dataset_partition:
            for sp_train in Skwargs.sample_num:
                w = Watch()

                marker = get_marker(sp_train, Skwargs.marker_prefix)
                datasetADE20K = DatasetADE(marker, dataset_partition=dp)

                print(f"CustomDataLoader Cache={Skwargs.cache} init {w.see_timedelta()}")

                dataloader = CustomDataLoader(datasetADE20K, cache=Skwargs.cache)

                for eidx in trange(epoch):
                    for bidx, bparts in enumerate(dataloader.get_iterator(batch_size=batch_size, device=target_device, shuffle=True)):
                        # print(eidx, bidx)
                        if eidx == 0 and bidx == 0:
                            for bpart in bparts:
                                pass
                                print(bpart.device, bpart.dtype, str_tensor_shape(bpart))

                            # X_bx3xHxW, F_bx2, Y_bx1xHxW, Y_cls_bx1 = bparts
                            # print(f"# F_bx2 {F_bx2.shape}")
                            # print(f"# Y_cls_bx1 {Y_cls_bx1.shape}")
                            # plt_multi_imgshow([X_bx3xHxW[0, :], torch.cat([X_bx3xHxW[0, :], Y_bx1xHxW[0, :]], dim=0), Y_bx1xHxW[0, :]], row_col=(1, 3))
                            #
                            # plt_show()

                print(f"CustomDataLoader Cache={Skwargs.cache} per epoch cost {w.see_timedelta() / epoch}")

                print(f"CustomDataLoader Cache={Skwargs.cache} total {w.total_timedelta()}")

    if Skwargs.show:
        print_sample_count()

    if Skwargs.delete:

        for dp in Skwargs.dataset_partition:
            for sp_train in Skwargs.sample_num:
                marker = get_marker(sp_train, Skwargs.marker_prefix)
                clean_sample_folder(dataset_partition=dp, marker=marker)
        print_sample_count()

"""

python e_preprocess_scripts/b4_preprocess_ade20k.py --task preprocess --dataset_partition train valid --sample_num 60000 --HW_size 640 --marker_prefix sp640_

python e_preprocess_scripts/b4_preprocess_ade20k.py --task speed_test --dataset_partition train --sample_num 100 --HW_size 640 --marker_prefix sp640_


on local
python e_preprocess_scripts/b4_preprocess_ade20k.py.py --task preprocess --dataset_partition train valid --sample_num 10 50 100 50000


python e_preprocess_scripts/b4_preprocess_ade20k.py.py --task preprocess --dataset_partition train valid --sample_num 100 500 1000

python e_preprocess_scripts/b4_preprocess_ade20k.py.py --task speed_test --dataset_partition train --sample_num 500 --epoch 5 --batch_size 64 --target_device gpu --cache
python e_preprocess_scripts/b4_preprocess_ade20k.py.py --task speed_test --dataset_partition train --sample_num 500 --epoch 5 --batch_size 64 --target_device gpu


# on server

python e_preprocess_scripts/b4_preprocess_ade20k.py.py --task preprocess --dataset_partition train valid --sample_num 10 50 100 500
fg %1  # 或者 fg %2



ls -l | grep ^- | wc -l

cd  /home/hongyiz/DriverD/b_data_train/data_c_cook/ade20k/train/


python e_preprocess_scripts/b4_preprocess_ade20k.py.py --show

python e_preprocess_scripts/b4_preprocess_ade20k.py.py --delete --dataset_partition train valid --sample_num 10 50 100 500 2500

"""
