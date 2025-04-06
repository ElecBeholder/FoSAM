import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from e_preprocess_scripts.a_preprocess_tools import CustomDataLoader, AbstractDataset
from utility.watch import Watch

from torch.utils.data import Dataset, DataLoader

import argparse
import random
import traceback

import numpy as np
import skimage
import torch
from shapely import Polygon, Point
from tqdm import tqdm, trange

import preset
from utility.fctn import read_json, load_image, save_tensor, save_image, load_tensor, save_jsonl, read_jsonl, save_json
from utility.torch_tools import str_tensor_shape, interpolate_int, add_alpha
import torch.nn.functional as F
import torchvision.transforms as transforms


def sort_points_clockwise(anchor_points):
    # 计算质心（几何中心）
    centroid_x = sum(point[0] for point in anchor_points) / len(anchor_points)
    centroid_y = sum(point[1] for point in anchor_points) / len(anchor_points)
    centroid = (centroid_x, centroid_y)

    # 定义一个函数来计算相对于质心的极角
    def angle_from_centroid(point):
        return np.arctan2(point[1] - centroid[1], point[0] - centroid[0])

    # 使用极角对锚点进行排序
    sorted_points = sorted(anchor_points, key=angle_from_centroid, reverse=True)

    return sorted_points


def wrap_label(label):
    return label.replace(' ', '')


def find_polygon(objs, point, idx2label, label2idx):
    labelidx2idxs = [[] for _ in range(len(idx2label))]
    for idx, obj in enumerate(objs):
        label = wrap_label(obj['label'])
        polygon = sort_points_clockwise(obj['polygon'])
        is_inside = Polygon(polygon).contains(Point(point))
        if is_inside:
            labelidx2idxs[label2idx[label]].append(idx)

    for idxs in labelidx2idxs:
        if len(idxs) > 0:
            tidx = idxs[0]
            obj = objs[tidx]
            label = wrap_label(obj['label'])
            polygon = sort_points_clockwise(obj['polygon'])
            return label, polygon, tidx

    return "", [], -1


def get_rrcc_by_polygon(polygon, canvas_H, canvas_W):
    polygon_xy_Nx2 = np.array(polygon, dtype=np.int32)
    polygon_y_N = polygon_xy_Nx2[:, 0]
    polygon_x_N = polygon_xy_Nx2[:, 1]
    polygon_x_N[polygon_x_N < 0] = canvas_H + polygon_x_N[polygon_x_N < 0]
    polygon_y_N[polygon_y_N < 0] = canvas_W + polygon_y_N[polygon_y_N < 0]
    polygon_x_N = np.clip(polygon_x_N, 0, canvas_H - 1)
    polygon_y_N = np.clip(polygon_y_N, 0, canvas_W - 1)
    rr, cc = skimage.draw.polygon(polygon_x_N, polygon_y_N)
    return rr, cc


class PreprocessCityscape(AbstractDataset):

    def __init__(self, dataset_partition='train', pre_downsample_degree=2, use_cache=True):
        self.dataset_partition = dataset_partition
        self.pre_downsample_degree = pre_downsample_degree
        self.pre_downsample_factor = 2 ** self.pre_downsample_degree
        self.canvas_H = 1024
        self.canvas_W = 2048
        self.HD = self.canvas_H // self.pre_downsample_factor
        self.WD = self.canvas_W // self.pre_downsample_factor
        self.LABEL_unlabeled = 'unlabeled'
        self.K = 41
        self.fdatasetname = 'cityscape'

        self.path_data_cache_cityscape = os.path.join(preset.dpath_data_cache, self.fdatasetname)
        self.path_data_cook_cityscape = os.path.join(preset.dpath_data_cook, self.fdatasetname)
        self.path_data_cache_cityscape_part = os.path.join(self.path_data_cache_cityscape, self.dataset_partition)
        self.path_data_cook_cityscape_part = os.path.join(self.path_data_cook_cityscape, self.dataset_partition)
        self.path_data_cook_cityscape_part_canvsize = os.path.join(self.path_data_cook_cityscape, self.dataset_partition, f"{self.HD}x{self.WD}")

        os.makedirs(self.path_data_cache_cityscape, exist_ok=True)
        os.makedirs(self.path_data_cook_cityscape, exist_ok=True)
        os.makedirs(self.path_data_cache_cityscape_part, exist_ok=True)
        os.makedirs(self.path_data_cook_cityscape_part, exist_ok=True)
        os.makedirs(self.path_data_cook_cityscape_part_canvsize, exist_ok=True)

        self.use_cache = use_cache

        self.itemkey_fpathX_fpathY_s = self.cache_all_fpath_X_Y(use_cache=self.use_cache)
        self.idx2label, self.label2idx = self.get_labels_ordered()

        print(f"max_oidx = {self.get_max_oidx()}")
        # self.print_fldr2unique_label_count()

        self.labelidx2fpathidxs = self.cache_labelidx2fpathidxs(use_cache=self.use_cache)

        self.namekeys = self.get_all_namekeys()

    def print_fldr2unique_label_count(self):

        if not '\\' in self.dataset_partition:
            fdir = os.path.join(preset.dpath_data_raw_cityscape_Y, self.dataset_partition)

            fldr2unique_labels = {}
            fldr2item_count = {}
            for fldr in os.listdir(fdir):
                fldr2unique_labels[fldr] = set()
                fldr2item_count[fldr] = 0
                fdir_fldr = os.path.join(fdir, fldr)
                for root, dirs, files in os.walk(fdir_fldr):
                    for file in files:
                        fpath_json = os.path.join(root, file)
                        if fpath_json.endswith('.json'):
                            fldr2item_count[fldr] += 1
                            objs_ordered = self.get_objs_ordered(fpath_json)
                            for objs in objs_ordered:
                                fldr2unique_labels[fldr].add(wrap_label(objs['label']))
            for fldr, unique_labels in fldr2unique_labels.items():
                print(fldr, fldr2item_count[fldr], len(unique_labels), sorted(list(unique_labels), key=lambda label: self.label2idx[label]))

    def cache_all_fpath_X_Y(self, use_cache=False):
        fname = 'itemkey_fpathX_fpathY_s'

        fpath = os.path.join(self.path_data_cache_cityscape_part, f"{fname}.json")
        if use_cache and os.path.isfile(fpath):
            print(f"CACHE {fpath}")
            return read_json(fpath)
        else:

            key2fpath_X = {}
            for root, dirs, files in os.walk(os.path.join(preset.dpath_data_raw_cityscape_X, self.dataset_partition)):
                for file in files:
                    fpath_png = os.path.join(root, file)
                    if fpath_png.endswith('.png'):
                        key = file.split('_leftImg8bit')[0]
                        key2fpath_X[key] = fpath_png
                        # print(key, fpath_png)

            key2fpath_Y = {}
            for root, dirs, files in os.walk(os.path.join(preset.dpath_data_raw_cityscape_Y, self.dataset_partition)):
                for file in files:
                    fpath_png = os.path.join(root, file)
                    if fpath_png.endswith('.json'):
                        key = file.split('_gtFine_polygons')[0]
                        key2fpath_Y[key] = fpath_png
                        # print(key, fpath_png)

            # Sanity Check
            keys = set(key2fpath_X.keys()) & set(key2fpath_Y.keys())

            itemkey_fpathX_fpathY_s = []
            count_missing = 0
            for key in keys:
                if key in key2fpath_X and key in key2fpath_Y:
                    bind = [key, key2fpath_X[key], key2fpath_Y[key]]
                    itemkey_fpathX_fpathY_s.append(bind)
                else:
                    print(f"{key} not in both feature or label")

            if count_missing == 0:
                print(f"Sanity Check Passed !")

            save_json(itemkey_fpathX_fpathY_s, fpath)
            print(f"SAVE {fpath}")

            return itemkey_fpathX_fpathY_s

    def get_labels(self):
        labels = set()
        for key, fpath_feature, fpath_label in self.itemkey_fpathX_fpathY_s:
            info = read_json(fpath_label)
            for obj in info['objects']:
                obj_label = wrap_label(obj['label'])
                labels.add(obj_label)
        idx2label = sorted(list(labels))
        label2idx = {label: idx for idx, label in enumerate(idx2label)}
        return idx2label, label2idx

    def get_labels_ordered(self):
        idx2label = [
            "person", "persongroup",
            "rider", "ridergroup",

            "bicycle", "bicyclegroup",
            "motorcycle", "motorcyclegroup",
            "car", "cargroup",
            "truck", "truckgroup",
            "bus", "train",
            "caravan", "trailer",

            "license plate",

            "traffic light", "traffic sign",
            "pole", "polegroup",

            "bridge", "fence", "guard rail", "tunnel", "building", "wall",
            "rail track", "sidewalk", "parking", "road",
            "vegetation", "terrain", "ground",
            "dynamic", "static", "ego vehicle", "sky",
            "out of roi", "rectification border",
            "unlabeled"
        ]

        idx2label = [wrap_label(label) for label in idx2label]
        label2idx = {label: idx for idx, label in enumerate(idx2label)}

        print(f"#idx2label : {len(idx2label)}")
        print(f"#label2idx : {len(label2idx)}")

        return idx2label, label2idx

    def get_objs_ordered(self, fpath_Y):
        info = read_json(fpath_Y)
        objs = info['objects']
        objs.sort(key=lambda obj: (
            self.label2idx[wrap_label(obj['label'])],
            len(obj['polygon'])
        )
                  )

        # important & small object at front
        return objs

    def get_max_oidx(self):
        max_oidx = 0
        for pathidx, (itemkey, fpath_X, fpath_Y) in enumerate(self.itemkey_fpathX_fpathY_s):
            objs = self.get_objs_ordered(fpath_Y)
            max_oidx = max(max_oidx, len(objs))
        return max_oidx

    # @watch_time
    def cache_oidx_HxW(self, objs, itemkey, use_cache=True):
        bits2dtype = {8: torch.uint8, 16: torch.int16, 32: torch.int32, 64: torch.int64}

        try:
            dirname = 'oidx_HxW_s'
            os.makedirs(os.path.join(self.path_data_cache_cityscape_part, dirname), exist_ok=True)

            fpath = os.path.join(self.path_data_cache_cityscape_part, dirname, f"{itemkey}.HxW.oidx.pth")
            if use_cache and os.path.isfile(fpath):
                # print(f"CACHE {fpath}")
                return load_tensor(fpath)
            else:
                N_objs = len(objs)
                base = 8 if N_objs < 255 else 16
                dtype_target = bits2dtype[base]
                oidx_default = 2 ** base - 1

                oidx_HxW = torch.ones((self.canvas_H, self.canvas_W), dtype=dtype_target) * oidx_default
                for oidx, obj in enumerate(objs[::-1]):
                    # from least to most important objs, mark from back to front

                    polygon = sort_points_clockwise(obj['polygon'])
                    rr, cc = get_rrcc_by_polygon(polygon, self.canvas_H, self.canvas_W)
                    oidx_HxW[rr, cc] = N_objs - 1 - oidx
                save_tensor(oidx_HxW, fpath)
                # print(f"SAVE {fpath}")
                return oidx_HxW
        except Exception as e:
            traceback.print_exc()
            print(f"ERROR : {itemkey}")
            base = 8
            dtype_target = bits2dtype[base]
            oidx_default = 2 ** base - 1

            return torch.ones((self.canvas_H, self.canvas_W), dtype=dtype_target) * oidx_default

    def cache_labelidx2fpathidxs(self, use_cache=False):
        fname = 'labelidx2fpathidxs'

        fpath = os.path.join(self.path_data_cache_cityscape_part, f"{fname}.jsonl")
        if use_cache and os.path.isfile(fpath):
            print(f"CACHE {fpath}")
            return read_jsonl(fpath)
        else:
            labelidx2pathidxs = [[] for _ in range(self.K)]
            for pathidx, (itemkey, fpath_X, fpath_Y) in enumerate(tqdm(self.itemkey_fpathX_fpathY_s)):

                objs = self.get_objs_ordered(fpath_Y)
                oidx_HxW = self.cache_oidx_HxW(objs, itemkey)

                oidxs_visible = torch.unique(oidx_HxW.flatten())

                for oidx in oidxs_visible:
                    if oidx != 255:
                        obj = objs[oidx]
                        label = wrap_label(obj['label'])
                        labelidx = self.label2idx[label]
                        labelidx2pathidxs[labelidx].append(pathidx)
            save_jsonl(labelidx2pathidxs, fpath)
            print(f"SAVE {fpath}")
            return labelidx2pathidxs

    def rank_itemkeys(self):
        itemkeys = []
        itemkey_2_Kobjs = {}
        itemkey_2_Nobjs = {}
        for itemkey, fpath_X, fpath_Y in tqdm(self.itemkey_fpathX_fpathY_s):
            objs = self.get_objs_ordered(fpath_Y)
            Nobjs = len(objs)
            Kobjs = len({obj["label"] for obj in objs})

            itemkeys.append(itemkey)
            itemkey_2_Kobjs[itemkey] = Kobjs
            itemkey_2_Nobjs[itemkey] = Nobjs

        itemkeys.sort(key=lambda itemkey: (itemkey_2_Kobjs[itemkey], itemkey_2_Nobjs[itemkey]), reverse=True)
        return itemkeys

    def make_a_sample(self, itemkey, idx_H, idx_W, fpath_X, fpath_Y, oidx_target):
        objs_ordered = self.get_objs_ordered(fpath_Y)

        obj_target = objs_ordered[oidx_target]
        label_target = wrap_label(obj_target['label'])
        labelidx_target = self.label2idx[label_target]

        labelidx_tracker_HxW = torch.ones((self.canvas_H, self.canvas_W), dtype=torch.uint8) * self.label2idx[self.LABEL_unlabeled]

        polygon_target = sort_points_clockwise(obj_target['polygon'])
        rr_t, cc_t = get_rrcc_by_polygon(polygon_target, self.canvas_H, self.canvas_W)
        labelidx_tracker_HxW[rr_t, cc_t] = labelidx_target

        labelidx_HSxWS = interpolate_int(labelidx_tracker_HxW, degree=self.pre_downsample_degree)

        view_rgb_3xHxW = load_image(fpath_X)
        view_rgb_3xHSxWS = F.interpolate(view_rgb_3xHxW.unsqueeze(0), size=(self.HD, self.WD), mode='bilinear', align_corners=True).squeeze(0)

        X_bind_3xHxW = (view_rgb_3xHSxWS * 255).to(dtype=torch.uint8)
        Y_bind_1xHxW = labelidx_HSxWS.unsqueeze(0).to(dtype=torch.uint8)

        idx_HS = idx_H // self.pre_downsample_factor
        idx_WS = idx_W // self.pre_downsample_factor

        save_tensor(X_bind_3xHxW, os.path.join(self.path_data_cook_cityscape_part_canvsize, f'{label_target}_{itemkey}_{idx_HS}x{idx_WS}.{str_tensor_shape(X_bind_3xHxW)}.uint8.X.pt'))
        save_tensor(Y_bind_1xHxW, os.path.join(self.path_data_cook_cityscape_part_canvsize, f'{label_target}_{itemkey}_{idx_HS}x{idx_WS}.{str_tensor_shape(Y_bind_1xHxW)}.uint8.Y.pt'))

        view_rgba_4xHSxWS = add_alpha(view_rgb_3xHSxWS)

        # canvas_HSxWS = 0 => alpha = alpha_mask
        # canvas_HSxWS = 1 => alpha = 1

        alpha_mask = 0.25
        view_rgba_4xHSxWS[-1, :, :] = (1 - alpha_mask) * (labelidx_HSxWS == labelidx_target) + alpha_mask

        color_focus = [1, 0, 1]
        view_rgba_4xHSxWS[:-1, idx_HS, idx_WS] = torch.tensor(color_focus, dtype=torch.float32)
        save_image(view_rgba_4xHSxWS, os.path.join(self.path_data_cook_cityscape_part_canvsize, f'{label_target}_{itemkey}_{idx_HS}x{idx_WS}.{str_tensor_shape(view_rgba_4xHSxWS)}.float32.png'))

    def clean_sample_folder(self):
        for fname in tqdm(list(os.listdir(self.path_data_cook_cityscape_part_canvsize))):
            os.remove(os.path.join(self.path_data_cook_cityscape_part_canvsize, fname))
        print(f"CLEAN {self.path_data_cook_cityscape_part_canvsize}")

    def prep_a_sample_by_label(self, label_target='person'):

        labelidx_target = self.label2idx[label_target]
        if labelidx_target >= len(self.labelidx2fpathidxs) - 1:
            print(f"DO NOT LET label_target = 'unlabeled'")
            return

        pathidxs = self.labelidx2fpathidxs[labelidx_target]
        if len(pathidxs) == 0:
            print(f'NOT FOUND label_target = {label_target}')
            return

        pathidx = random.choice(pathidxs)
        itemkey, fpath_X, fpath_Y = self.itemkey_fpathX_fpathY_s[pathidx]

        objs_ordered = self.get_objs_ordered(fpath_Y)
        oidx_HxW = self.cache_oidx_HxW(objs_ordered, itemkey)

        oidxs_valid = [oidx for oidx, obj in enumerate(objs_ordered) if wrap_label(obj['label']) == label_target]

        oidx_target = -1
        idx_H, idx_W = -1, -1
        random.shuffle(oidxs_valid)
        for oidx_candidate in oidxs_valid:
            focus_grid_Nx2 = torch.argwhere(oidx_HxW == oidx_candidate)

            N, _ = focus_grid_Nx2.shape
            if N > 0:
                idx_H, idx_W = focus_grid_Nx2[np.random.randint(N)].tolist()
                oidx_target = oidx_candidate
                break
        if oidx_target == -1:
            print(f"ERROR : label_target = {label_target}, itemkey = {itemkey}")
            return
        self.make_a_sample(itemkey, idx_H, idx_W, fpath_X, fpath_Y, oidx_target)

    def prep_N_samples_by_label(self, samples_per_class=5):

        labels = ["person", "persongroup",
                  "rider", "ridergroup",

                  "bicycle", "bicyclegroup",
                  "motorcycle", "motorcyclegroup",
                  "car", "cargroup",
                  "truck", "truckgroup",
                  "bus", "train",
                  "caravan", "trailer",

                  "license plate",

                  "traffic light", "traffic sign",
                  "pole", "polegroup",

                  "bridge", "fence", "guard rail", "tunnel", "building", "wall",
                  "rail track", "sidewalk", "parking", "road",
                  "vegetation", "terrain", "ground",
                  "dynamic", "static", "ego vehicle", "sky"]
        # TODO sky sample has problem, vertices are not in order
        for label in tqdm(labels):
            label = wrap_label(label)
            for _ in range(samples_per_class):
                try:
                    self.prep_a_sample_by_label(label_target=label)
                except Exception as e:
                    traceback.print_exc()
                    print(f"ERROR : trgt_label={label}")

    def get_all_label2namekeys(self):
        label2namekeys = {}

        for fname in os.listdir(self.path_data_cook_cityscape_part_canvsize):
            if fname.endswith('.X.pt'):
                name = fname.split('.')[0]
                label = name.split('_')[0]
                if label not in label2namekeys:
                    label2namekeys[label] = []
                label2namekeys[label].append(name)

        return label2namekeys

    def get_all_namekeys(self):
        namekeys = []

        for fname in os.listdir(self.path_data_cook_cityscape_part_canvsize):
            if fname.endswith('.X.pt'):
                namekey = fname.split('.')[0]

                namekeys.append(namekey)

        return namekeys

    def __len__(self) -> int:
        return len(self.namekeys)

    def __getitem__(self, idx):
        namekey = self.namekeys[idx]

        fpath_X = os.path.join(self.path_data_cook_cityscape_part_canvsize, f'{namekey}.3x{self.HD}x{self.WD}.uint8.X.pt')
        fpath_Y = os.path.join(self.path_data_cook_cityscape_part_canvsize, f'{namekey}.1x{self.HD}x{self.WD}.uint8.Y.pt')
        idx_H, idx_W = map(float, namekey.split('_')[-1].split('x'))

        X_rgb_3xHxW = load_tensor(fpath_X).to(dtype=torch.float32) / 255.0
        X_focus_2 = torch.tensor([idx_H / self.HD, idx_W / self.WD], dtype=torch.float32)
        Y_1xHxW = (load_tensor(fpath_Y) < 40).to(dtype=torch.float32)

        return X_rgb_3xHxW, X_focus_2, Y_1xHxW

    def load_rgba_sample_by_namekey(self, namekey):
        fpath = os.path.join(self.path_data_cook_cityscape_part_canvsize, f"{namekey}.4x{self.HD}x{self.WD}.float32.png")

        x_rgba_4xHxW = load_image(fpath, mode='RGBA').to(dtype=torch.float32)
        return x_rgba_4xHxW


def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Preprocess the Cityscape dataset with specified parameters.")

    # Add arguments
    parser.add_argument(
        '--dataset_partition', type=str, required=True,
        nargs='+',  # This allows multiple values
        choices=['train', 'val'], help="Dataset partition(s) to use: 'train', 'val', or both."
    )

    parser.add_argument(
        '--downsample_degree', type=int, nargs='+', required=True,
        help="One or more downsample degrees to use (e.g., 2 3 4)."
    )
    parser.add_argument(
        '--samples_per_class', type=int, required=True,
        help="Number of samples per class to prepare (e.g., 25)."
    )
    parser.add_argument(
        '--clean', type=str, choices=['yes', 'no'], default='yes',
        help="Whether to clean the sample folder before processing: 'yes' (default) or 'no'."
    )
    parser.add_argument(
        '--use_cache', type=bool, default=True, nargs='?',
        const=True, help="Use cache: True or False. Default is True."
    )

    # Parse arguments
    args = parser.parse_args()

    # Iterate over each downsample degree specified
    for dd in args.downsample_degree:
        for dp in args.dataset_partition:
            print(f"Processing with downsample_degree={dd}, dataset_partition='{dp}', samples_per_class={args.samples_per_class}, clean='{args.clean}', use_cache={args.use_cache}")

            # Create an instance of PreprocessCityscape
            ppcc = PreprocessCityscape(
                dataset_partition=dp,
                pre_downsample_degree=dd,
                use_cache=args.use_cache
            )

            # Clean the sample folder if requested
            if args.clean == 'yes':
                ppcc.clean_sample_folder()

            # Prepare the specified number of samples per class
            ppcc.prep_N_samples_by_label(samples_per_class=args.samples_per_class)


if __name__ == "__main__":
    # main()

    if preset.pc_name == 'XPS':
        ppcc_train_dataset = PreprocessCityscape(dataset_partition='train', pre_downsample_degree=2, use_cache=True)

        target_device = torch.device('cuda:0')

        epoch = 20
        batch_size = 16

        ###########################################################CustomDataLoader Cache=True###########################################################
        w = Watch()
        dataloader = CustomDataLoader(ppcc_train_dataset, cache=True)
        print(f"CustomDataLoader Cache=True init {w.see_timedelta()}")

        for i in range(epoch):
            for bparts in dataloader.get_iterator(batch_size=batch_size, device=target_device, shuffle=True):
                [str_tensor_shape(bpart) for bpart in bparts]

        print(f"CustomDataLoader Cache=True iter {w.see_timedelta()}")

        print(f"CustomDataLoader Cache=True total {w.total_timedelta()}")

        ###########################################################CustomDataLoader Cache=False###########################################################
        w = Watch()
        dataloader = CustomDataLoader(ppcc_train_dataset, cache=False)
        print(f"CustomDataLoader Cache=False init {w.see_timedelta()}")

        for i in range(epoch):
            for bparts in dataloader.get_iterator(batch_size=batch_size, device=target_device, shuffle=True):
                [str_tensor_shape(bpart) for bpart in bparts]

        print(f"CustomDataLoader Cache=False iter {w.see_timedelta()}")

        print(f"CustomDataLoader Cache=False total {w.total_timedelta()}")

        ###########################################################DataLoader###########################################################
        w = Watch()
        dataloader = DataLoader(ppcc_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        print(f"DataLoader init {w.see_timedelta()}")

        for i in range(epoch):
            for bparts in dataloader:
                [str_tensor_shape(bpart.to(device=target_device)) for bpart in bparts]

        print(f"DataLoader iter {w.see_timedelta()}")

        print(f"DataLoader total {w.total_timedelta()}")

    #
    # for batch_idx, data in enumerate(tqdm(dataloader)):
    #     print(batch_idx, [str_tensor_shape(ts) for ts in data])
    # 
    # for batch_idx, data in enumerate(tqdm(dataloader)):
    #     print(batch_idx, [str_tensor_shape(ts) for ts in data])

    # for dd in [2]:
    #     ppcc_train_d = PreprocessCityscape(dataset_partition='train', downsample_degree=dd, use_cache=True)
    #     ppcc_train_d.clean_sample_folder()
    #     ppcc_train_d.prep_N_samples_by_label(samples_per_class=25)
    #
    # for dd in [2]:
    #     ppcc_valid_d = PreprocessCityscape(dataset_partition='val', downsample_degree=dd, use_cache=True)
    #     ppcc_valid_d.clean_sample_folder()
    #     ppcc_valid_d.prep_N_samples_by_label(samples_per_class=25)

    # python e_preprocess_scripts/g241005_b1_preprocess_cityscapes.py -h
    # python e_preprocess_scripts/g241005_b1_preprocess_cityscapes.py --dataset_partition train val --downsample_degree 2 --samples_per_class 10
    # python e_preprocess_scripts/g241005_b1_preprocess_cityscapes.py --dataset_partition train val --downsample_degree 0 1 2 --samples_per_class 100
