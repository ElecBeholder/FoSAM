import os
from datetime import timedelta

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import trange

import preset
from d_model.nn_A0_utils import try_gpu
from e_preprocess_scripts.b6_preprocess_aria_adt import DatasetADT
from utility.fctn import save_image


def segmentation_overlay(img_ds_RGB_Bx3xHxW, seg_mask_Bx1xHxW, overlay_color=(0.5, 0.7, 1), alpha=0.8):
    """
    在原始图像上对分割区域应用半透明颜色叠加，未被分割掩盖的区域保持原始颜色。

    参数:
    - img_ds_RGB_Bx3xHxW (torch.Tensor): 原始 RGB 图像，形状为 (B, 3, H, W)。
    - seg_mask_Bx1xHxW (torch.Tensor): 分割掩码，范围为 [0, 1]，形状为 (B, 1, H, W)。
    - overlay_color (tuple): 覆盖区域的颜色 (RGB)，范围为 [0, 1]。
    - alpha (float): 半透明的程度，范围为 [0, 1]，默认为 0.5。

    返回:
    - torch.Tensor: 混合后的图像，形状为 (B, 3, H, W)。
    """
    assert img_ds_RGB_Bx3xHxW.shape[1] == 3, "输入图像必须为 RGB 格式 (B, 3, H, W)"
    assert seg_mask_Bx1xHxW.shape[1] == 1, "分割掩码必须为单通道 (B, 1, H, W)"
    assert 0 <= alpha <= 1, "Alpha 必须在 [0, 1] 范围内"

    # 将 overlay_color 转为张量 (1, 3, 1, 1)
    overlay_color_tensor = torch.tensor(overlay_color, device=img_ds_RGB_Bx3xHxW.device).view(1, 3, 1, 1)

    # 扩展掩码到 3 通道
    seg_mask_Bx3xHxW = seg_mask_Bx1xHxW.expand(-1, 3, -1, -1)

    # 混合公式：output = (1 - mask) * original + mask * (alpha * overlay_color + (1 - alpha) * original)
    blended_img = img_ds_RGB_Bx3xHxW * (1 - seg_mask_Bx3xHxW) + seg_mask_Bx3xHxW * (
            alpha * overlay_color_tensor + (1 - alpha) * img_ds_RGB_Bx3xHxW
    )

    return blended_img


def gen_gif_from_pngs(folder_path, output_gif_path, interval_ms=500):
    """
    读取指定文件夹中的 PNG 图片，按文件名排序后生成一个 GIF 动画。

    参数:
    - folder_path (str): PNG 文件所在的文件夹路径。
    - output_gif_path (str): 输出 GIF 文件的路径。
    - duration (int): 每帧持续时间，单位为毫秒 (默认 500 毫秒)。
    """
    # 获取文件夹中所有的 PNG 文件，并按文件名排序
    png_files = sorted(
        [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')]
    )

    if not png_files:
        raise ValueError("指定文件夹中没有找到 PNG 文件！")

    # 打开 PNG 文件并存入列表
    images = [Image.open(png) for png in png_files]

    # 保存为 GIF
    images[0].save(
        output_gif_path,
        save_all=True,
        append_images=images[1:],  # 添加剩余帧
        duration=interval_ms,  # 每帧持续时间
        loop=0  # 无限循环
    )

    print(f"GIF 已成功生成并保存到: {output_gif_path}")


dataset = DatasetADT()
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
dpath_show_aria_adt_gifs = os.path.join(preset.dpath_data_show, 'aria_adt_gifs')

os.makedirs(dpath_show_aria_adt_gifs, exist_ok=True)

total_ms = timedelta(minutes=1, seconds=57).total_seconds() * 1000

N = len(dataset)
ms_per_frame = int(total_ms / N)

MM_device = try_gpu()
pink_color = torch.tensor([1.0, 0.0, 1.0]).to(device=MM_device)

kname_ms_size_s = [
    ['pspnet640', 440, 640],
    ['hrnet640', 1295, 640],

    ['pspnet80', 54, 80],
    ['hrnet80', 220, 80],

    # ['fake40ms', 40, 640],
]

default_size = 1408 // 4

for kname, latency_ms, size_px in kname_ms_size_s:
    dpath_kname = os.path.join(dpath_show_aria_adt_gifs, kname)
    os.makedirs(dpath_kname, exist_ok=True)

    cur_seg_A_Bx1xHxW = torch.zeros(1, 1, default_size, default_size).to(device=MM_device)
    process_seg_A_Bx1xHxW = torch.zeros(1, 1, default_size, default_size).to(device=MM_device)

    delayed_frames = 0
    for i in trange(1000):
        img_RGB_3xHxW, F_HW_2, seg_A_1xHxW, _ = [v.to(device=MM_device) for v in dataset[i]]

        img_ds_RGB_Bx3xHxW = F.interpolate(img_RGB_3xHxW.unsqueeze(0), size=(default_size, default_size), mode='bilinear', align_corners=False)
        seg_ds_A_Bx1xHxW = F.interpolate(seg_A_1xHxW.unsqueeze(0), size=(size_px, size_px), mode='bilinear', align_corners=False)
        seg_ds_A_Bx1xHxW = F.interpolate(seg_ds_A_Bx1xHxW, size=(default_size, default_size), mode='bilinear', align_corners=False)

        if delayed_frames == 0:
            cur_seg_A_Bx1xHxW = process_seg_A_Bx1xHxW
            process_seg_A_Bx1xHxW = seg_ds_A_Bx1xHxW
            delayed_frames = latency_ms // ms_per_frame + (latency_ms % ms_per_frame > 0)
            print(f'delayed_frames = latency_ms / ms_per_frame = {latency_ms}/{ms_per_frame} = {(latency_ms / ms_per_frame):.2f} = {delayed_frames}')
        delayed_frames -= 1

        bind_RGBA_Bx3xHxW = segmentation_overlay(img_ds_RGB_Bx3xHxW, cur_seg_A_Bx1xHxW)

        idx_H, idx_W = (torch.round((F_HW_2 * (default_size - 1)))).to(dtype=torch.int64).tolist()
        bind_RGBA_Bx3xHxW[:, :, idx_H, :] = pink_color[None, :, None]  # 设置整个行的 RGBA 值
        bind_RGBA_Bx3xHxW[:, :, :, idx_W] = pink_color[None, :, None]  # 设置整个列的 RGBA 值

        fpath = os.path.join(dpath_kname, f'{kname}.{str(i).zfill(4)}.png')
        save_image(bind_RGBA_Bx3xHxW[0].detach().cpu(), fpath)

    gen_gif_from_pngs(dpath_kname, os.path.join(dpath_show_aria_adt_gifs, f'{kname}.gif'), interval_ms=ms_per_frame)
