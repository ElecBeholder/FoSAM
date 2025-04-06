# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import traceback
from typing import Any, List, Tuple, Type

import torch
import torch.nn.functional as F

from torch import nn, Tensor

import preset
from d_model.nn_A0_utils import init_weights_random
from d_model.nn_C1_mobilenetv2 import MobileNetV2
from utility.fctn import save_image

from utility.watch import watch_time
from l_sam.forveated_sam.efficient_sam_decoder_cls_token import MaskDecoder, PromptEncoder
from l_sam.forveated_sam.efficient_sam_encoder import ImageEncoderViT
from l_sam.forveated_sam.two_way_transformer import TwoWayAttentionBlock, TwoWayTransformer
import torchvision.models as models


def crop_and_resize_padding(batched_images, masks, size=80):
    B, C, H, W = batched_images.shape  # 批次，通道，高度，宽度
    cropped_images = []

    for i in range(B):
        # 获取第 i 个 mask 和 image
        mask = masks[i].squeeze(0)  # (H_mask, W_mask)
        image = batched_images[i]  # (C, H, W)

        # 将 mask 调整为与 image 相同的尺寸
        # 确保 mask 是 4D 张量 (1, 1, H_mask, W_mask)
        mask = mask.unsqueeze(0).unsqueeze(0).float()  # (1, 1, H_mask, W_mask)
        mask_resized = F.interpolate(mask, size=(H, W), mode='bilinear').squeeze(0).squeeze(0)  # (H, W)

        # 找出 mask > 0 的区域
        coords = torch.nonzero(mask_resized > 0)  # (N, 2), N 为满足条件的像素数

        if coords.shape[0] == 0:  # 如果没有有效区域，跳过
            cropped_images.append(F.interpolate(image.unsqueeze(0), size=(size, size)))
            continue

        # 计算边界框
        y_min = coords[:, 0].min()-50  # 所有 y 坐标的最小值
        x_min = coords[:, 1].min()-50  # 所有 x 坐标的最小值
        y_max = coords[:, 0].max()+50  # 所有 y 坐标的最大值
        x_max = coords[:, 1].max()+50  # 所有 x 坐标的最大值

        # 裁剪图像
        cropped = (image)[:, y_min:y_max + 1, x_min:x_max + 1]  # (C, h, w)

        # 计算缩放比例
        h, w = cropped.shape[1], cropped.shape[2]
        if h > w:
            new_h = size
            new_w = int(w * (size / h))
        else:
            new_w = size
            new_h = int(h * (size / w))

        # 调整大小到目标尺寸
        resized = F.interpolate(cropped.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)

        # 计算填充
        pad_h = (size - new_h) // 2
        pad_w = (size - new_w) // 2
        padding = (pad_w, size - new_w - pad_w, pad_h, size - new_h - pad_h)

        # 进行填充
        padded = F.pad(resized, padding, mode='constant', value=0)

        # 保存结果
        cropped_images.append(padded)

        # try:
        #     save_image(padded.squeeze(0), f"{preset.root_path}/l_sam/view/{i}.png")
        # except Exception as err:
        #     print(traceback.format_exc())

    # 拼接成批次形式
    return torch.cat(cropped_images, dim=0)  # (B, C, size, size)


def crop_and_resize(batched_images, masks, size=80):
    """
    将每张图像根据其 mask 的最小外接正方形进行裁剪，保证 mask 居中，然后统一 resize 到 (size, size)。

    Args:
        batched_images (Tensor): 形状 [B, C, H, W]，原始图像批次
        masks (Tensor): 形状 [B, 1, H_m, W_m] 或 [B, H_m, W_m]，每张图对应的mask
        size (int): 输出裁剪后再resize到的宽高尺寸
    Returns:
        Tensor: 形状 [B, C, size, size] 的裁剪并缩放后的结果
    """
    B, C, H, W = batched_images.shape
    cropped_images = []

    for i in range(B):
        # ========== 1) 预处理 mask，插值到与 image 同大小 ==========
        mask_i = masks[i]
        if mask_i.dim() == 3 and mask_i.size(0) == 1:
            # [1, H_m, W_m] -> [H_m, W_m]
            mask_i = mask_i.squeeze(0)
        # 现在 mask_i 形状是 [H_m, W_m]
        # 如果 H_m != H 或 W_m != W，需要插值到 [H, W]
        if mask_i.shape[0] != H or mask_i.shape[1] != W:
            mask_i = mask_i.unsqueeze(0).unsqueeze(0).float()  # [1, 1, H_m, W_m]
            mask_i = F.interpolate(mask_i, size=(H, W), mode='bilinear', align_corners=False)
            mask_i = mask_i.squeeze(0).squeeze(0)  # [H, W]

        # ========== 2) 找到 mask > 0 的坐标，并计算 bounding box ==========
        coords = torch.nonzero(mask_i > 0)  # [N, 2], 每行 (y, x)
        if coords.shape[0] == 0:
            # 如果没有有效区域，就简单 resize 整张图到 (size, size) 作为退化方案
            # 或者你也可选择返回空图
            fallback = F.interpolate(batched_images[i].unsqueeze(0), size=(size, size), mode='bilinear')
            cropped_images.append(fallback)
            continue

        y_min, x_min = coords[:, 0].min()-int(0.2*(coords[:, 0].max()-coords[:, 0].min())), coords[:, 1].min()-int(0.2*(coords[:, 1].max()-coords[:, 1].min()))
        y_max, x_max = coords[:, 0].max()+int(0.2*(coords[:, 0].max()-coords[:, 0].min())), coords[:, 1].max()+int(0.2*(coords[:, 1].max()-coords[:, 1].min()))

        # ========== 3) 计算使 bounding box 成为正方形，并让 mask 居中 ==========
        #   (a) 计算当前 bbox 中心和长宽
        bbox_h = (y_max - y_min + 1).item()
        bbox_w = (x_max - x_min + 1).item()
        center_y = (y_min + y_max).float() / 2.0
        center_x = (x_min + x_max).float() / 2.0

        #   (b) 需要的 side = max(bbox_h, bbox_w)
        side = max(bbox_h, bbox_w)

        #   (c) 让正方形 bbox 以 (center_y, center_x) 为中心
        half_side = side // 2  # 这里用整除，如果需要更精确可用别的策略
        # 为了简化，我们让上/左从 center - half_side 出发
        # 下/右为上/左 + side - 1
        new_y_min = int(center_y - half_side)
        new_x_min = int(center_x - half_side)
        new_y_max = new_y_min + side - 1
        new_x_max = new_x_min + side - 1

        # ========== 4) 边界 clamp ==========
        # 确保 [new_y_min, new_y_max] 落在 [0, H-1]
        if new_y_min < 0:
            new_y_min = 0
        if new_y_max >= H:
            new_y_max = H - 1
        # 确保 [new_x_min, new_x_max] 落在 [0, W-1]
        if new_x_min < 0:
            new_x_min = 0
        if new_x_max >= W:
            new_x_max = W - 1

        # 同时要保证实际抓到的区域大小和 side 尽量一致
        # 但是如果图像边缘不够，就可能比 side 小
        # 最终实际区域： [new_y_min : new_y_max+1, new_x_min : new_x_max+1]

        # ========== 5) 裁剪图像 ==========
        image_i = batched_images[i]  # shape (C, H, W)
        cropped = image_i[:, new_y_min:new_y_max + 1, new_x_min:new_x_max + 1]

        # ========== 6) 统一缩放到 (size, size) ==========
        resized = F.interpolate(cropped.unsqueeze(0), size=(size, size), mode='bilinear', align_corners=False)
        # shape: [1, C, size, size]

        # 记录结果
        cropped_images.append(resized)

        # ========== (可选) 保存可视化 ==========
        # 如果不需要可以去掉这几行
        # try:
        #     save_image(resized.squeeze(0), f"{preset.root_path}/l_sam/view/{i}.png")
        # except Exception as err:
        #     print(traceback.format_exc())

    # 拼成 batch
    return torch.cat(cropped_images, dim=0)  # shape: [B, C, size, size]


class FoveatedSam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
            self,
            image_encoder: ImageEncoderViT,
            prompt_encoder: PromptEncoder,
            decoder_max_num_input_points: int,
            mask_decoder: MaskDecoder,
            pixel_mean: List[float] = [0.485, 0.456, 0.406],
            pixel_std: List[float] = [0.229, 0.224, 0.225],
            class_num: int = 10,
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.decoder_max_num_input_points = decoder_max_num_input_points
        self.mask_decoder = mask_decoder
        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(1, 3, 1, 1), False
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(pixel_std).view(1, 3, 1, 1), False
        )

        # self.classify_model = models.mobilenet_v2(pretrained=True)
        # self.classify_model.classifier[1] = nn.Linear(self.classify_model.last_channel, class_num)

    # @watch_time
    @torch.jit.export
    def predict_masks(
            self,
            image_embeddings: torch.Tensor,
            batched_points: torch.Tensor,
            batched_point_labels: torch.Tensor,
            multimask_output: bool,
            input_h: int,
            input_w: int,
            output_h: int = -1,
            output_w: int = -1,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Predicts masks given image embeddings and prompts. This only runs the decoder.

        Arguments:
          image_embeddings: A tensor of shape [B, C, H, W] or [B*max_num_queries, C, H, W]
          batched_points: A tensor of shape [B, max_num_queries, num_pts, 2]
          batched_point_labels: A tensor of shape [B, max_num_queries, num_pts]
        Returns:
          A tuple of two tensors:
            low_res_mask: A tensor of shape [B, max_num_queries, 256, 256] of predicted masks
            iou_predictions: A tensor of shape [B, max_num_queries] of estimated IOU scores
        """

        batch_size, max_num_queries, num_pts, _ = batched_points.shape
        num_pts = batched_points.shape[2]
        rescaled_batched_points = self.get_rescaled_pts(batched_points, input_h, input_w)

        if num_pts > self.decoder_max_num_input_points:
            rescaled_batched_points = rescaled_batched_points[
                                      :, :, : self.decoder_max_num_input_points, :
                                      ]
            batched_point_labels = batched_point_labels[
                                   :, :, : self.decoder_max_num_input_points
                                   ]
        elif num_pts < self.decoder_max_num_input_points:
            rescaled_batched_points = F.pad(
                rescaled_batched_points,
                (0, 0, 0, self.decoder_max_num_input_points - num_pts),
                value=-1.0,
            )
            batched_point_labels = F.pad(
                batched_point_labels,
                (0, self.decoder_max_num_input_points - num_pts),
                value=-1.0,
            )

        sparse_embeddings = self.prompt_encoder(
            rescaled_batched_points.reshape(
                batch_size * max_num_queries, self.decoder_max_num_input_points, 2
            ),
            batched_point_labels.reshape(
                batch_size * max_num_queries, self.decoder_max_num_input_points
            ),
        )
        sparse_embeddings = sparse_embeddings.view(
            batch_size,
            max_num_queries,
            sparse_embeddings.shape[1],
            sparse_embeddings.shape[2],
        )

        low_res_masks, iou_predictions, cls_prediction = self.mask_decoder(
            image_embeddings,
            self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            multimask_output=multimask_output,
        )
        _, num_predictions, low_res_size, _ = low_res_masks.shape

        if output_w > 0 and output_h > 0:
            output_masks = F.interpolate(
                low_res_masks, (output_h, output_w), mode="bicubic"
            )
            output_masks = torch.reshape(
                output_masks,
                (batch_size, max_num_queries, num_predictions, output_h, output_w),
            )
        else:
            output_masks = torch.reshape(
                low_res_masks,
                (
                    batch_size,
                    max_num_queries,
                    num_predictions,
                    low_res_size,
                    low_res_size,
                ),
            )

        iou_predictions = torch.reshape(
            iou_predictions, (batch_size, max_num_queries, -1)
        )
        return output_masks, iou_predictions, cls_prediction

    def get_rescaled_pts(self, batched_points: torch.Tensor, input_h: int, input_w: int):
        return torch.stack(
            [
                torch.where(
                    batched_points[..., 0] >= 0,
                    batched_points[..., 0] * self.image_encoder.img_size / input_w,
                    -1.0,
                ),
                torch.where(
                    batched_points[..., 1] >= 0,
                    batched_points[..., 1] * self.image_encoder.img_size / input_h,
                    -1.0,
                ),
            ],
            dim=-1,
        )

    # @watch_time
    @torch.jit.export
    def get_image_embeddings(self, batched_images) -> torch.Tensor:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_images: A tensor of shape [B, 3, H, W]
        Returns:
          List of image embeddings each of of shape [B, C(i), H(i), W(i)].
          The last embedding corresponds to the final layer.
        """
        batched_images = self.preprocess(batched_images)
        return self.image_encoder(batched_images)

    def forward(
            self,
            batched_images: torch.Tensor,
            batched_points: torch.Tensor,
            batched_point_labels: torch.Tensor,
            scale_to_original_image_size: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_images: A tensor of shape [B, 3, H, W]
          batched_points: A tensor of shape [B, num_queries, max_num_pts, 2]
          batched_point_labels: A tensor of shape [B, num_queries, max_num_pts]

        Returns:
          A list tuples of two tensors where the ith element is by considering the first i+1 points.
            low_res_mask: A tensor of shape [B, 256, 256] of predicted masks
            iou_predictions: A tensor of shape [B, max_num_queries] of estimated IOU scores
        """
        batch_size, _, input_h, input_w = batched_images.shape

        image_embeddings = self.get_image_embeddings(batched_images)

        masks, iou_pred, cls_prediction = self.predict_masks(
            image_embeddings,
            batched_points,
            batched_point_labels,
            multimask_output=True,
            input_h=input_h,
            input_w=input_w,
            output_h=input_h if scale_to_original_image_size else -1,
            output_w=input_w if scale_to_original_image_size else -1,
        )

        # cls_pred = F.softmax(self.cls_prediction_head(cls_token_out), dim=1)

        # image_BxCxHxW = crop_and_resize(batched_images, masks.squeeze(1).squeeze(1), 224)
        #
        # masked_images_BxCxHxW = F.sigmoid(masks.squeeze(1)) * batched_images
        # B, C, H, W = masked_images_BxCxHxW.shape  # 批次，通道，高度，宽度
        #
        # for i in range(B):
        #     masked_image = masked_images_BxCxHxW[i]  # (C, H, W)
        #
        #     try:
        #         save_image(masked_image.squeeze(0), f"{preset.root_path}/l_sam/view/{i}.png")
        #     except Exception as err:
        #         print(traceback.format_exc())

        cls_pred = F.softmax(cls_prediction, dim=1)

        return masks, cls_pred, iou_pred

    # @watch_time
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        if (
                x.shape[2] != self.image_encoder.img_size
                or x.shape[3] != self.image_encoder.img_size
        ):
            x = F.interpolate(
                x,
                (self.image_encoder.img_size, self.image_encoder.img_size),
                mode="bilinear",
            )
        return (x - self.pixel_mean) / self.pixel_std


def build_foveated_sam(img_size, encoder_patch_embed_dim, encoder_num_heads, num_multimask_outputs=3, class_num=10, checkpoint_state_dict=None):
    encoder_patch_size = 16
    encoder_depth = 12
    encoder_mlp_ratio = 4.0
    encoder_neck_dims = [256, 256]
    decoder_max_num_input_points = 1
    decoder_transformer_depth = 2
    decoder_transformer_mlp_dim = 2048
    decoder_num_heads = 8
    decoder_upscaling_layer_dims = [64, 32]
    num_multimask_outputs = num_multimask_outputs
    iou_head_depth = 3
    iou_head_hidden_dim = 256
    activation = "gelu"
    normalization_type = "layer_norm"
    normalize_before_activation = False

    assert activation == "relu" or activation == "gelu"
    if activation == "relu":
        activation_fn = nn.ReLU
    else:
        activation_fn = nn.GELU

    image_encoder = ImageEncoderViT(
        img_size=img_size,
        patch_size=encoder_patch_size,
        in_chans=3,
        patch_embed_dim=encoder_patch_embed_dim,
        normalization_type=normalization_type,
        depth=encoder_depth,
        num_heads=encoder_num_heads,
        mlp_ratio=encoder_mlp_ratio,
        neck_dims=encoder_neck_dims,
        act_layer=activation_fn,
    )

    image_embedding_size = image_encoder.image_embedding_size
    encoder_transformer_output_dim = image_encoder.transformer_output_dim

    sam = FoveatedSam(
        image_encoder=image_encoder,
        prompt_encoder=PromptEncoder(
            embed_dim=encoder_transformer_output_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(img_size, img_size),
        ),
        decoder_max_num_input_points=decoder_max_num_input_points,
        mask_decoder=MaskDecoder(
            transformer_dim=encoder_transformer_output_dim,
            transformer=TwoWayTransformer(
                depth=decoder_transformer_depth,
                embedding_dim=encoder_transformer_output_dim,
                num_heads=decoder_num_heads,
                mlp_dim=decoder_transformer_mlp_dim,
                activation=activation_fn,
                normalize_before_activation=normalize_before_activation,
            ),
            num_multimask_outputs=num_multimask_outputs,
            activation=activation_fn,
            normalization_type=normalization_type,
            normalize_before_activation=normalize_before_activation,
            iou_head_depth=iou_head_depth - 1,
            iou_head_hidden_dim=iou_head_hidden_dim,
            upscaling_layer_dims=decoder_upscaling_layer_dims,
            class_num=class_num

        ),
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
        class_num=class_num,
    )
    if checkpoint_state_dict is not None:
        sam.load_state_dict(checkpoint_state_dict, strict=False)

    return sam
