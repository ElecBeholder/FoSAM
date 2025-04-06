import torch
import torchvision.transforms as transforms

def cd_cdf(tensor_x: torch.Tensor) -> torch.Tensor:
    """
    Cauchy Distribution CDF
    """
    return torch.arctan(tensor_x) / torch.pi + 0.5


def a_gd_cdf(tensor_x: torch.Tensor, a_gd_cdf_constant=torch.sqrt(torch.tensor(2. / torch.pi))) -> torch.Tensor:
    """
    Approximate Gaussian Distribution CDF
    """

    return torch.tanh(a_gd_cdf_constant * tensor_x) / 2. + 0.5


# def standardize_BxCxHxW(img_BxCxHxW: torch.Tensor):
#     B, C, H, W = img_BxCxHxW.shape
#     res_img_BxCxHxW = img_BxCxHxW
#     if H * W > 1:
#         value_mean = torch.mean(img_BxCxHxW, dim=[-2, -1], keepdim=True)
#         value_std = torch.std(img_BxCxHxW, dim=[-2, -1], keepdim=True)
#         res_img_BxCxHxW = (img_BxCxHxW - value_mean) / value_std
#     return res_img_BxCxHxW

def standardize_BxCxHxW(img_BxCxHxW: torch.Tensor):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    return normalize(img_BxCxHxW)


def scale01_BxCxHxW(img_BxCxHxW: torch.Tensor):
    B, C, H, W = img_BxCxHxW.shape
    res_img_BxCxHxW = img_BxCxHxW
    if H * W > 1:
        value_max = torch.amax(img_BxCxHxW, dim=[-2, -1], keepdim=True)
        value_min = torch.amin(img_BxCxHxW, dim=[-2, -1], keepdim=True)

        img_BxCxHxW[:] = 1.0 - (value_max - img_BxCxHxW) / (value_max - value_min)

    return res_img_BxCxHxW


def merge_seg_cls(seg_Bx1xHxW, cls_BxK):
    B, K = cls_BxK.shape
    _, _, H, W = seg_Bx1xHxW.shape

    cls_BxKxHxW = cls_BxK[:, :, None, None].expand(-1, -1, H, W)


    background = torch.zeros((B, 1+K, H, W)).to(cls_BxKxHxW.device)
    background[:,0,:,:] = 1.0
    foreground = torch.cat([torch.zeros((B, 1, H, W)).to(cls_BxKxHxW.device),cls_BxKxHxW],dim=1).to(cls_BxKxHxW.device)


    merged_BxKxHxW  = seg_Bx1xHxW * foreground + background * (1-seg_Bx1xHxW)

    return merged_BxKxHxW


def merge_seg_label(seg_Bx1xHxW, cls_B):
    B = cls_B.shape[0]
    _, _, H, W = seg_Bx1xHxW.shape

    seg_Bx1xHxW = (seg_Bx1xHxW > 0.5).to(torch.int64)
    cls_Bx1xHxW = cls_B[:, None, None, None].expand(-1, -1, H, W)

    merged_BxHxW = (seg_Bx1xHxW * (cls_Bx1xHxW + 1)).to(torch.int64).squeeze(1)

    return merged_BxHxW

if __name__ == '__main__':
    # B, C, H, W = 4, 3, 128, 128
    #
    # # Generate a random tensor with size BxCxHxW
    # img_BxCxHxW = torch.rand((B, C, H, W))
    # x = standardize_BxCxHxW(img_BxCxHxW)
    B, K, H, W = 2, 5, 80, 80  # Batch大小、类别数量、图像尺寸

    # 生成随机分割掩码 (Bx1xHxW)，二值化
    seg_Bx1xHxW = torch.sigmoid(torch.rand(B, 1, H, W))

    # 生成随机分类分数 (BxK)，假设分数在0到1之间
    cls_BxK = torch.rand(B, K)
    cls_B = torch.argmax(cls_BxK, dim=1)

    merged = merge_seg_cls(seg_Bx1xHxW, cls_BxK)

    merged_label = merge_seg_label(seg_Bx1xHxW, cls_B)

# if __name__ == '__main__':
#     B, C, H, W = 4, 3, 128, 128

#     # Generate a random tensor with size BxCxHxW
#     img_BxCxHxW = torch.rand((B, C, H, W))
#     x = standardize_BxCxHxW(img_BxCxHxW)
