import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

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


    merged_BxKxHxW  =  torch.where(seg_Bx1xHxW > 0, foreground, background)

    return merged_BxKxHxW


def merge_seg_label(seg_Bx1xHxW, cls_B):
    B = cls_B.shape[0]
    _, _, H, W = seg_Bx1xHxW.shape

    seg_Bx1xHxW = (seg_Bx1xHxW > 0.5).to(torch.int64)
    cls_Bx1xHxW = cls_B[:, None, None, None].expand(-1, -1, H, W)

    # merged_BxHxW = torch.where(seg_Bx1xHxW > 0, cls_Bx1xHxW, 0).squeeze(1)
    merged_BxHxW = torch.where(seg_Bx1xHxW > 0, cls_Bx1xHxW + 1, 0).squeeze(1)
    merged_BxHxW = merged_BxHxW.long()

    return merged_BxHxW


# def forward(self, conv_out, segSize=None, res=None):
#     conv5 = conv_out[-1]
#     #print(conv5.shape) torch.Size([B, 720, 80, 80])
#     x = self.cbr(conv5)
#     x = self.conv_last(x)
#     if res is not None:
#         # not enter here
#         x += res
#     x = nn.functional.sigmoid(x).float() - 0.5
#     #print('x shape', x.shape)
    
#     cls_pred = self.cls_net(conv5)  # [20, 20]
#     B, K = cls_pred.shape
#     _, _, H, W = x.shape
#     cls_pred = cls_pred.unsqueeze(-1).unsqueeze(-1)  #[20, 20, 1, 1]
#     cls_pred = cls_pred.expand(-1, -1, H, W) #[20,20,64,128]
#     #print('cls_pred = cls_pred.expand(-1, -1, H, W) shape: ', cls_pred.shape)
#     #print('cls_pred[:,:0:1,:,:].shape', cls_pred[:,0:1,:,:].shape)
#     cls_pred_new = cls_pred.clone()
#     cls_pred_new[:, :-1, :, :] = cls_pred[:, :-1, :, :] * x.cuda()
#     #print('cls_pred[:,:0:1,:,:] = cls_pred[:,:0:1,:,:] * x.cuda()', cls_pred.shape)
#     #print('C1 cls_pred shape', cls_pred_new.shape)
#     #print(x.shape)
#     return cls_pred_new



# def merge_seg_cls_new(seg_Bx1xHxW, cls_BxK):
#     B, K = cls_BxK.shape
#     _, _, H, W = seg_Bx1xHxW.shape

#     cls_BxKxHxW = cls_BxK[:, :, None, None].expand(-1, -1, H, W)

#     cls_BxKxHxW_new = cls_BxKxHxW.clone()
#     cls_BxKxHxW_new[:, 1:, :, :] = cls_BxKxHxW[:, 1:, :, :] * seg_Bx1xHxW.to(cls_BxKxHxW.device)
#     # cls_BxKxHxW_new = F.softmax(cls_BxKxHxW_new, dim=1)


#     # background = torch.zeros((B, 1+K, H, W)).to(cls_BxKxHxW.device)
#     # background[:,0,:,:] = 1.0
#     # foreground = torch.cat([torch.zeros((B, 1, H, W)).to(cls_BxKxHxW.device),cls_BxKxHxW],dim=1).to(cls_BxKxHxW.device)


#     # merged_BxKxHxW  =  torch.where(seg_Bx1xHxW > 0.5, foreground, background)

#     return cls_BxKxHxW_new

def merge_seg_cls_new(seg_Bx1xHxW, cls_BxK):
    B, K = cls_BxK.shape
    _, _, H, W = seg_Bx1xHxW.shape

    cls_BxKxHxW = cls_BxK[:, :, None, None].expand(-1, -1, H, W)

    cls_BxKxHxW_new = cls_BxKxHxW.clone()
    cls_BxKxHxW_new[:, 0:1, :, :] = cls_BxKxHxW[:, 0:1, :, :] * (-seg_Bx1xHxW).to(cls_BxKxHxW.device)
    # cls_BxKxHxW_new = F.softmax(cls_BxKxHxW_new, dim=1)


    # background = torch.zeros((B, 1+K, H, W)).to(cls_BxKxHxW.device)
    # background[:,0,:,:] = 1.0
    # foreground = torch.cat([torch.zeros((B, 1, H, W)).to(cls_BxKxHxW.device),cls_BxKxHxW],dim=1).to(cls_BxKxHxW.device)


    # merged_BxKxHxW  =  torch.where(seg_Bx1xHxW > 0.5, foreground, background)

    return cls_BxKxHxW_new

def merge_seg_label_new(seg_Bx1xHxW, cls_B):
    B = cls_B.shape[0]
    _, _, H, W = seg_Bx1xHxW.shape

    seg_Bx1xHxW = (seg_Bx1xHxW > 0.5).to(torch.int64)
    cls_Bx1xHxW = cls_B[:, None, None, None].expand(-1, -1, H, W)

    # merged_BxHxW = torch.where(seg_Bx1xHxW > 0, cls_Bx1xHxW, 0).squeeze(1)
    merged_BxHxW = torch.where(seg_Bx1xHxW > 0, cls_Bx1xHxW + 1, 0).squeeze(1)
    merged_BxHxW = merged_BxHxW.long()

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


