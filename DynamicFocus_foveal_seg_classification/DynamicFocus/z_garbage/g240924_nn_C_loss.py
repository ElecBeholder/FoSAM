import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from matplotlib import pyplot as plt
from torch.autograd import Variable

from d_model.nn_B0_deformed_sampler import get_grid_Bx2xHSxWS
from utility.plot_tools import plt_imgshow, plt_multi_imgshow
from utility.torch_tools import interpolate_int


def weighted_cosine_loss(output_softmax, target):
    """
    Computes weighted cosine loss by sampling the same number of samples from each class
    to balance class imbalance.

    Parameters:
        output_softmax: Tensor of shape (B, K, H, W), the softmax outputs of the model.
        target: Tensor of shape (B, H, W), the integer ground truth labels (int64).

    Returns:
        loss: Scalar, the computed cosine loss.
    """
    eps = 1e-8  # Small value to prevent division by zero

    B, K, H, W = output_softmax.shape
    N_samples = B * H * W  # Total number of samples

    # Flatten tensors to shape (N_samples, K) for output_softmax and (N_samples,) for target
    output_flat = output_softmax.permute(0, 2, 3, 1).reshape(-1, K)  # Shape: (N_samples, K)
    target_flat = target.reshape(-1)  # Shape: (N_samples,)

    # Convert target to one-hot encoding
    target_one_hot = torch.nn.functional.one_hot(target_flat, num_classes=K).float()  # Shape: (N_samples, K)

    # Compute the number of samples in each class
    class_counts = target_one_hot.sum(dim=0)  # Shape: (K,)

    # Find the classes that are present
    classes_present = (class_counts > 0)
    classes_indices = torch.nonzero(classes_present).squeeze()  # Indices of present classes

    num_classes_present = classes_indices.numel()  # Number of classes present
    avg_samples_per_class = N_samples // num_classes_present  # Average samples per class

    # Sample avg_samples_per_class samples from each class
    sampled_indices = []
    for k in classes_indices:
        k = k.item()  # Ensure k is an integer

        # Get indices of samples belonging to class k
        class_indices = torch.nonzero(target_flat == k).squeeze()
        num_samples_in_class = class_indices.numel()

        if num_samples_in_class == 0:
            continue  # Skip classes with zero samples

        if num_samples_in_class >= avg_samples_per_class:
            # Randomly select samples without replacement
            perm = torch.randperm(num_samples_in_class)[:avg_samples_per_class]
            sampled_class_indices = class_indices[perm]
        else:
            # Randomly sample with replacement
            if num_samples_in_class == 1:
                sampled_class_indices = class_indices.repeat(avg_samples_per_class)
            else:
                rand_indices = torch.randint(0, num_samples_in_class, (avg_samples_per_class,))
                sampled_class_indices = class_indices[rand_indices]

        sampled_indices.append(sampled_class_indices)

    if len(sampled_indices) == 0:
        raise ValueError("No samples found for any class.")

    # Concatenate sampled indices
    sampled_indices = torch.cat(sampled_indices)

    # Get sampled outputs and targets
    output_sampled = output_flat[sampled_indices]
    target_sampled = target_one_hot[sampled_indices]

    # Normalize outputs and targets
    output_norms = torch.norm(output_sampled, dim=1, keepdim=True) + eps
    output_normalized = output_sampled / output_norms

    target_norms = torch.norm(target_sampled, dim=1, keepdim=True) + eps
    target_normalized = target_sampled / target_norms

    # Compute cosine similarity
    cosine_similarity = torch.sum(output_normalized * target_normalized, dim=1)

    # Compute cosine loss
    loss = 0.5 * (1 - torch.mean(cosine_similarity))

    return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, eps=1e-7, size_average=True, one_hot=True, ignore_label=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.size_average = size_average
        self.one_hot = one_hot
        self.ignore = ignore_label

    # def one_hot(index, classes):
    #     # index is not flattened (pypass ignore) ############
    #     # size = index.size()[:1] + (classes,) + index.size()[1:]
    #     # view = index.size()[:1] + (1,) + index.size()[1:]
    #     #####################################################
    #     # index is flatten (during ignore) ##################
    #     size = index.size()[:1] + (classes,)
    #     view = index.size()[:1] + (1,)
    #     #####################################################
    #
    #     # mask = torch.Tensor(size).fill_(0).to(device)
    #     mask = torch.Tensor(size).fill_(0).cuda()
    #     index = index.view(view)
    #     ones = 1.
    #
    #     return mask.scatter_(1, index, ones)

    def forward(self, input, target):
        # ph, pw = input.size(2), input.size(3)
        # h, w = target.size(1), target.size(2)
        # if ph != h or pw != w:
        #     input = F.upsample(input=input, size=(h, w), mode='bilinear')
        # pred = F.softmax(input, dim=1).to(input.device)
        # # pred_t = pred.gather(1, target.unsqueeze(1))
        # pred_t = pred[:,0,:,:]
        #
        # mask = target != self.ignore
        # pred_t_mask = pred_t.clone()
        # # pred_t_mask = pred_t.clone().masked_select(mask)
        # pred_t_mask = pred_t_mask.clamp(self.eps, 1. - self.eps)
        # pixel_loss = pred_t_mask
        # # pixel_loss = (-1 * (1 - pred_t_mask) ** self.gamma) * (pred_t_mask.log())
        # return pixel_loss.mean()

        '''
        only support ignore at 0
        '''
        ph, pw = input.size(2), input.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            input = F.upsample(input=input, size=(h, w), mode='bilinear')
        # print(target)
        B, C, H, W = input.size()
        input = input.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
        target = target.view(-1)
        # print('max_target', max(target))
        # print('min_target', min(target))
        if self.ignore is not None:
            # if target.sum() == 0:
            #     target[0] = 1
            # if min(target) != 0:
            #     target[-1] = 0
            valid = (target != self.ignore)
            # input = input[valid]
            # target = target[valid]
            masked_input = torch.zeros((target[valid].size()[0], input.size(1))).to(input.device)
            for c in range(input.size(1)):
                masked_input[:, c] = input[:, c].masked_select(valid)
            input = masked_input
            # input = input.clone().masked_select(valid)
            target = target.clone().masked_select(valid)

        # if self.one_hot: target = one_hot(target, input.size(1))

        if self.one_hot:
            index = target.clone()
            classes = input.size(1)
            size = index.size()[:1] + (classes,)
            view = index.size()[:1] + (1,)
            mask = (torch.Tensor(size).fill_(0)).to(input.device)
            index = index.view(view)
            ones = 1.
            target_local = mask.scatter_(1, index, ones)
        probs = F.softmax(input, dim=1)

        # probs = input.clone()
        # print('probs', probs)
        # print('target_local', target_local)
        probs = (probs * target_local).sum(1)
        probs = probs.clamp(self.eps, 1. - self.eps)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)
        # print(1 - probs)
        # print(torch.pow((1 - probs), self.gamma))
        # print(log_p)
        # print(-(torch.pow((1 - probs), self.gamma)) * log_p)
        batch_loss = (-(torch.pow((1 - probs), self.gamma)) * log_p)
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
            # print('got average batch_loss:', loss)
        else:
            loss = batch_loss.sum()
        # loss = loss.to(input.device)
        # print(loss)
        return loss


class EdgeLoss(nn.Module):
    def __init__(self, K, kernel_size, H, W, downsample_factor=4):
        super(EdgeLoss, self).__init__()
        self.K = K
        self.kernel_size = kernel_size
        self.kernel_pad = kernel_size // 2
        self.downsample_factor = downsample_factor

        self.H = H
        self.W = W
        self.HS = self.H // self.downsample_factor
        self.WS = self.W // self.downsample_factor

        self.downsample_degree = int(np.log2(downsample_factor))

        self.gaussian_blur = T.GaussianBlur(kernel_size=3, sigma=1)
        self.edge_kernel_3x3 = torch.Tensor([[-1, -1, -1],
                                             [-1, 8, -1],
                                             [-1, -1, -1]])

        self.edge_pad = 1
        self.edge_step = 1
        self.edge_size = 3

        self.mseloss_fctn = nn.MSELoss()

    def forward(self, xs_pred_BxCxHSxWS: torch.Tensor, x_BxCxHxW: torch.Tensor, y_BxHxW: torch.Tensor, dm_pred_Bx1xHSxWS: torch.Tensor):
        y_BxHSxWS = interpolate_int(y_BxHxW, degree=self.downsample_degree)

        y_BxKxHSxWS = F.one_hot(y_BxHSxWS, num_classes=self.K).permute(0, 3, 1, 2)

        y_BxKxHSPxWSP = F.pad(y_BxKxHSxWS, (self.edge_pad, self.edge_pad, self.edge_pad, self.edge_pad), "replicate")
        y_gaussian_BxKxHSPxWSP = self.gaussian_blur(y_BxKxHSPxWSP)

        y_BxKxHSxHSx3x3 = y_gaussian_BxKxHSPxWSP.unfold(-2, self.edge_size, self.edge_step).unfold(-2, self.edge_size, self.edge_step)

        kernel_3x3 = self.edge_kernel_3x3.to(device=y_BxKxHSxHSx3x3.device)

        dm_target_Bx1xHSxHS = torch.sum(torch.abs(torch.sum(y_BxKxHSxHSx3x3 * kernel_3x3[None, None, None, None, :, :], dim=[-2, -1])), dim=1, keepdim=True) / 16.0

        ls_mse = self.mseloss_fctn(dm_pred_Bx1xHSxWS, dm_target_Bx1xHSxHS)
        dm_target_Bx1xHSPxWSP = F.pad(dm_target_Bx1xHSxHS, (self.kernel_pad, self.kernel_pad, self.kernel_pad, self.kernel_pad), mode='replicate')

        grid_target_BxHSxWSx2 = get_grid_Bx2xHSxWS(dm_target_Bx1xHSPxWSP, self.HS, self.WS, kernel_size=self.kernel_size)

        xs_target_Bx3xHSxWS = F.grid_sample(x_BxCxHxW, grid_target_BxHSxWSx2, mode='nearest', align_corners=False)
        B, HS, WS = y_BxHSxWS.shape
        # images = []
        # titles = []
        # for b in range(B):
        #     images.extend([dm_pred_Bx1xHSxWS[b], dm_target_Bx1xHSxHS[b], xs_pred_BxCxHSxWS[b], xs_target_Bx3xHSxWS[b]])
        #     titles.extend(["dm_pred_Bx1xHSxWS", "dm_target_Bx1xHSxHS", "xs_pred_BxCxHSxWS", "xs_target_Bx3xHSxWS"])
        #
        # plt_multi_imgshow(images, titles, row_col=(4, 4))
        #
        # plt.show(block=True)

        return ls_mse


class FocalLossStd(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLossStd, self).__init__()

        self.focal_loss = FocalLoss(gamma=gamma, size_average=size_average)

    def forward(self, ys_pred_gs_BxKxHSxWS, ys_real_gs_BxHSxWS, x_Bx3xHxW, x_Bx2, y_real_BxHxW, grid_pred_BxHSxWSx2, xs_pred_gs_Bx3xHSxWS, dm_pred_Bx1xHSxWS):
        return self.focal_loss(ys_pred_gs_BxKxHSxWS, ys_real_gs_BxHSxWS)


class ObjDeformedJointLossStd(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(DoubleMSELossStd, self).__init__()

        self.focal_loss = FocalLoss(gamma=gamma, size_average=size_average)
        self.mse_loss = nn.MSELoss()

    def forward(self, ys_pred_gs_BxKxHSxWS, ys_real_gs_BxHSxWS, x_Bx3xHxW, x_Bx2, y_real_BxHxW, grid_pred_BxHSxWSx2, xs_pred_gs_Bx3xHSxWS, dm_pred_Bx1xHSxWS):
        B, K, HS, WS = ys_pred_gs_BxKxHSxWS.shape
        B, _, H, W = x_Bx3xHxW.shape

        downsample_factor = H // HS
        downsample_degree = int(np.log2(downsample_factor))

        y_real_objmap_Bx1xHxW = (y_real_BxHxW < K - 1).to(dtype=torch.float32).unsqueeze(1)

        y_real_objmap_Bx1xHSxWS = F.interpolate(y_real_objmap_Bx1xHxW, size=(HS, WS), mode='bilinear', align_corners=True)

        # plt_multi_imgshow([dm_pred_Bx1xHSxWS[0], y_real_objmap_Bx1xHSxWS[0]], titles=['dm_pred_Bx1xHSxWS', 'y_real_objmap_Bx1xHSxWS'], row_col=(1, 2))
        # plt.show(block=True)

        ls_focal = self.focal_loss(ys_pred_gs_BxKxHSxWS, ys_real_gs_BxHSxWS)
        ls_mse = self.mse_loss(dm_pred_Bx1xHSxWS, y_real_objmap_Bx1xHSxWS)

        weights = [0.5, 0.5]

        print(f"ls_focal:{weights[0]} {ls_focal}")
        print(f"ls_mse:{weights[1]} {ls_mse}")

        ls_res = weights[0] * ls_focal + weights[1] * ls_mse
        return ls_res


class DoubleMSELossStd(nn.Module):
    def __init__(self):
        super(DoubleMSELossStd, self).__init__()

        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, ys_pred_gs_BxKxHSxWS, ys_real_gs_BxHSxWS, x_Bx3xHxW, x_Bx2, y_real_BxHxW, grid_pred_BxHSxWSx2, xs_pred_gs_Bx3xHSxWS, dm_pred_Bx1xHSxWS):
        B, K, HS, WS = ys_pred_gs_BxKxHSxWS.shape
        B, _, H, W = x_Bx3xHxW.shape

        downsample_factor = H // HS
        downsample_degree = int(np.log2(downsample_factor))

        y_real_objmap_Bx1xHxW = (y_real_BxHxW < K - 1).to(dtype=torch.float32).unsqueeze(1)

        y_real_objmap_Bx1xHSxWS = F.interpolate(y_real_objmap_Bx1xHxW, size=(HS, WS), mode='bilinear', align_corners=True)

        # plt_multi_imgshow([dm_pred_Bx1xHSxWS[0], y_real_objmap_Bx1xHSxWS[0]], titles=['dm_pred_Bx1xHSxWS', 'y_real_objmap_Bx1xHSxWS'], row_col=(1, 2))
        # plt.show(block=True)

        ls_mse_zoom = self.ce_loss(ys_pred_gs_BxKxHSxWS, ys_real_gs_BxHSxWS)
        ls_mse_dens = self.ce_loss(dm_pred_Bx1xHSxWS, y_real_objmap_Bx1xHSxWS.to(dtype=torch.float64))

        weights = [0.5, 0.5]
        print()
        print(f"ls_mse_zoom:{weights[0]} {ls_mse_zoom}")
        print(f"ls_mse_dens:{weights[1]} {ls_mse_dens}")

        ls_res = weights[0] * ls_mse_zoom + weights[1] * ls_mse_dens
        return ls_res


if __name__ == '__main__':
    pass

    B = 5
    C = 3
    H = 256
    W = 512
    downsample_factor = 4
    HS = H // downsample_factor
    WS = W // downsample_factor
    K = 41
    kernel_size = 64 + 1

    xs_pred_BxCxHSxWS = torch.randn(B, C, HS, WS)
    x_BxCxHxW = torch.randn(B, C, H, W)
    y_BxHxW = torch.randint(0, 41, (B, H, W))

    edgeloss_fctn = EdgeLoss(K=K, kernel_size=kernel_size, H=H, W=W, downsample_factor=downsample_factor)
    focaloss_fctn = FocalLoss(gamma=2)

    ls_edge = edgeloss_fctn(xs_pred_BxCxHSxWS, x_BxCxHxW, y_BxHxW)
    ls_focal = focaloss_fctn(xs_pred_BxCxHSxWS, y_BxHxW)
