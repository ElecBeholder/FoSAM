import os, os.path, random
import json
import torch
from torch.nn import functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
# import albumentations as A

def img_transform(img):
    # 0-255 to 0-1
    img = np.float32(np.array(img)) / 255.
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img.copy())
    return img

def imresize(im, size, interp='bilinear'):
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return im.resize(size, resample)

def b_imresize(im, size, interp='bilinear'):
    return F.interpolate(im, size, mode=interp)



