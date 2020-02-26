import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# ===================================================

def makeDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def toDevice(Obj):
    if torch.cuda.is_available():
        Obj = Obj.cuda()

    return Obj

def normalize(img, IMSIZE=32):

    img = np.reshape(img, (-1, IMSIZE, IMSIZE))
    img = np.swapaxes(img, 1, 0)
    img = np.swapaxes(img, 2, 1)

    img = img - np.min(img, (0,1))
    img = img / np.max(img, (0,1))

    return img


def preFiltering(img, thr=1, b=3, IMAGE_SIZE=32, N_CHANELS=3, filt_sigma=1, filt_size=3):

    is_mean = False
    is_median = True

    st = round((b-1)/2)
    if len(img.shape) > 3:
        s  = img.shape[0]
    else:
        s = 1
    img = img.reshape((N_CHANELS, IMAGE_SIZE ,IMAGE_SIZE))

    res_img = np.zeros(img.shape)
    for n in range(N_CHANELS):
        
        if is_mean:
            x = ndimage.gaussian_filter(img[n], sigma=filt_sigma)
            res_img[n] = x

        elif is_median:
            x = ndimage.median_filter(img[n], size=filt_size)
            res_img[n] = x
        else:
            x = np.pad(img[n], st, mode="constant", constant_values=-100)
            for r in range(1,IMAGE_SIZE+1):
                for c in range(1,IMAGE_SIZE+1):
                    blk = x[r-st:r+st+1, c-st:c+st+1]
                    y = blk.reshape(-1)
                    y = np.delete(y, np.where(y == -100))
                    m = np.median(y)

                    if abs(x[r,c] - m) >= thr:
                        blk[1,1]  = -100
                        blk = blk.reshape(-1)
                        blk = np.delete(blk, np.where(blk == -100))

                        res_img[n, r-1, c-1] = np.sum(blk)/blk.shape[0]
                    else:
                        res_img[n,r-1,c-1] = x[r,c]

    return res_img.reshape((s, N_CHANELS, IMAGE_SIZE ,IMAGE_SIZE))
