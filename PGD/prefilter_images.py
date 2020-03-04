'''
Input pre-processing if any
'''

import os
import torch

import numpy as np
import argparse
from scipy import ndimage

#--------------------------------------------------------------

print("PID = %d\n" % os.getpid())
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda =  torch.cuda.is_available()
#--------------------------------------------------------------

parser = argparse.ArgumentParser(description='Input pre-processingh')

parser.add_argument('--model', default='vgg16', help='The target model') # vgg16 resnet18
parser.add_argument('--samples', default=1000, type=int, help='The number of image samples to attack.')

parser.add_argument('--is_median', default=True, type=int, help='If apply median prefintering')
parser.add_argument('--is_mean',   default=False, type=int, help='If apply mean prefintering')
parser.add_argument('--is_custom',  default=False, type=int, help='If apply custom prefintering')

parser.add_argument("--filt_size",  default=3, type=int, help="Pre-filtering: window size for median and custom filters.")
parser.add_argument("--filt_sigma", default=0.5, type=int, help="Pre-filtering sigma for the mean filtering.")

args = parser.parse_args()

# =================================================================================
def makeDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def prefiltImage(img, thr=1, b=3, IMAGE_SIZE=32, N_CHANELS=3):

    st = round((b-1)/2)
    res_img = np.zeros(img.shape)

    for n in range(N_CHANELS):
        if args.is_mean:
            res_img[n] = ndimage.gaussian_filter(img[n], sigma=args.filt_sigma)

        elif args.is_median:
            res_img[n] = ndimage.median_filter(img[n], size=args.filt_size)

        elif args.is_custom:
            x = np.pad(img[n], st, mode="constant", constant_values=-100)
            for r in range(1, IMAGE_SIZE + 1):
                for c in range(1, IMAGE_SIZE + 1):
                    blk = x[r - st:r + st + 1, c - st:c + st + 1]
                    y = blk.reshape(-1)
                    y = np.delete(y, np.where(y == -100))
                    m = np.median(y)

                    if abs(x[r, c] - m) >= thr:
                        blk[1, 1] = -100
                        blk = blk.reshape(-1)
                        blk = np.delete(blk, np.where(blk == -100))

                        res_img[n, r - 1, c - 1] = np.sum(blk) / blk.shape[0]
                    else:
                        res_img[n, r - 1, c - 1] = x[r, c]
    return res_img

# ===============================================================================
if __name__ == '__main__':

    print(args)

    if args.is_mean:
        pref = "mean_filt"
    elif args.is_median:
        pref = "median_filt"
    elif args.is_custom:
        pref = "custom_filt"

    data_dir_ = "data/attacked/cifar/%s/PGD" % args.model
    save_dir_ = "%s/%s" % (data_dir_, pref)
    makeDir(save_dir_)

    # --- prefilter ---------------------------------------------------------------------
    for n in range(1, args.samples+1):
        input_org = np.load("%s/org_img_%d.npy" % (data_dir_, n)).squeeze()
        input_adv = np.load("%s/adv_img_%d.npy" % (data_dir_, n)).squeeze()
        target = np.load("%s/targets_%d.npy" % (data_dir_, n))

        input_org = prefiltImage(input_org)
        input_adv = prefiltImage(input_adv)

        np.save("%s/org_img_%d.npy" % (save_dir_, n), input_org)
        np.save("%s/adv_img_%d.npy" % (save_dir_, n), input_adv)
        np.save("%s/targets_%d.npy" % (save_dir_, n), target)