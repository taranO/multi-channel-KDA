from scipy.fftpack import dct, idct
import os
import shutil
import glob
import numpy as np
from sys import version_info

import matplotlib.pyplot as plt


# ======================================================================================================================

def directPermutation(IMAGE_SIZE = 28):
    return np.random.permutation(IMAGE_SIZE * IMAGE_SIZE)

def inversePermutation(permutation):
    return np.argsort(permutation)

def applyPermutation(data, permutation, IMAGE_SIZE=28, N_CHANELS=1):
    dim = data.shape

    data = np.reshape(data, (-1, IMAGE_SIZE ** 2, N_CHANELS))
    data = data[:, permutation, :]

    return np.reshape(data, dim)

def signPermutation(IMAGE_SIZE = 28, subband="", is_zero=False):
    if is_zero:
        permutation = np.zeros((1, IMAGE_SIZE**2))
    else:
        permutation = np.random.normal(size=IMAGE_SIZE**2)
        permutation[permutation >= 0] = 1
        permutation[permutation != 1] = -1

    if subband == "d": # D - diagonal
        permutation = np.reshape(permutation, (IMAGE_SIZE, IMAGE_SIZE))
        permutation[0:IMAGE_SIZE//2, :] = 1
        permutation[:, 0:IMAGE_SIZE // 2] = 1

    elif subband == "v": # V - vertical
        permutation = np.reshape(permutation, (IMAGE_SIZE, IMAGE_SIZE))
        permutation[:, 0:IMAGE_SIZE // 2] = 1
        permutation[IMAGE_SIZE//2:IMAGE_SIZE, :] = 1

    elif subband == "h": # H - horizontal
        permutation = np.reshape(permutation, (IMAGE_SIZE, IMAGE_SIZE))
        permutation[0:IMAGE_SIZE//2, :] = 1
        permutation[:, IMAGE_SIZE//2:IMAGE_SIZE] = 1

    elif subband == "dhv":
        permutation = np.reshape(permutation, (IMAGE_SIZE, IMAGE_SIZE))
        permutation[0:IMAGE_SIZE//2, 0:IMAGE_SIZE//2] = 1

    return np.reshape(permutation, (IMAGE_SIZE**2))

def applySignPermutation(data, permutation, IMAGE_SIZE=28, is_denoise=False):
    dim = data.shape

    data = np.reshape(data, (-1, IMAGE_SIZE ** 2))
    data = np.multiply(data, np.tile(permutation, (data.shape[0], 1)))


    return np.reshape(data, dim)


def applyDCTPermutation(data, permutation, IMAGE_SIZE=28, N_CHANELS=1, is_denoise=False):
    n = data.shape[0]
    for i in range(n):
        for c in range(N_CHANELS):

            xdct = dct(dct(data[i, c, :, :]).T)

            if is_denoise:
                img = np.copy(xdct)
                mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
                mask[0:IMAGE_SIZE//2, 0:IMAGE_SIZE//2] = 1
                img = np.multiply(img, mask)
                # denoising in coordinate domain
                i_img = idct(idct(img).T)

                i_img -= np.min(i_img)
                i_img /= np.max(i_img)
                i_img *= 255
                i_img = i_img.astype(np.uint8)

                # d_img = cv2.fastNlMeansDenoising(i_img, 5, 7, 21)
                d_img = ndimage.median_filter(i_img, 3)
                d_img = d_img/255
                d_img -= 0.5
                # replase the LL components
                d_img_dct = dct(dct(d_img).T)
                xdct[0:IMAGE_SIZE//2, 0:IMAGE_SIZE//2] = d_img_dct[0:IMAGE_SIZE//2, 0:IMAGE_SIZE//2]

            xdct = applySignPermutation(xdct, permutation, IMAGE_SIZE, is_denoise)
            data[i, c, :, :] = idct(idct(xdct).T)
            nrm = np.sqrt(np.sum(data[i, c, :, :]**2))
            data[i, c, :, :] /= nrm

        # # visualisation
        # fig = plt.figure()
        #
        # plt.subplot(1,3,1)
        # plt.imshow(data[i, :, :, 0], cmap=plt.cm.gray) # , cmap=plt.cm.gray
        # plt.colorbar()
        # plt.axis('off')
        #
        # xdct = dct(dct(data[i, :, :, 0]).T)
        # xdct = lib.applySignPermutation(xdct, permutation, IMAGE_SIZE)
        #
        # plt.subplot(1,3,2)
        # plt.imshow(xdct, cmap=plt.cm.gray) # , cmap=plt.cm.gray
        # plt.axis('off')
        #
        # data[i, :, :, 0] = idct(idct(xdct).T)
        #
        # plt.subplot(1,3,3)
        # plt.imshow(data[i, :, :, 0], cmap=plt.cm.gray) # , cmap=plt.cm.gray
        # plt.axis('off')
        # plt.colorbar()
        #
        # plt.show()

    return data

def doNormalisation(data):
    data -= np.min(np.min(data, axis=1), axis=0)
    data /= np.max(np.max(data, axis=1), axis=0)
    data -= 0.5

    return data

