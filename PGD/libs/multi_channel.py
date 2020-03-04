'''
Multi-channel model with KDA
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
import copy

import libs.vgg as modelvgg
import libs.resnet as modelresnet

import numpy as np
from scipy.fftpack import dct, idct

# ===============================================================================================

class MultiChannel(nn.Module):
    def __init__(self, model_name="vgg16", MODEL_DIR="", EPOCHS=[], P=[1,2,3],SUBBANDS=["d", "h", "v"],
                 IMAGE_SIZE=32, N_CHANELS=3, use_cuda=True, is_prefilt=False, is_random_channels=False, channel_n=9):
        super(MultiChannel, self).__init__()

        if model_name == "vgg16":
            net = modelvgg.VGG('VGG16')
        elif model_name == "resnet18":
            net = modelresnet.ResNet18()

        self.image_size  = IMAGE_SIZE
        self.n_channel   = N_CHANELS
        self.use_cuda    = use_cuda
        self.is_prefilt  = is_prefilt
        self.epochs      = EPOCHS
        self.p           = P
        self.subbands    = SUBBANDS
        self.NET         = []
        self.permutation = []

        N = len(P) * len(SUBBANDS)
        if is_random_channels:
            indx = np.random.choice(N, size=(channel_n), replace=False)
        else:
            inds = range(N)

        print(indx)
        i = -1
        for p in P:
            for sb in SUBBANDS:
                if i not in indx:
                    i += 1
                    continue

                i += 1

                # --- load model ----
                FROM_DIR = MODEL_DIR % (sb, p)
                net.load_state_dict(torch.load(FROM_DIR + "/%s_%d.pth" % (model_name, EPOCHS[i]), map_location=lambda storage, loc: storage))

                if use_cuda:
                    net.cuda()
                else:
                    net.cpu()
                net.eval()
                self.NET.append(copy.deepcopy(net))
                # --- end load model ----

                # --- load permutation ----
                self.permutation.append(np.load(FROM_DIR + "/permutation_cifar_%s_dct_sign_permutation_subband_%s_%d.npy" % (model_name, sb, p)))
                # --- end load permutation ----


    def forward(self, x):
        
        z = x.data.cpu().numpy()
           
        if self.is_prefilt:
            z = self.preFiltering(z, IMAGE_SIZE=32, N_CHANELS=3)
  
        pred_labels = np.zeros((z.shape[0], 10))

        N = len(self.NET)
        for i in range(N):

            a  = self.applyDCTPermutation(z.copy(), self.permutation[i])
            inputs = torch.FloatTensor(a)

            if self.use_cuda:
                inputs = inputs.cuda()

            inputs = Variable(inputs, volatile=True)
            a = self.NET[i](inputs).data.cpu().numpy()
            pred_labels += a

        cl= pred_labels.argmax(1)
        pred_labels[:] = 0
        pred_labels[:, cl] = 1
        
        return Variable(torch.FloatTensor(pred_labels), volatile=True)

    def preFiltering(self, img, thr=1, b=3, IMAGE_SIZE=32, N_CHANELS=3):

        st      = round((b - 1) / 2)
        res_img = np.zeros(img.shape)

        for k in range(img.shape[0]):
            for n in range(N_CHANELS):
                x = np.pad(img[k,n], st, mode="constant", constant_values=-100)
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

                            res_img[k, n, r - 1, c - 1] = np.sum(blk) / blk.shape[0]
                        else:
                            res_img[k, n, r - 1, c - 1] = x[r, c]


        return res_img

    def applyDCTPermutation(self, data, permutation):

        n = data.shape[0]

        for i in range(n):
            for c in range(self.n_channel):
                xdct = dct(dct(data[i, c, :, :]).T)
                xdct = self.applySignPermutation(xdct, permutation)
                data[i, c, :, :] = idct(idct(xdct).T)
                nrm = np.sqrt(np.sum(data[i, c, :, :]**2))
                data[i, c, :, :] /= nrm

        return data

    def applySignPermutation(self, data, permutation):
        dim = data.shape

        data = np.reshape(data, (-1, self.image_size ** 2))
        data = np.multiply(data, np.tile(permutation, (data.shape[0], 1)))

        return np.reshape(data, dim)