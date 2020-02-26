''' Multi-channel model with Key based Diversified Aggregation'''


import numpy as np
from datetime import datetime
from scipy.fftpack import dct, idct
from scipy import ndimage, misc
import copy


# ===============================================================================================

class MultiChannel():
    def __init__(self, model,
                    type = "",
                    epochs = 100,
                    optimazer = "adam",
                    learning_rate = 1e-3,
                    batch_size = 64,
                    permt = [1,2,3],
                    subbands = ["d", "h", "v"],
                    model_dir = "",
                    img_size = 28,
                    img_channels = 1,
                    is_zero   = False,
                    use_cuda  = True,
                    is_median = False,
                    is_mean   = False,
                    is_custom = False,
                    filt_size = 3,
                    filt_sigma = 1):

        super(MultiChannel, self).__init__()

        self.image_size  = img_size
        self.n_channel   = img_channels
        self.use_cuda    = use_cuda
        self.is_zero     = is_zero

        self.is_prefilt  = True if is_median or is_mean or is_custom else False
        self.is_median   = is_median
        self.is_mean     = is_mean
        self.is_custom   = is_custom
        self.filt_size   = filt_size
        self.filt_sigma  = filt_sigma

        self.type = type

        self.model         = model
        self.epochs        = epochs
        self.optimazer     = optimazer
        self.learning_rate = learning_rate
        self.batch_size    = batch_size

        self.P           = permt
        self.SUBBANDS    = subbands
        self.NET         = []
        self.EPOCHS      = []
        self.permutation = []

        self.model_dir = model_dir
        self.name = self.type + "_is_zero_" + str(self.is_zero) + "_subband_%s_p%d"


    def train(self, data):

        for p in self.P: # number of channels per subband
            for subband in self.SUBBANDS: # dct subbands

                print("\n\n" + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ": TYPE = %s, p = %d, subband=%s\n\n" % (self.type, p, subband))

                name = self.name % (subband, p)

                # dct subband permutation per channel
                permutation = self.signPermutation(subband)
                np.save(self.model_dir + "/permutation_" + name, permutation)
                data.train_data      = self.applyDCTPermutation(data.train_data, permutation)
                data.validation_data = self.applyDCTPermutation(data.validation_data, permutation)

                # classifier training per channel
                for epoch in range(self.epochs):
                    self.model.train(data,
                                     learning_rate=self.learning_rate,
                                     num_epochs=self.epochs,
                                     batch_size=self.batch_size,
                                     oprimazer=self.optimazer,
                                     file=self.model_dir + "/" + name + "_epochs_%d" % epoch)

    def test_init(self,sess, nn_param, EPOCHS):
        i = -1
        for p in self.P:
            for sb in self.SUBBANDS:
                i += 1

                # --- load model ----
                pref = self.model_dir + "/" + self.name % (sb, p)
                model = copy.deepcopy(self.model)
                model.load(pref + "_epochs_%d" % EPOCHS[i])
                self.NET.append(model)
                # --- end load model ----

                # --- load permutation ----
                self.permutation.append(np.load(self.model_dir + "/permutation_" + self.name % (sb, p) + ".npy"))
                # --- end load permutation ----

    def predict(self, x):

        z = x.copy()
        if self.is_prefilt:
            z = self.preFiltering(z, IMAGE_SIZE=self.image_size, N_CHANELS=self.n_channel)

        pred_labels = np.zeros((z.shape[0], 10))

        N = len(self.NET)
        for i in range(N):

            inputs = self.applyDCTPermutation(z.copy(), self.permutation[i])
            pred_labels += self.NET[i].model.predict(inputs)

        return pred_labels


    def predictRandomChannels(self, x, chanel_n):

        z = x.copy()
        if self.is_prefilt:
            z = self.preFiltering(z, IMAGE_SIZE=self.image_size, N_CHANELS=self.n_channel)

        pred_labels = np.zeros((z.shape[0], 10))

        N = len(self.NET)
        indx = np.random.choice(N, size=(chanel_n), replace=False)
        for i in indx:

            inputs = self.applyDCTPermutation(z.copy(), self.permutation[i])
            pred_labels += self.NET[i].model.predict(inputs)

        return pred_labels

    def preFiltering(self, img, thr=1,IMAGE_SIZE=32, N_CHANELS=3):

        st      = round((self.filt_size - 1) / 2)
        res_img = np.zeros(img.shape)

        for k in range(img.shape[0]):
            for n in range(N_CHANELS):
                x = img[k,:,:, n].copy()

                if self.is_mean:
                    x = ndimage.gaussian_filter(x, sigma=self.filt_sigma)
                    res_img[k, :, :, n] = x

                if self.is_median:
                    x = ndimage.median_filter(x, size=self.filt_size)
                    res_img[k, :, :, n] = x

                if self.is_custom:
                    x = np.pad(x, st, mode="constant", constant_values=-100)
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

                                res_img[k, r - 1, c - 1, n] = np.sum(blk) / blk.shape[0]
                            else:
                                res_img[k, r - 1, c - 1, n] = x[r, c]


        return res_img

    # ------------------------------------------------------------------------------------------------------------------
    def applyDCTPermutation(self, data, permutation):

        n = data.shape[0]

        for i in range(n):
            for c in range(self.n_channel):
                xdct = dct(dct(data[i, :, :, c]).T)
                xdct = self.applySignPermutation(xdct, permutation)
                data[i, :, :, c] = idct(idct(xdct).T)
                nrm = np.sqrt(np.sum(data[i, :, :, c]**2))
                data[i, :, :, c] /= nrm

        return data

    def applySignPermutation(self, data, permutation):
        dim = data.shape

        data = np.reshape(data, (-1, self.image_size ** 2))
        data = np.multiply(data, np.tile(permutation, (data.shape[0], 1)))

        return np.reshape(data, dim)

    def signPermutation(self, subband=""):
        if self.is_zero:
            permutation = np.zeros((1, self.image_size ** 2))
        else:
            permutation = np.random.normal(size=self.image_size ** 2)
            permutation[permutation >= 0] = 1
            permutation[permutation != 1] = -1

        if subband == "d":  # D - diagonal
            permutation = np.reshape(permutation, (self.image_size, self.image_size))
            permutation[0:self.image_size // 2, :] = 1
            permutation[:, 0:self.image_size // 2] = 1

        elif subband == "v":  # V - vertical
            permutation = np.reshape(permutation, (self.image_size, self.image_size))
            permutation[:, 0:self.image_size // 2] = 1
            permutation[self.image_size // 2:self.image_size, :] = 1

        elif subband == "h":  # H - horizontal
            permutation = np.reshape(permutation, (self.image_size, self.image_size))
            permutation[0:self.image_size // 2, :] = 1
            permutation[:, self.image_size // 2:self.image_size] = 1

        elif subband == "dhv":
            permutation = np.reshape(permutation, (self.image_size, self.image_size))
            permutation[0:self.image_size // 2, 0:self.image_size // 2] = 1

        return np.reshape(permutation, (self.image_size ** 2))