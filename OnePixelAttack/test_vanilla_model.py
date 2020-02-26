import os
import sys
import numpy as np

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from models import *
from utils import progress_bar
from torch.autograd import Variable

from differential_evolution import differential_evolution
import models.vgg as modelvgg
import models.resnet as modelresnet
from utils import progress_bar
from torch.autograd import Variable
from scipy import ndimage
import matplotlib

from libs.utils import *

# ======================================================================================================================

print("PID = %d\n" % os.getpid())
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()

# ======================================================================================================================

parser = argparse.ArgumentParser(description='One pixel attack with PyTorch')
parser.add_argument('--model', default='vgg16', help='The target model')
parser.add_argument("--epoch",  default=100, type=int, help="The number of epochs.")
parser.add_argument('--samples', default=1000, type=int, help='The number of image samples to attack.')
parser.add_argument('--pixels', default=1, type=int, help='The number of pixels that can be perturbed.')


args = parser.parse_args()

# ======================================================================================================================

def testOnStoredData(net, N, DATA_DIR="", is_attacked=True):

    org_labels = np.zeros((1, N))
    pred_labels = np.zeros((N, 10))

    for j in range(N):
        targets = np.load(DATA_DIR + "/labels_%d.npy" % (j+1)).astype(np.int)

        pref = "adv" if is_attacked else "org"
        inputs = np.load(DATA_DIR + "/%s_img_%d.npy" % (pref, j + 1))
        inputs = Variable(torch.FloatTensor(inputs), volatile=True)
        if use_cuda:
            inputs = inputs.cuda()

        outputs = net(inputs)

        org_labels[0, j] = targets
        pred_labels[j]   = np.reshape(outputs.data.cpu().numpy(), (10))

    diff = org_labels - pred_labels.argmax(1)
    diff[diff != 0] = 1
    total_error = 100 * np.sum(diff) / N

    return total_error, np.argwhere(diff != 0)[:, 1]

# ======================================================================================================================
if __name__ == '__main__':

    data_dir_ = "data/attacked/cifar/%s/p%d" % (args.model, args.pixels)

    if args.model == "vgg16":
        net = modelvgg.VGG('VGG16')
    elif args.model == "resnet18":
        net = modelresnet.ResNet18()

    # load model
    net.load_state_dict(
        torch.load('checkpoints/%s/%s_%d.pth' % (args.model, args.model, args.epoch), map_location=lambda storage, loc: storage))

    if use_cuda:
        net.cuda()
    else:
        net.cpu()
    cudnn.benchmark = True
    net.eval()

    # ---------------------------------------------------------------------------------------------------------
    error, adv_ind = testOnStoredData(net, args.samples, data_dir_, is_prefilt=args.is_prefilt, is_attacked=True)
    print("%d: error=%.4f\n" % (args.samples, error))

