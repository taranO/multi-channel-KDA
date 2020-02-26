from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import libs.vgg as modelvgg
import libs.resnet as modelresnet

from utils import progress_bar
from torch.autograd import Variable

import numpy as np
from datetime import datetime
import libs.lib as lib
from libs.utils import *

# ======================================================================================================================

print("PID = %d\n" % os.getpid())
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()

# ======================================================================================================================

parser = argparse.ArgumentParser(description='Train multi-channel system with Key based Diversified Aggregation per channel.')

parser.add_argument('--model', default='vgg16', help='The target model')
parser.add_argument('--permut', default=9, type=int, help='The number of channels.')

parser.add_argument("--epochs",  default=100, type=int, help="The number of epochs.")
parser.add_argument("--lr",         default=1e-3, type=float, help="Learning rate.")
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")

args = parser.parse_args()

# ======================================================================================================================
# Training
def train(net, optimizer, epoch, permutation, IMAGE_SIZE=32, N_CHANELS=3):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs = torch.FloatTensor(lib.applyDCTPermutation(inputs.numpy(), permutation, IMAGE_SIZE, N_CHANELS))

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data.cpu().numpy()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    progress_bar(batch_idx, len(trainloader), 'TRAIN Loss: %.3f | Acc: %.3f%% (%d/%d)'
                 % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    return net

def test(net, epoch, permutation, save_to, IMAGE_SIZE=32, N_CHANELS=3):
    global best_acc

    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(testloader):

        inputs = torch.FloatTensor(lib.applyDCTPermutation(inputs.numpy(), permutation, IMAGE_SIZE, N_CHANELS))
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data.cpu().numpy()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    progress_bar(batch_idx, len(testloader), 'TEST: Loss: %.3f | Acc: %.3f%% (%d/%d)'
                 % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc or epoch == 100:
        print('Saving..')
        makeDir(save_to)

        torch.save(net.state_dict(), save_to + "/%s_%d.pth" % (args.model, epoch))
        best_acc = acc

# ======================================================================================================================
if __name__ == '__main__':

    save_dir_ = "checkpoints/multi_channel/%s/permutations_%d"  % (args.model, args.permut)

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    #--------------------------------------------------------------

    for i in [1,2,3]:
        for sb in ["d", "h", "v"]:

            print("\n\n subband: %s, i = %d" % (sb, i))
            best_acc = 0
            # - to save
            save_to = "%s/%s_%d" % (save_dir_, sb, i)
            name = "cifar_" + args.model + "_dct_sign_permutation_subband_%s_%d"
            makeDir(save_to)

            # - permutation
            permutation = lib.signPermutation(IMAGE_SIZE=32, subband=sb)
            name    = name % (sb, i)
            np.save(save_to + "/permutation_" + name, permutation)

            # - model
            if args.model == "vgg16":
                net = modelvgg.VGG('VGG16')
                optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
            elif args.model == "resnet18":
                net = modelresnet.ResNet18()
                optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)

            if use_cuda:
                net.cuda()
                cudnn.benchmark = True

            criterion = nn.CrossEntropyLoss()

            # - train & test
            for epoch in range(args.epochs):
                net = train(net, optimizer, epoch, permutation)
                test(net, epoch, permutation, save_to)
