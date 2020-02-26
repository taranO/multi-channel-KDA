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

# from models import *
import libs.vgg as modelvgg
import libs.resnet as modelresnet
from utils import progress_bar

from torch.autograd import Variable
from libs.utils import *

# ======================================================================================================================

print("PID = %d\n" % os.getpid())
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()

# ======================================================================================================================

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

parser.add_argument('--model', default='vgg16', help='The target model')

parser.add_argument("--epochs",  default=100, type=int, help="The number of epochs.")
parser.add_argument("--lr",         default=1e-3, type=float, help="Learning rate.")
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")


args = parser.parse_args()

best_acc = 0  # best test accuracy

# ======================================================================================================================
# Training
def train(net, epoch, optimizer):

    net.train()
    train_loss = 0
    correct    = 0
    total      = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    progress_bar(batch_idx, len(trainloader), 'TRAIN Loss: %.3f | Acc: %.3f%% (%d/%d)'
        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(net, epoch, save_dir=""):

    global best_acc

    net.eval()
    test_loss = 0
    correct   = 0
    total     = 0

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    progress_bar(batch_idx, len(testloader), 'TEST: Loss: %.3f | Acc: %.3f%% (%d/%d)'
        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc or epoch == 100:
        print('Saving..')

        torch.save(net.state_dict(), save_dir + "/%s_%d.pth" % (args.model, epoch))
        best_acc = acc

# ======================================================================================================================
if __name__ == '__main__':

    save_dir = "checkpoints/" + args.model
    makeDir(save_dir)

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    #---------------------------------------------
    # Model
    print('==> Building model..')
    if args.model == "vgg16":
        net = modelvgg.VGG('VGG16')
        oprimazer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    elif args.model == "resnet18":
        net = modelresnet.ResNet18()
        oprimazer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)

    if use_cuda:
        net.cuda()
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------------------------

    for epoch in range(args.epochs):
        train(net, epoch, oprimazer)
        test(net, epoch, save_dir)

