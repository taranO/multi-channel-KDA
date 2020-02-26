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
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import TensorDataset

from differential_evolution import differential_evolution

import libs.multi_channel as model
import datetime
from libs.utils import *
#--------------------------------------------------------------

print("PID = %d\n" % os.getpid())
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()

# ======================================================================================================================

parser = argparse.ArgumentParser(description='One pixel attack with PyTorch')
parser.add_argument('--model', default='resnet18', help='The target model')
parser.add_argument('--permut', default=9, type=int, help='The number of permutations.')
parser.add_argument('--pixels', default=1, type=int, help='The number of pixels that can be perturbed.')
parser.add_argument('--maxiter', default=10, type=int, help='The maximum number of iteration in the DE algorithm.')
parser.add_argument('--popsize', default=400, type=int, help='The number of adverisal examples in each iteration.')
parser.add_argument('--samples', default=1000, type=int, help='The number of image samples to attack.')
parser.add_argument('--is_prefilt', default=True, type=int, help='')

args = parser.parse_args()

# ======================================================================================================================
def perturb_image(xs, img):
	if xs.ndim < 2:
		xs = np.array([xs])
	batch = len(xs)
	imgs = img.repeat(batch, 1, 1, 1)
	xs = xs.astype(int)

	count = 0
	for x in xs:
		pixels = np.split(x, len(x)/5)
		
		for pixel in pixels:
			x_pos, y_pos, r, g, b = pixel
			imgs[count, 0, x_pos, y_pos] = (r/255.0-0.4914)/0.2023
			imgs[count, 1, x_pos, y_pos] = (g/255.0-0.4822)/0.1994
			imgs[count, 2, x_pos, y_pos] = (b/255.0-0.4465)/0.2010
		count += 1

	return imgs

def predict_classes(xs, img, target_calss, net, minimize=True):
    global use_cuda

    imgs_perturbed = perturb_image(xs, img.clone())
    input = Variable(imgs_perturbed, volatile=True)
    if use_cuda:
        input = input.cuda()
    predictions = F.softmax(net(input)).data.cpu().numpy()[:, target_calss]

    return predictions if minimize else 1 - predictions

def attack_success(x, img, target_calss, net, targeted_attack=False, verbose=False):
    global use_cuda

    attack_image = perturb_image(x, img.clone())
    input = Variable(attack_image, volatile=True)
    if use_cuda:
        input = input.cuda()
    confidence = F.softmax(net(input)).data.cpu().numpy()[0]
    predicted_class = np.argmax(confidence)

    if (verbose):
        print("Confidence: %.4f"%confidence[target_calss])
    if (targeted_attack and predicted_class == target_calss) or (not targeted_attack and predicted_class != target_calss):
        return True


def attack(img, label, net, target=None, pixels=1, maxiter=75, popsize=400, verbose=False):
    # img: 1*3*W*H tensor
    # label: a number

    global use_cuda

    targeted_attack = target is not None
    target_calss = target if targeted_attack else label

    bounds = [(0,32), (0,32), (0,255), (0,255), (0,255)] * pixels

    popmul = max(1, int(popsize/len(bounds)))

    predict_fn = lambda xs: predict_classes(
        xs, img, target_calss, net, target is None)
    callback_fn = lambda x, convergence: attack_success(
        x, img, target_calss, net, targeted_attack, verbose)

    inits = np.zeros([popmul*len(bounds), len(bounds)])
    for init in inits:
        for i in range(pixels):
            init[i*5+0] = np.random.random()*32
            init[i*5+1] = np.random.random()*32
            init[i*5+2] = np.random.normal(128,127)
            init[i*5+3] = np.random.normal(128,127)
            init[i*5+4] = np.random.normal(128,127)

    attack_result = differential_evolution(predict_fn, bounds, maxiter=maxiter, popsize=popmul,
        recombination=1, atol=-1, callback=callback_fn, polish=False, init=inits)

    attack_image = perturb_image(attack_result.x, img)
    attack_var = Variable(attack_image, volatile=True)
    if use_cuda:
        attack_var = attack_var.cuda()

    predicted_class = np.argmax(net(attack_var).data.cpu().numpy()[0])

    return attack_image, predicted_class

def attack_all(net, loader, pixels=1, targeted=False, maxiter=75, popsize=400, verbose=False, save_attacked_to=""):

    global use_cuda
    correct = 0
    success = 0
    count = 0
    M = 9 if targeted else 1

    for batch_idx, (input, target) in enumerate(loader):

        img_var = Variable(input, volatile=True)
        if use_cuda:
            img_var = img_var.cuda()

        _, indices = torch.max(net(img_var), 1)

        if target[0] == int(indices.data.cpu().numpy()[0]):
            correct += 1

        targets = [None] if not targeted else range(10)

        A   = np.zeros((M, 3, 32, 32))
        O   = np.zeros((M, 3, 32, 32))
        Ocl = np.zeros((M))
        Acl = np.zeros((M))

        i = 0
        for target_calss in targets:
            if (targeted):
                if (target_calss == target[0]):
                    continue
            attack_image, pr_cl = attack(input, target[0], net, target_calss, pixels=pixels, maxiter=maxiter,
                                         popsize=popsize, verbose=verbose)

            Ocl[i] = target[0]
            Acl[i] = target_calss
            O[i]   = np.reshape(input.cpu().numpy(), (3, 32, 32))
            A[i]   = np.reshape(attack_image.cpu().numpy(), (3, 32, 32))
            i +=1

            success += 1 if target[0] != pr_cl else 0

        if save_attacked_to != "":
            count += 1
            np.save(save_attacked_to + "/adv_img_%d" % count, A)
            np.save(save_attacked_to + "/org_img_%d" % count, O)
            np.save(save_attacked_to + "/targets_%d" % count, Acl)
            np.save(save_attacked_to + "/labels_%d"  % count, Ocl)


        print("%s\t #sampel = %d, correct = %d, success = %d" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), count, correct, success))
        if batch_idx == args.samples:
            break

    return float(success)/(count*M)

# ======================================================================================================================
if __name__ == '__main__':

    save_to_ = "data/attacked/cifar/multi_channel/%s/permutations_%d/p%d" % (args.model, args.permut, args.pixels)
    makeDir(save_to_)
    # --------------------------

    # -- load data ---
    print("==> Load data...")
    tranfrom = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=tranfrom)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)

    # -- load model ---

    model_dir_ = "checkpoints/multi_channel/%s/permutations_%d" % (args.model, args.permut)
    EPOCHS = [100, 100, 100, 100, 100, 100, 100, 100, 100]
    P = np.asarray(range(1, round(args.permut / 3) + 1))

    net = model.MultiChannel(args.model, MODEL_DIR=model_dir_, EPOCHS=EPOCHS, P=P, use_cuda=True, is_prefilt=args.is_prefilt)

    # -- start attck ---
    print("==> Starting attck...")
    results = attack_all(net, testloader, pixels=args.pixels, targeted=args.targeted, maxiter=args.maxiter,
                         popsize=args.popsize, verbose=args.verbose, save_attacked_to=save_to_)

    print("Final success rate: %.4f"%results)