'''
PGD attack: vanilla classifier
'''

import os
import foolbox
import torch

import torchvision

import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

import numpy as np
import argparse

import libs.vgg as modelvgg
import libs.resnet as modelresnet

#--------------------------------------------------------------

print("PID = %d\n" % os.getpid())
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()
#--------------------------------------------------------------

parser = argparse.ArgumentParser(description='PGD attack with PyTorch')

parser.add_argument('--model', default='resnet18', help='The target model') # vgg16   resnet18
parser.add_argument('--epoch', default=100, type=int, help='The number of training epochs.')
parser.add_argument('--epsilon', default=0.5, type=int,  help='PGD attack parameter')
parser.add_argument('--iterations', default=100,  type=int, help='PGD attack parameter')
parser.add_argument('--samples', default=1000, type=int, help='The number of image samples to attack.')

args = parser.parse_args()

# =================================================================================
def makeDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

# ===============================================================================
if __name__ == '__main__':

    print(args)

    save_attacked_to = "data/attacked/cifar/%s/PGD" % args.model
    makeDir(save_attacked_to)

    # --- Load data --------------------------
    tranfrom_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    data_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=tranfrom_test)
    dataloader = torch.utils.data.DataLoader(data_set, batch_size=1, shuffle=False, num_workers=1)

    # ----- Load model -----------------
    if args.model == "vgg16":
        net = modelvgg.VGG('VGG16')
    elif args.model == "resnet18":
        net = modelresnet.ResNet18()

    net.load_state_dict(
        torch.load('./checkpoint/%s/adam_lr10-3/%s_%d.pth' % (args.model, args.model, args.epoch),
                   map_location=lambda storage, loc: storage))

    if use_cuda:
        net.cuda()
    else:
        net.cpu()
    cudnn.benchmark = True
    net.eval()

    # --- Attack init ---------------------------------------------
    fmodel = foolbox.models.PyTorchModel(net, bounds=(0, 1), num_classes=10)
    attack = foolbox.v1.attacks.PGD(fmodel)

    # ---- Attack --------------------------------------------
    error = 0
    n = 0
    for batch_idx, (input, target) in enumerate(dataloader):
        if n >= args.samples:
            break

        n += 1
        target = int(target.data.numpy())
        input  = input.data.cpu().numpy().squeeze()

        # renormalisation
        min_   = np.min(input)
        input -= min_
        max_   = np.max(input)
        input /= max_

        # apply attack on source image
        adversarial = attack(input, target, epsilon=args.epsilon, iterations=args.iterations, random_start=True)

        adversarial *= max_
        adversarial += min_
        adv_label = np.argmax(fmodel.forward_one(adversarial))

        if adv_label != target:
            error += 1

        input *= max_
        input += min_

        np.save("%s/adv_img_%d.npy" % (save_attacked_to, n), adversarial)
        np.save("%s/labels_%d.npy" % (save_attacked_to, n), np.asarray(adv_label))
        np.save("%s/org_img_%d.npy" % (save_attacked_to, n), input)
        np.save("%s/targets_%d.npy" % (save_attacked_to, n), target)

        print("n=%d:\t error %0.5f" % (n, 100*error/n))
