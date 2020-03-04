'''
PGD attack: test transferability
'''

import os
import foolbox
import torch
import torch.backends.cudnn as cudnn

import numpy as np
import argparse

import libs.multi_channel as model
import libs.vgg as modelvgg
import libs.resnet as modelresnet
#--------------------------------------------------------------

print("PID = %d\n" % os.getpid())
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda =  torch.cuda.is_available()
#--------------------------------------------------------------

parser = argparse.ArgumentParser(description='PGD attack: test transferability')
parser.add_argument('--model', default='vgg16', help='The target model: vgg16,  resnet18')
parser.add_argument('--epoch', default=100, type=int, help='Trained model epoch')
parser.add_argument('--samples', default=1000, type=int, help='The number of image samples to attack.')

parser.add_argument('--is_multi_channel', default=True, type=int, help='Is to test multy-channel model or vanilla one')
parser.add_argument('--permut', default=9, type=int, help='The number of channels in multi-channel system.')
parser.add_argument("--is_random_channels", default=True, type=int, help="Is to use a subset of channels for prediciton.")
parser.add_argument("--channel_n", default=9, type=int, help="The number of channels for predition if is_random_channels=True.")

parser.add_argument('--prefilt_type', default="", type=str, help='Pre-filtering type, if any: mean_filt, median_filt, custom_filt')


args = parser.parse_args()

# =================================================================================
def makeDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

# ===============================================================================
if __name__ == '__main__':

    print(args)

    if args.is_multi_channel:
        R = 10 if args.is_random_channels and args.permut != args.channel_n else 1
        data_dir_ = "data/attacked/cifar/%s/PGD/%s" % (args.model, args.prefilt_type)
    else:
        R = 1
        data_dir_ = "data/attacked/cifar/%s/PGD" % args.model

    # --- Test ---------------------------------------------------------------------
    Error_org = []
    Error_adv = []
    for r in range(R):

        if args.is_multi_channel:
            epochs_ = [100, 100, 100, 100, 100, 100, 100, 100, 100]
            SUBBANDS = ["d", "h", "v"]
            P = np.asarray(range(1, round(args.permut / 3) + 1))

            model_dir_ = "checkpoint/multi_channel/" + args.model + "_adam_lr10-3/%s_%d"

            # init multi-channel model
            net = model.MultiChannel(args.model, MODEL_DIR=model_dir_, EPOCHS=epochs_, P=P, use_cuda=use_cuda,
                                     is_prefilt=args.is_prefilt, SUBBANDS=SUBBANDS, is_random_channels=args.is_random_channels, channel_n=args.channel_n)
        else:
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

        fmodel = foolbox.models.PyTorchModel(net, bounds=(0, 1), num_classes=10)

        #----TEST ---------------------
        adv_error = 0
        org_error = 0
        for n in range(1, args.samples+1):

            input_org = np.load("%s/org_img_%d.npy" % (data_dir_, n))
            input_adv = np.load("%s/adv_img_%d.npy" % (data_dir_, n))
            target    = np.load("%s/targets_%d.npy" % (data_dir_, n))
            
            # original image prediction
            pred_lab = np.argmax(fmodel.forward_one(input_org))
            if target != pred_lab:
                org_error += 1

            # adversarial image prediction
            adv_label = np.argmax(fmodel.forward_one(input_adv))
            if adv_label != pred_lab:
                adv_error += 1

        Error_org.append(org_error)
        Error_adv.append(adv_error)

        print("R=%d:\tclassification error %0.5f, \torg error %0.5f, \ttotal error %0.5f" %
              (r, 100 * adv_error / args.samples, 100 * org_error / args.samples, 100 * (adv_error + org_error) / args.samples))

    org_error = np.mean(np.asarray(org_error))
    adv_error = np.mean(np.asarray(Error_adv))

    print("\n\naverage: \tclassification error %0.5f, \torg error %0.5f, \ttotal error %0.5f" % (
        100 * adv_error / n, 100 * org_error / n, 100 * (adv_error + org_error) / n))