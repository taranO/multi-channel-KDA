import argparse
import numpy as np

import torchvision
import torchvision.transforms as transforms

import libs.vgg as modelvgg
import libs.resnet as modelresnet

from torch.autograd import Variable

import libs.lib as lib
from libs.utils import *

# ======================================================================================================================

print("PID = %d\n" % os.getpid())

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()
print(use_cuda)

# ======================================================================================================================

parser = argparse.ArgumentParser(description='One pixel attack with PyTorch')

parser.add_argument('--model', default='resnet18', help='The target model') # vgg16   resnet18
parser.add_argument('--samples', default=1000, type=int, help='The number of image samples to attack.')
parser.add_argument('--pixels', default=1, type=int, help='The number of pixels that can be perturbed.')
parser.add_argument('--permut', default=9, type=int, help='....')

parser.add_argument('--pref', default="adv", help='....') # adv   org
parser.add_argument('--is_prefilt', default=True, type=int, help='...')

parser.add_argument("--is_random_channels", default=False, type=int, help="Is to use a subset of channels for prediciton.")
parser.add_argument("--channel_n", default=9, type=int, help="The number of channels for predition if is_random_channels=True.")

args = parser.parse_args()

is_save_stat = False
K = 2

# ======================================================================================================================
def predictRandomChannels(Net, Permut, channel_n=3, is_vanila=True, is_prefilt=False, DATA_DIR="", IMAGE_SIZE=32, N_CHANELS=3):

    global is_save_stat, K

    Org_labels = np.zeros((1, args.samples))
    Pred_labels = np.zeros((args.samples, 10))
    i = 0

    N = len(Net)
    indx = np.random.choice(N, size=(channel_n), replace=False)
    for j in indx:
        # --- classification ----
        pred_labels, org_labels = testOnStoredData(Net[j], args.samples, Permut[j], is_prefilt, DATA_DIR, IMAGE_SIZE, N_CHANELS)

        print("\n")
        if is_save_stat:
            org_labels = org_labels.astype(int)
            m = org_labels.shape[1]
            Stat = np.zeros(m)
            PL = pred_labels.argsort(1)[:,-K:].transpose()

            for ii in range(m):

                if org_labels[0, ii] == PL[0, ii]:
                    Stat[ii] = abs(pred_labels[ii, PL[0, ii]]) - abs(pred_labels[ii, PL[1, ii]])
                elif org_labels[0, ii] == PL[1, ii]:
                    Stat[ii] = abs(pred_labels[ii, PL[1, ii]]) - abs(pred_labels[ii, PL[0, ii]])

                print("ii=%d, p0=%.8f, p1=%.8f, d=%.8f" % (ii, pred_labels[ii, PL[0, ii]], pred_labels[ii, PL[1, ii]], Stat[ii]))
            if is_vanila:
                np.save("stat_%s_p%d_sb_%s" % (args.model, p, sb), Stat)
            else:
                np.save("stat_adv_%s_p%d_sb_%s" % (args.model, p, sb), Stat)

        if i == 1:
            Org_labels = org_labels
        Pred_labels += pred_labels
        # --- end classification ----

        # Pred_labels.argsort(0)[::-1]

    diff = Org_labels - Pred_labels.argmax(1)
    diff[diff != 0] = 1
    total_error = 100 * np.sum(diff) / args.samples

    print("\np=%d, error=%0.4f\n" % (N, total_error))

    return total_error

def testClasifier(net, EPOCHS= [], MODEL_DIR="", P = [1,2,3], SUBBANDS  = ["d", "h", "v"], is_vanila=True, is_prefilt=False,
                  DATA_DIR="", IMAGE_SIZE=32, N_CHANELS=3):

    global is_save_stat, K

    Org_labels = np.zeros((1, args.samples))
    Pred_labels = np.zeros((args.samples, 10))
    i = 0

    for p in P:
        for sb in SUBBANDS:
            print("%d %s" % (p, sb))
            # --- load model ----
            FROM_DIR = MODEL_DIR % (sb, p)
            print(FROM_DIR + "/%s_%d.pth" % (args.model, EPOCHS[i]))
            net.load_state_dict(torch.load(FROM_DIR + "/%s_%d.pth" % (args.model, EPOCHS[i]), map_location=lambda storage, loc: storage))
            i += 1

            if use_cuda:
                net.cuda()
            else:
                net.cpu()

            net.eval()
            # --- end load model ----

            # --- load permutation ----
            permutation = np.load(FROM_DIR + "/permutation_cifar_%s_dct_sign_permutation_subband_%s_%d.npy" % (args.model, sb, p))
            # --- end load permutation ----

            # --- classification ----
            pred_labels, org_labels = testOnStoredData(net, args.samples, permutation, is_prefilt, DATA_DIR, IMAGE_SIZE, N_CHANELS)

            if is_save_stat:
                org_labels = org_labels.astype(int)
                m = org_labels.shape[1]
                Stat = np.zeros(m)
                PL = pred_labels.argsort(1)[:,-K:].transpose()

                for ii in range(m):

                    if org_labels[0, ii] == PL[0, ii]:
                        Stat[ii] = abs(pred_labels[ii, PL[0, ii]]) - abs(pred_labels[ii, PL[1, ii]])
                    elif org_labels[0, ii] == PL[1, ii]:
                        Stat[ii] = abs(pred_labels[ii, PL[1, ii]]) - abs(pred_labels[ii, PL[0, ii]])

                    print("ii=%d, p0=%.8f, p1=%.8f, d=%.8f" % (ii, pred_labels[ii, PL[0, ii]], pred_labels[ii, PL[1, ii]], Stat[ii]))
                if is_vanila:
                    np.save("stat_%s_p%d_sb_%s" % (args.model, p, sb), Stat)
                else:
                    np.save("stat_adv_%s_p%d_sb_%s" % (args.model, p, sb), Stat)


            if i == 1:
                Org_labels = org_labels
            Pred_labels += pred_labels
            
            # --- end classification ----

            diff = Org_labels - Pred_labels.argmax(1)
            diff[diff != 0] = 1
            #print(diff)
            total_error = 100 * np.sum(diff) / args.samples

    return total_error

def testClasifierRandom(net, EPOCHS= [], MODEL_DIR="", P = [1,2,3], SUBBANDS  = ["d", "h", "v"], channel_n=3, is_vanila=True, is_prefilt=False,
                  DATA_DIR="", IMAGE_SIZE=32, N_CHANELS=3):

    global is_save_stat, K

    Org_labels = np.zeros((1, args.samples))
    Pred_labels = np.zeros((args.samples, 10))
    i = 0

    N = len(P)*len(SUBBANDS)
    indx = np.random.choice(N, size=(channel_n), replace=False)
    print(indx)

    for p in P:
        for sb in SUBBANDS:
            if i not in indx:
                i += 1
                continue
            # --- load model ----
            FROM_DIR = MODEL_DIR % (sb, p)
            net.load_state_dict(torch.load(FROM_DIR + "/%s_%d.pth" % (args.model, EPOCHS[i]), map_location=lambda storage, loc: storage))
            i += 1

            if use_cuda:
                net.cuda()
            else:
                net.cpu()

            net.eval()
            # --- end load model ----

            # --- load permutation ----
            permutation = np.load(FROM_DIR + "/permutation_cifar_%s_dct_sign_permutation_subband_%s_%d.npy" % (args.model, sb, p))
            # --- end load permutation ----

            #print("\nnet %d\n" % i)
            # --- classification ----
            pred_labels, org_labels = testOnStoredData(net, args.samples, permutation, is_prefilt, DATA_DIR, IMAGE_SIZE, N_CHANELS)

            if is_save_stat:
                org_labels = org_labels.astype(int)
                m = org_labels.shape[1]
                Stat = np.zeros(m)
                PL = pred_labels.argsort(1)[:,-K:].transpose()

                for ii in range(m):

                    if org_labels[0, ii] == PL[0, ii]:
                        Stat[ii] = abs(pred_labels[ii, PL[0, ii]]) - abs(pred_labels[ii, PL[1, ii]])
                    elif org_labels[0, ii] == PL[1, ii]:
                        Stat[ii] = abs(pred_labels[ii, PL[1, ii]]) - abs(pred_labels[ii, PL[0, ii]])

                    print("ii=%d, p0=%.8f, p1=%.8f, d=%.8f" % (ii, pred_labels[ii, PL[0, ii]], pred_labels[ii, PL[1, ii]], Stat[ii]))
                if is_vanila:
                    np.save("stat_%s_p%d_sb_%s" % (args.model, p, sb), Stat)
                else:
                    np.save("stat_adv_%s_p%d_sb_%s" % (args.model, p, sb), Stat)

            if i == 1 or np.max(Org_labels) == 0:
                Org_labels = org_labels
            Pred_labels += pred_labels
            # --- end classification ----

            diff = Org_labels - Pred_labels.argmax(1)
            diff[diff != 0] = 1
            total_error = 100 * np.sum(diff) / args.samples

    return total_error

def testOnStoredData(net, N, permutation, is_prefilt=False, DATA_DIR="", IMAGE_SIZE=32, N_CHANELS=3):

    org_labels = np.zeros((1, N))
    pred_labels = np.zeros((N, 10))

    # Stat = []
    for j in range(N):

        targets = np.load(DATA_DIR + "/labels_%d.npy" % (j+1)).astype(np.int)
        if is_prefilt:
            if os.path.isfile(DATA_DIR + "/%s_filt_img_%d.npy" % (args.pref, (j+1))):
                inputs = np.load(DATA_DIR + "/%s_filt_img_%d.npy" % (args.pref, (j+1)))
            else:
                inputs = np.load(DATA_DIR + "/%s_img_%d.npy" % (args.pref, (j+1)))
                inputs = preFiltering(inputs, IMAGE_SIZE=32, N_CHANELS=3)
                np.save(DATA_DIR + "/%s_filt_img_%d" % (args.pref, (j+1)), inputs)
        else:
            inputs = np.load(DATA_DIR + "/%s_img_%d.npy" % (args.pref, (j+1)))
            if len(inputs.shape) <= 3:
                inputs = inputs.reshape((1, inputs.shape[0], inputs.shape[1], inputs.shape[2]))


        
        a = lib.applyDCTPermutation(inputs, permutation, IMAGE_SIZE, N_CHANELS)
        inputs = torch.FloatTensor(a)
        inputs = Variable(torch.FloatTensor(inputs), volatile=True)
        if use_cuda:
            inputs = inputs.cuda()

        outputs = net(inputs)

        org_labels[0, j] = targets
        pred_labels[j]   = np.reshape(outputs.data.cpu().numpy(), (10))

    return pred_labels, org_labels


# ======================================================================================================================
if __name__ == '__main__':

    R = 10
    if args.permut == args.channel_n or not args.is_random_channels:
        R = 1

    data_dir_  = "data/attacked/cifar/multi_channel/%s/permutations_%d/p%d" % (args.model, args.permut, args.pixels)
    model_dir_ = "checkpoints/multi_channel/%s/permutations_%d" % (args.model, args.permut)

    SUBBANDS  = ["d", "h", "v"]
    EPOCHS    = [100, 100, 100, 100, 100, 100, 100, 100, 100]
    P = np.asarray(range(1,round(args.permut/3)+1))

    if args.model == "vgg16":
        Net = modelvgg.VGG('VGG16')
    elif args.model == "resnet18":
        Net = modelresnet.ResNet18()

    E = []
    for r in range(R):

        if args.is_random_channels:
            error = testClasifierRandom(Net, EPOCHS, model_dir_, SUBBANDS=SUBBANDS, P=P, channel_n=args.channel_n,
                                        is_vanila=args.is_vanila, is_prefilt=args.is_prefilt,
                                        DATA_DIR=data_dir_, IMAGE_SIZE=32, N_CHANELS=3)
        else:
            error = testClasifier(Net, EPOCHS, model_dir_, SUBBANDS=SUBBANDS, P=P, is_vanila=args.is_vanila,
                                  is_prefilt=args.is_prefilt, DATA_DIR=data_dir_, IMAGE_SIZE=32, N_CHANELS=3)

        print("R = %d\terror = %0.8f" % (r, error))

        E.append(error)

    print("\n\n Average error = %0.8f" % np.mean(np.asarray(E)))































