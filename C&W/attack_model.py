## test_attack.py -- sample code to test attack procedure
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.


########################################################################################################################
import tensorflow as tf
import numpy as np
import time
import glob
import os
from datetime import datetime
import argparse

from libs.setup_cifar import CIFAR, CIFARModel
from libs.setup_mnist import MNIST, FashionMNIST, MNISTModel

from libs.l2_attack import CarliniL2
from libs.l0_attack import CarliniL0
from libs.li_attack import CarliniLi

########################################################################################################################
print("PID = %d\n" % os.getpid())

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

########################################################################################################################
parser = argparse.ArgumentParser(description="Attack vanilla model.")

parser.add_argument("--type",      default="mnist", help="The dataset type.")
parser.add_argument("--data_dir",  default="data/attacked", help="The dataset path.")
parser.add_argument("--model_dir", default="checkpoints", help="Path where to save models.")
parser.add_argument("--epoch",     default=50, type=int, help="The number of epochs.")
parser.add_argument("--samples",   default=1000, type=int, help="The number of test samples.")

parser.add_argument("--attack_type", default="carlini_l2", help="The attack type.")

args = parser.parse_args()

########################################################################################################################

def show(img):
    """
    Show MNSIT digits in the console.
    """
    remap = "  .*#" + "#" * 100
    img = (img.flatten() + .5) * 3
    if len(img) != 784:
        return
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i * 28:i * 28 + 28]]))


def generate_data(data, samples, targeted = True, start = 0, inception = False):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets = []
    labels = []
    for i in range(samples):
        if targeted:
            if inception:
                seq = random.sample(range(1, 1001), 10)
            else:
                seq = range(data.test_labels.shape[1])

            for j in seq:
                if (j == np.argmax(data.test_labels[start + i])) and (inception == False):
                    continue
                inputs.append(data.test_data[start + i])
                targets.append(np.eye(data.test_labels.shape[1])[j])
                labels.append(np.argmax(data.test_labels[start + i]))
        else:
            inputs.append(data.test_data[start + i])
            targets.append(data.test_labels[start + i])
            labels.append(np.argmax(data.test_labels[start + i]))

    inputs = np.array(inputs)
    targets = np.array(targets)
    labels = np.array(labels)

    return inputs, targets, labels


def create_result_dir(results_path = 'results'):
    # --results_path
    try:
        os.mkdir(results_path)
    except(FileExistsError):
        pass
    # delete all existing files
    files = glob.glob(results_path + '/*')
    for f in files:
        os.remove(f)
########################################################################################################################

if __name__ == "__main__":
    with tf.Session() as sess:

        target_attacks = 9
        if args.type == "mnist":
            data, model = MNIST(), MNISTModel("%s/mnist_epoch_%d" % (args.model_dir, args.epoch), sess)
            N_CHANELS   = 1
            IMAGE_SIZE  = 28
        elif args.type == "fashion-mnist":
            data, model = FashionMNIST(), MNISTModel("%s/fashion_mnist_epoch_%d" % (args.model_dir, args.epoch), sess)
            N_CHANELS   = 1
            IMAGE_SIZE  = 28
        elif args.type == "cifar":
            data, model = CIFAR(), CIFARModel("%s/cifar_epoch_%d" % (args.model_dir, args.epoch), sess)
            N_CHANELS   = 3
            IMAGE_SIZE  = 32

        save_to_dir_ = "%s/%s/%s" % (args.data_dir, args.type, args.attack_type)

        # ------------------------------------------------------
        if args.attack_type == "carlini_l2":
            attack = CarliniL2(sess, model)
        elif args.attack_type == "carlini_l0":
            attack = CarliniL0(sess, model)
        elif args.attack_type == "carlini_li":
            attack = CarliniLi(sess, model)

        print("Load test data\n")
        test_inputs, test_targets, test_labels = generate_data(data, samples = args.samples, targeted = True,
                                                start = 0, inception = False)

        print("Start process\n")
        for i in range(args.samples):

            print("\n" + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": i = %d\n" % i)

            inputs = test_inputs[i * target_attacks:i * target_attacks + target_attacks]
            targets = test_targets[i * target_attacks:i * target_attacks + target_attacks]
            labels = test_labels[i * target_attacks:i * target_attacks + target_attacks]

            adv = attack.attack(inputs, targets)
           
            adv_all = np.reshape(adv, (-1, IMAGE_SIZE, IMAGE_SIZE, N_CHANELS))
            org_all = np.reshape(inputs, (-1, IMAGE_SIZE, IMAGE_SIZE, N_CHANELS))

            np.save(save_to_dir_ + "/adv_img_%d" % i,
                    np.reshape(adv_all, (-1, IMAGE_SIZE, IMAGE_SIZE, N_CHANELS)))
            np.save(save_to_dir_ + "/org_img_%d" % i,
                    np.reshape(org_all, (-1, IMAGE_SIZE, IMAGE_SIZE, N_CHANELS)))
            np.save(save_to_dir_ + "/targets_%d" % i, targets)
            np.save(save_to_dir_ + "/labels_%d" % i, labels)


