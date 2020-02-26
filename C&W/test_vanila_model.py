## train_models.py -- train the neural network models for attacking
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import os
import argparse
import tensorflow as tf
import keras

import numpy as np
from libs.setup_mnist import *
from libs.setup_cifar import *

from libs.setup_cifar import CIFAR, CIFARModel
from libs.setup_mnist import MNIST, FashionMNIST, MNISTModel

########################################################################################################################
print("PID = %d\n" % os.getpid())

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

########################################################################################################################
parser = argparse.ArgumentParser(description="Test vanilla model")

parser.add_argument("--type",        default="mnist",      help="The dataset type.") # mnist  fashion_mnist   cifar
parser.add_argument("--attack_type", default="carlini_li", help="The attack type.")
parser.add_argument("--data_dir",    default="data/attacked", help="The dataset path.")
parser.add_argument("--model_dir",   default="checkpoints",        help="Path where to save models.")

parser.add_argument("--epoch",      default=50,    type=int, help="The number of epochs.")
parser.add_argument("--samples",    default=1000, type=int, help="The number of test samples.")

args = parser.parse_args()

########################################################################################################################

def prepare_data_for_classification(data, IMAGE_SIZE, N_CHANELS):
    test = data.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, N_CHANELS)

    if np.max(test) > 0.75:
        test -= 0.5

    return test

########################################################################################################################
if __name__ == "__main__":

    pref = "adv" # org   adv
    data_dir_ = args.data_dir + "/" + args.type + "/" + args.attack_type

    # with tf.Session() as sess:
    #     keras.backend.set_session(sess)
    with tf.compat.v1.Session() as sess:
        tf.compat.v1.keras.backend.set_session(sess)

        # test data and parameters
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

        # --------------------------------------------------------------------------------------------------------------
        N = 0
        org_labels  = [] # original labels
        pred_labels = [] # labels predicted by multi-channel model
        for i in range(args.samples):

            label = np.load(data_dir_ + "/labels_%d.npy" % i)
            input = np.load(data_dir_ + "/%s_img_%d.npy" % (pref, i))
            input = prepare_data_for_classification(input, IMAGE_SIZE, N_CHANELS)
            N += input.shape[0]

            prediction = model.model.predict(input)

            if i == 0:
                pred_labels = prediction
            else:
                pred_labels = np.vstack((pred_labels, prediction))
            org_labels.append(label)

        # classificaiton error
        diff = org_labels - np.reshape(pred_labels.argmax(1), (args.samples, -1))
        diff[diff != 0] = 1
        total_error = 100 * np.sum(diff) / N

        print('datatype = %s, attack_type = %s: \t error = %0.2f' %  (args.type, args.attack_type, total_error))
