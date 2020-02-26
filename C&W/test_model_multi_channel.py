"""Defending against adversarial attacks by randomized diversification"""

import argparse
import tensorflow as tf
import keras

import numpy as np
import datetime

from libs.setup_mnist import *
from libs.setup_cifar import *
import libs.model_multi_channel as mcm




########################################################################################################################
print("PID = %d\n" % os.getpid())

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

########################################################################################################################
parser = argparse.ArgumentParser(description="Test multi-channel system with Key based Diversified Aggregation.")

parser.add_argument("--type",        default="mnist",      help="The dataset type.") # mnist  fashion_mnist   cifar
parser.add_argument("--attack_type", default="carlini_l2", help="The attack type.")
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

    print(f"\n\n{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    start = datetime.datetime.now()

    P = [1,2,3]  # number of channels per subband
    SUBBANDS = ["d", "h", "v"]  # DCT subbands
    MODEL_DIR = args.model_dir + "/" + args.type
    EPOCHS = [args.epoch for i in range(len(P)*len(SUBBANDS))]

    pref = "adv" # adv  org
    data_dir_ = args.data_dir + "/" + args.type + "/" + args.attack_type

    # with tf.Session() as sess:
    #     keras.backend.set_session(sess)
    with tf.compat.v1.Session() as sess:
        tf.compat.v1.keras.backend.set_session(sess)

        # test data and parameters
        if args.type == "mnist":
            nn_param = [32, 32, 64, 64, 200, 200]
            model = MNISTModelAllLayers(nn_param, session=sess)
            IMAGE_SIZE = 28
            N_CHANELS = 1

        elif args.type == "fashion_mnist":
            nn_param = [32, 32, 64, 64, 200, 200]
            model = MNISTModelAllLayers(nn_param, session=sess)
            IMAGE_SIZE = 28
            N_CHANELS = 1
        elif args.type == "cifar":
            nn_param = [64, 64, 128, 128, 256, 256]
            model = CIFARModelAllLayers(nn_param, session=sess)
            IMAGE_SIZE = 32
            N_CHANELS = 3

        # multi-channel model initialization with classifier defined in model variable
        multi_channel_model = mcm.MultiChannel(model,
                                               type         = args.type,
                                               permt        = P,
                                               subbands     = SUBBANDS,
                                               model_dir    = MODEL_DIR,
                                               img_size     = IMAGE_SIZE,
                                               img_channels = N_CHANELS,
                                               is_median    = args.is_median,
                                               is_mean      = args.is_mean,
                                               is_custom    = args.is_custom,
                                               filt_size    = args.filt_size,
                                               filt_sigma    = args.filt_sigma)
        multi_channel_model.test_init(sess, nn_param, EPOCHS)


        # --------------------------------------------------------------------------------------------------------------
        N = 0
        org_labels  = [] # original labels
        pred_labels = [] # labels predicted by multi-channel model
        for i in range(args.samples):

            label = np.load(data_dir_ + "/labels_%d.npy" % i)
            input = np.load(data_dir_ + "/%s_img_%d.npy" % (pref, i))
            input = prepare_data_for_classification(input, IMAGE_SIZE, N_CHANELS)
            N += input.shape[0]

            if args.is_random_channels:
                prediction = multi_channel_model.predictRandomChannels(input, args.channel_n)
            else:
                prediction = multi_channel_model.predict(input)

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

    time = datetime.datetime.now() - start
    print(f"\n\n total time: {time}")