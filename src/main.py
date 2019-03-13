#!/usr/bin/env python3

import argparse
import os

import numpy as np
import pprint as pp
import matplotlib.pyplot as plt
import time
import datetime
import pytz

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

import model


def timestamp():
    def utc_to_local(utc_dt):
        local_tz = pytz.timezone('US/Pacific')
        local_dt = utc_dt.replace(tzinfo=pytz.utc).astimezone(local_tz)
        return local_tz.normalize(local_dt)
    now = datetime.datetime.utcnow()
    dt = utc_to_local(now)
    formatted = dt.strftime("%Y%m%d-%H%M%S")
    return formatted


time = timestamp()


def main():
    global args
    args = parse_args()
    x, y = load_datasets()
    m = model.train((x, y), args.num_epochs, args.batch_size)
    m.predict(model, x[0:1])
    plot()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script to train HDR inference model')
    parser.add_argument('--input-dir', dest='input_dir', type=str,
                        default='/proj/data/proc_nn_dataset',
                        help='Base path to training input images dataset')
    parser.add_argument('--output-fn', dest='output_fn', type=str,
                        default='hdr-infer-model.h5',
                        help='Output model filename')
    parser.add_argument('--max-samples', dest='max_samples', type=int,
                        default=None)
    parser.add_argument('--epochs', dest='num_epochs', type=int,
        default=1)
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        default=32)

    return parser.parse_args()


def load_datasets():
    x_fd = os.path.join(args.input_dir, 'x')
    y_fd = os.path.join(args.input_dir, 'y')

    x = []
    y = []
    for i, imgx_fn in enumerate(os.listdir(x_fd)):
        if (args.max_samples is not None) and (i + 1 > args.max_samples):
            break
        imgx_fp = os.path.join(x_fd, imgx_fn)
        imgy_fn = 'y' + imgx_fn[1:]
        imgy_fp = os.path.join(y_fd, imgy_fn)
        if os.path.isfile(imgx_fp):
            assert(os.path.isfile(imgy_fp))
            #x = image.img_to_array(image.load_img(imgx_fp))
            #print("Original Image:")
            #print(x[0:3, 0:3, :])
            #red = x[:, :, 0:1] - VGG_MEAN[2]
            #green = x[:, :, 1:2] - VGG_MEAN[1]
            #blue = x[:, :, 2:3] - VGG_MEAN[0]
            #bgr = np.concatenate([blue, green, red], axis = -1)
            #print(bgr.shape)
            #print("My Procesing:")
            #print(bgr[0:3, 0:3, :])
            #x_process = preprocess_input(x, mode = 'tf')
            #print("Keras Processing:")
            #print(x_process[0:3, 0:3, :])
            x.append(image.img_to_array(image.load_img(imgx_fp)))
            y.append(image.img_to_array(image.load_img(imgy_fp)))

    x = (np.asarray(x) / 127.5) - 1
    #x = np.asarray(x) / 255
    print(x.shape)
    y = (np.asarray(y) / 127.5) - 1
    print(y.shape)

    # Normalize input images
    #x = preprocess_input(x)
    #y = preprocess_input(y)

    # Shuffle training pairs
    idx = list(range(x.shape[0]))
    np.random.shuffle(idx)
    x = x[idx]
    y = y[idx]

    return x, y


def plot(history):
    plt.plot(range(0,len(history.history["psnr"])), history.history["psnr"])
    plt.plot(range(0,len(history.history["val_psnr"])), history.history["val_psnr"])
    plt.legend(["train","dev"], loc='center left')
    plt.ylabel('PSNR')
    plt.xlabel('Epoch')
    plt.savefig("output/psnr_{}.png".format(time), dpi=100)
    plt.clf()

    plt.plot(range(0,len(history.history["loss"])), history.history["loss"])
    plt.plot(range(0,len(history.history["val_loss"])), history.history["val_loss"])
    plt.legend(["train","dev"], loc='center left')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig("output/loss_{}.png".format(time), dpi=100)
    plt.clf()


if __name__ == '__main__':
    main()
