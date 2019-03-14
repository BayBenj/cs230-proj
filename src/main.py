#!/usr/bin/env python3

import argparse
import os

import numpy as np
import pprint as pp
import matplotlib.pyplot as plt
import datetime
import pytz
import math

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

def main():
    global args

    args = parse_args()

    time = timestamp()

    # Create and train model
    XY_train, XY_dev = load_trdev_datasets()
    m = model.train(XY_train, XY_dev, args.num_epochs, args.batch_size)

    # Save outputs
    out_fd = os.path.join(args.output_dir, 'model_{}'.format(time))
    os.makedirs(out_fd)

    if args.save_model:
        model.save(m, out_fd)
    model.predict_imgs(m, (XY_dev[0][0:1], XY_dev[1][0:1]), out_fd)
    model.plot(out_fd)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Script to train HDR inference model')
    parser.add_argument('--input-dir', dest='input_dir', type=str,
                        default='/proj/data/split_nn_dataset',
                        help='Base path to training input images dataset')
    parser.add_argument('--output-dir', dest='output_dir', type=str,
                        default='./model_out/',
                        help='Output directory')
    parser.add_argument('--save-model', dest='save_model', action='store_true',
                        default=False,
                        help='Save model weights in H5 file')
    parser.add_argument('--val-split', dest='val_split', type=float,
                        default=0.8)
    parser.add_argument('--max-samples', dest='max_samples', type=int,
                        default=1000)
    parser.add_argument('--epochs', dest='num_epochs', type=int,
        default=5)
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        default=32)

    args = parser.parse_args()

    assert((args.val_split > 0.0) and (args.val_split <= 1.0))

    return args

def load_dataset(dataset_fd, max_samples):
    x_fd = os.path.join(dataset_fd, 'x')
    y_fd = os.path.join(dataset_fd, 'y')

    x = []
    y = []
    for i, imgx_fn in enumerate(os.listdir(x_fd)):
        if (max_samples is not None) and (i + 1 > max_samples):
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

def load_trdev_datasets():
    train_fd = os.path.join(args.input_dir, 'train')
    dev_fd = os.path.join(args.input_dir, 'dev')

    max_train_samples = args.val_split * args.max_samples
    max_dev_samples = math.ceil((1.0 - args.val_split) * args.max_samples)

    return load_dataset(train_fd, max_train_samples), \
        load_dataset(dev_fd, max_dev_samples)

if __name__ == '__main__':
    main()
