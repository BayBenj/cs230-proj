#!/usr/bin/env python3

import argparse
import os

import numpy as np
import pprint as pp

import tensorflow as tf
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose
from keras.applications import VGG16
import keras.backend as K


def main():
    global args
    args = parse_args()
    x, y = load_datasets()
    train_model((x, y))


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
        default=10)
    parser.add_argument('--batch_size', dest='batch_size', type=int,
        default=32)
    parser.add_argument('--no-rand-seed', dest='rnd_seed',
        action='store_false', default=True)

    return parser.parse_args()

def train_model(dataset):
    model = assemble_model()

    X, Y = dataset

    model.summary()

    model.fit(X, Y, epochs=args.num_epochs, batch_size=args.batch_size,
        verbose=2, validation_split=0.2, shuffle=True)


def encoder():
    vgg16_input = VGG16(include_top=False, weights='imagenet',
        input_shape=(224, 224, 3))
    for layer in vgg16_input.layers[:17]:
        layer.trainable = False
    return vgg16_input


def decoder(x):
    x = Conv2DTranspose(256, (3, 3), strides=(2, 2), activation='linear', padding="same")(x)
    x = Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='linear', padding="same")(x)
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='linear', padding="same")(x)
    x = Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='linear', padding="same")(x)
    x = Conv2D(3, (1, 1), activation='sigmoid', padding="same")(x)
    return x


def assemble_model():
    vgg16 = encoder()
    x = vgg16.layers[-2].output
    x = decoder(x)

    model = Model(inputs=vgg16.input, outputs=x)

    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

def load_datasets():
    x_fd = os.path.join(args.input_dir, 'x')
    y_fd = os.path.join(args.input_dir, 'y')
    
    x = []
    y = []
    for i, imgx_fn in enumerate(os.listdir(x_fd)):
        if (args.max_samples is not None) and (i+1 > args.max_samples):
            break
        imgx_fp = os.path.join(x_fd, imgx_fn)
        imgy_fn = 'y' + imgx_fn[1:]
        imgy_fp = os.path.join(y_fd, imgy_fn)
        if os.path.isfile(imgx_fp):
            assert(os.path.isfile(imgy_fp))
            x.append(image.img_to_array(image.load_img(imgx_fp)))
            y.append(image.img_to_array(image.load_img(imgy_fp)))
    
    x = np.asarray(x)
    y = np.asarray(y)

    # Normalize input images
    # VGG16.preprocess_inputs()
    
    # Shuffle training pairs
    idx = list(range(x.shape[0]))

    if args.rnd_seed:
        np.random.seed(3141516)

    np.random.shuffle(idx)
    x = x[idx]
    y = y[idx]

    return x, y


if __name__ == '__main__':
    main()

