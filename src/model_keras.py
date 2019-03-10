#!/usr/bin/env python3

import argparse
import os

import numpy as np
import tensorflow as tf

from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose
# from keras.optimizers import
from keras.applications import VGG16
import keras.backend as K

kernel_size = (4, 4)

def main():
    global args

    args = parse_args()

    train_set, dev_set = load_datasets()

    train_model(train_set, dev_set)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Script to train HDR inference model')
    parser.add_argument('--input-dir', dest='input_dir', type=str,
        default='.//',
        help='Base path to training input images dataset')
    parser.add_argument('--output-fn', dest='output_fn', type=str,
        default='hdr-infer-model.h5',
        help='Output model filename')

    return parser.parse_args()

def train_model(train_set, dev_set):
    model = assemble_model()

    # model.fit(...)


def assemble_model():
    # Encoder (VGG16)
    vgg16_input = VGG16(include_top=False, weights='imagenet',
        input_shape=(224, 224, 3))
    for layer in vgg16_input.layers[:17]:
        layer.trainable = False

    x = vgg16_input.layers[-2].output

    # Decoder
    x = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(x)
    x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
    x = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(x)
    x = Conv2DTranspose(3, (1, 1), activation='sigmoid', padding='same')(x)

    model = Model(inputs=vgg16_input.input, outputs=x)

    model.summary()

    return model.compile(optimizer='adam', loss='binary_crossentropy',
        metrics=['accuracy'])

def load_datasets():
    return [], []

if __name__ == '__main__':
    main()
