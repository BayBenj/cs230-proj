#!/usr/bin/env python3

import argparse
import os

import numpy as np
import tensorflow as tf

from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose
# from keras.optimizers import
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
import keras.backend as K

import pprint as pp

kernel_size = (4, 4)

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
        default=100)
    parser.add_argument('--epochs', dest='num_epochs', type=int,
        default=100)
    parser.add_argument('--batch_size', dest='batch_size', type=int,
        default=32)

    return parser.parse_args()

def train_model(dataset):
    model = assemble_model()

    X, Y = dataset

    model.summary()

    model.fit(X, Y, epochs=args.num_epochs, batch_size=args.batch_size,
        verbose=2, validation_split=0.2, shuffle=True)

def customLoss(yTrue, yPred):
    return K.mean(K.square(yTrue - yPred))

def PSNRMetric(yTrue, yPred):
    return 20 * K.log( 1 / K.mean(K.square(yTrue - yPred)) ) / 2.3
#    print(mse)
#    if mse == 0:
#        return 100
#    PIXEL_MAX = 255.0
#    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
#
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

    model.compile(optimizer='adam', loss=customLoss,
        metrics=['accuracy', PSNRMetric])

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
    
    x = np.asarray(x) / 255
    y = np.asarray(y) / 255

    # Normalize input images
    #x = preprocess_input(x)
    #y = preprocess_input(y)
    
    # Shuffle training pairs
    idx = list(range(x.shape[0]))
    np.random.shuffle(idx)
    x = x[idx]
    y = y[idx]

    return x, y

if __name__ == '__main__':
    main()
