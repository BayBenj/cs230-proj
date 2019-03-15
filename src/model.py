import argparse
import os
from contextlib import redirect_stdout

import numpy as np
import pprint as pp
import matplotlib.pyplot as plt
import datetime
import pytz

import tensorflow as tf
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, BatchNormalization, Concatenate
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.optimizers import Adam, SGD
from keras.callbacks import History
import keras.backend as K

CONV_FACTOR = np.log(10)
history = History()
VGG_MEAN = [103.939, 116.779, 123.68]

def ldr_encoder():
    vgg16_input = VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3))
    for layer in vgg16_input.layers[:17]:
        layer.trainable = False
    inp_img = vgg16_input.layers[0].input
    skip1 = vgg16_input.layers[2].output
    skip2 = vgg16_input.layers[5].output
    skip3 = vgg16_input.layers[9].output
    skip4 = vgg16_input.layers[13].output
    print(skip1.shape)
    print(skip2.shape)
    print(skip3.shape)
    print(skip4.shape)
    result = vgg16_input.layers[-2].output
    return inp_img, skip1, skip2, skip3, skip4, result, vgg16_input

def decoder_layer(nn_in, n_filters, filter_size, stride, pad='same',
        acti='relu'):
    nn = Conv2DTranspose(n_filters, filter_size,
        strides=stride, padding=pad, activation=acti)(nn_in)
    nn = BatchNormalization()(nn)

    return nn

def hdr_decoder(inp_img, skip1, skip2, skip3, skip4, latent_rep):
    network = decoder_layer(latent_rep, 512, 3, 2)

    network = Concatenate(axis = -1)([network, skip4])
    network = Conv2D(512, 1, activation='relu')(network)
    network = decoder_layer(network, 256, 3, 2)

    network = Concatenate(axis = -1)([network, skip3])
    network = Conv2D(256, 1, activation='relu')(network)
    network = decoder_layer(network, 128, 3, 2)

    network = Concatenate(axis = -1)([network, skip2])
    network = Conv2D(128, 1, activation='relu')(network)
    network = decoder_layer(network, 64, 3, 2)

    network = Concatenate(axis = -1)([network, skip1])
    network = Conv2D(64, 1, activation='relu')(network)
    network = Conv2D(3, 1, activation='relu')(network)

    network = Concatenate(axis = -1)([network, inp_img])
    result = Conv2D(3, 1, activation='relu')(network)

    return result

def custom_loss(yTrue, yPred):
    return K.mean(K.square(yTrue - yPred))

def psnr(yTrue, yPred):
    return 10 * K.log(4 / K.mean(K.square(yTrue - yPred))) / CONV_FACTOR

def assemble():
    # Encoder (VGG16)
    inp_img, skip1, skip2, skip3, skip4, latent_rep, vgg16_input = ldr_encoder()

    # Decoder
    x = hdr_decoder(inp_img, skip1, skip2, skip3, skip4, latent_rep)

    model = Model(inputs=vgg16_input.input, outputs=x)

    optimi = Adam()

    model.compile(optimizer=optimi, loss=custom_loss, metrics=[psnr])

    return model

def train(model, XY_train, XY_dev, epochs, batch_size):
    X_train, Y_train = XY_train
    X_dev, Y_dev = XY_dev

    model.summary()

    model.fit(X_train, Y_train, batch_size, epochs,
        validation_data=(X_dev, Y_dev),
        verbose=1,
        shuffle=True,
        callbacks=[history])

def load_weights(model, model_fp):
    model.load_weights(model_fp)

def write_summary(model, out_fd):
    with open(os.path.join(out_fd, 'summary.txt'), 'w') as summ_file:
        with redirect_stdout(summ_file):
            model.summary()

def predict_imgs(model, imgs, out_fd, fn_tag):
    img_X, img_Y = imgs

    out_img = model.predict(img_X[0:1])
    out_img = out_img * 255.0
    print(out_img.shape)

    X_img = image.array_to_img(img_X[0] * 255.0)
    Y_img = image.array_to_img(img_Y[0] * 255.0)
    Yhat_img = image.array_to_img(out_img[0])

    X_img.save(os.path.join(out_fd, '{}-x.jpg'.format(fn_tag)))
    Y_img.save(os.path.join(out_fd, '{}-y.jpg'.format(fn_tag)))
    Yhat_img.save(os.path.join(out_fd, '{}-yhat.jpg'.format(fn_tag)))

def plot(out_fd):
    plt.plot(range(0,len(history.history["psnr"])), history.history["psnr"])
    plt.plot(range(0,len(history.history["val_psnr"])), history.history["val_psnr"])
    plt.legend(["train","dev"], loc='center left')
    plt.ylabel('PSNR')
    plt.xlabel('Epoch')
    plt.savefig(os.path.join(out_fd, 'psnr_chart.png'), dpi=100)
    plt.clf()

    plt.plot(range(0,len(history.history["loss"])), history.history["loss"])
    plt.plot(range(0,len(history.history["val_loss"])), history.history["val_loss"])
    plt.legend(["train","dev"], loc='center left')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig(os.path.join(out_fd, 'loss_chart.png'), dpi=100)
    plt.clf()

def save(m, out_fd):
    m.save(os.path.join(out_fd, 'model.h5'))
