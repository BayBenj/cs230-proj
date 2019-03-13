import argparse
import os

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


def decoder_layer(nn_in, n_filters, filter_size, stride, pad="same", act="linear"):
    nn = Conv2DTranspose(n_filters, filter_size, strides=stride, padding=pad, activation=act)(nn_in)
    nn = BatchNormalization()(nn)
    return nn


def hdr_decoder(inp_img, skip1, skip2, skip3, skip4, latent_rep):
    network = decoder_layer(latent_rep, 512, 3, 2)
    
    network = Concatenate(axis = -1)([network, skip4])
    network = Conv2D(512, 1, activation='linear')(network)
    network = decoder_layer(network, 256, 3, 2)
    
    network = Concatenate(axis = -1)([network, skip3])
    network = Conv2D(256, 1, activation='linear')(network)
    network = decoder_layer(network, 128, 3, 2)
    
    network = Concatenate(axis = -1)([network, skip2])
    network = Conv2D(128, 1, activation='linear')(network)
    network = decoder_layer(network, 64, 3, 2)
    
    network = Concatenate(axis = -1)([network, skip1])
    network = Conv2D(64, 1, activation='linear')(network)
    network = Conv2D(3, 1, activation='linear')(network)
    
    network = Concatenate(axis = -1)([network, inp_img])
    result = Conv2D(3, 1, activation='linear')(network)
    #result = decoder_layer(network, 3, 1, 1, act='sigmoid')
    return result


def custom_loss(yTrue, yPred):
    return K.mean(K.square(yTrue - yPred))


def psnr(yTrue, yPred):
    return 10 * K.log(2 / K.mean(K.square(yTrue - yPred))) / CONV_FACTOR


def assemble():
    # Encoder (VGG16)
    inp_img, skip1, skip2, skip3, skip4, latent_rep, vgg16_input = ldr_encoder()

    # Decoder
    x = hdr_decoder(inp_img, skip1, skip2, skip3, skip4, latent_rep)

    model = Model(inputs=vgg16_input.input, outputs=x)
    model.compile(optimizer='adam', loss=custom_loss, metrics=[psnr])

    return model


def train(dataset, epochs, batch_size):     
    model = assemble()
    X, Y = dataset
    model.summary()
    model.fit(X, Y, batch_size, epochs, verbose=1, validation_split=0.2, shuffle=True, callbacks=[history])
    return model


def predict_imgs(model, img, time):
    out_img = model.predict(img[0:1])
    out_img = (out_img + 1) * 127.5
    print(out_img.shape)
    pred_img = image.array_to_img(out_img[0])
    inp_img = image.array_to_img((img[0] + 1) * 127.5)
    pred_img.save('output/out_img_{}.jpg'.format(time))
    inp_img.save('output/inp_img_{}.jpg'.format(time))


def plot(time):
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

