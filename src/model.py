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
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, LeakyReLU, \
    BatchNormalization, Dropout, Concatenate, Activation, UpSampling2D
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.optimizers import Adam, SGD
from keras.callbacks import History, EarlyStopping
import keras.backend as K

CONV_FACTOR = np.log(10)
history = History()
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.2, patience=3)
VGG_MEAN = [103.939, 116.779, 123.68]

def ldr_encoder():
    vgg16_enc = {}

    vgg16 = VGG16(include_top=False, weights='imagenet',
        input_shape=(224, 224, 3))

    # for layer in vgg16.layers[:17]:
    for layer in vgg16.layers:
        layer.trainable = False

    vgg16_enc['img_inp'] = vgg16.input
    vgg16_enc['inp'] = vgg16.layers[0].input
    vgg16_enc['b1c2'] = vgg16.layers[2].output
    vgg16_enc['b2c2'] = vgg16.layers[5].output
    vgg16_enc['b3c3'] = vgg16.layers[9].output
    vgg16_enc['b4c3'] = vgg16.layers[13].output
    vgg16_enc['b5c3'] = vgg16.layers[17].output
    vgg16_enc['b5p'] = vgg16.layers[18].output

    vgg16.summary()

    return vgg16_enc

def latent_layers(x, drpo_rate=None, enable_bn=False):
    # x = Conv2D(512, 1)(x)
    # if enable_bn:
    #     x = BatchNormalization()(x)
    # x = activation_layer(x)
    # if drpo_rate is not None:
    #     x = Dropout(drpo_rate)(x)

    x = Conv2D(512, 1)(x)
    if enable_bn:
        x = BatchNormalization()(x)
    x = activation_layer(x)
    if drpo_rate is not None:
        x = Dropout(drpo_rate)(x)
    
    # x = upscale_layer(x, 512,
    #     drpo_rate=drpo_rate, enable_bn=enable_bn)

    return x

def upscale_layer(x, num_filters, filter_size=(3, 3), strides=(2, 2),
        padding='same', drpo_rate=None, enable_bn=False):
    x = Conv2DTranspose(num_filters, filter_size,
        strides=strides, padding=padding)(x)
    if enable_bn:
        x = BatchNormalization()(x)
    x = activation_layer(x)
    if drpo_rate is not None:
        x = Dropout(drpo_rate)(x)
    
    return x

def concat_conv_layer(x, ldr_layer, num_filters, filter_size=(1, 1),
        strides=(1, 1), activ='relu', output_layer=False,
        drpo_rate=None, enable_bn=False):
    x = Concatenate(axis=-1)([x, ldr_layer])
    x = Conv2D(num_filters, filter_size, strides=strides)(x)
    if enable_bn:
        x = BatchNormalization()(x)
    x = activation_layer(x, output_layer)    
    if drpo_rate is not None:
        x = Dropout(drpo_rate)(x)

    return x

def activation_layer(x, output_layer=False):
    if not output_layer:
        return LeakyReLU(alpha=0.3)(x)
        # return Activation('relu')(x)
    else:
        return LeakyReLU(alpha=0.3)(x)
        # return Activation('sigmoid')(x)

def hdr_decoder(x, ldr_enc, drpo_rate=None, enable_bn=False):
    x = upscale_layer(x, 512,
        drpo_rate=drpo_rate, enable_bn=enable_bn)
    # x = concat_conv_layer(x, ldr_enc['b5c3'], 512,
    #     drpo_rate=drpo_rate, enable_bn=enable_bn)
    # x = upscale_layer(x, 256,
    #     drpo_rate=drpo_rate, enable_bn=enable_bn)

    x = concat_conv_layer(x, ldr_enc['b4c3'], 512,
        drpo_rate=drpo_rate, enable_bn=enable_bn)
    x = upscale_layer(x, 256,
        drpo_rate=drpo_rate, enable_bn=enable_bn)

    x = concat_conv_layer(x, ldr_enc['b3c3'], 256,
        drpo_rate=drpo_rate, enable_bn=enable_bn)
    x = upscale_layer(x, 128,
        drpo_rate=drpo_rate, enable_bn=enable_bn)
    
    x = concat_conv_layer(x, ldr_enc['b2c2'], 128,
        drpo_rate=drpo_rate, enable_bn=enable_bn)
    x = upscale_layer(x, 64,
        drpo_rate=drpo_rate, enable_bn=enable_bn)
    
    x = concat_conv_layer(x, ldr_enc['b1c2'], 64,
        drpo_rate=drpo_rate, enable_bn=enable_bn)

    x = Conv2D(3, 1)(x)
    if enable_bn:
        x = BatchNormalization()(x)
    x = activation_layer(x)
    if drpo_rate is not None:
        x = Dropout(drpo_rate)(x)
    
    x = concat_conv_layer(x, ldr_enc['inp'], 3, output_layer=True,
        drpo_rate=drpo_rate, enable_bn=enable_bn)

    return x

def total_variation_loss(yPred):
    img_nrows = 224
    img_ncols = 224
    assert K.ndim(yPred) == 4
    if K.image_data_format() == 'channels_first':
        a = K.square(yPred[:, :, :img_nrows - 1, :img_ncols - 1] - yPred[:, :, 1:, :img_ncols - 1])
        b = K.square(yPred[:, :, :img_nrows - 1, :img_ncols - 1] - yPred[:, :, :img_nrows - 1, 1:])
    else:
        a = K.square(yPred[:, :img_nrows - 1, :img_ncols - 1, :] - yPred[:, 1:, :img_ncols - 1, :])
        b = K.square(yPred[:, :img_nrows - 1, :img_ncols - 1, :] - yPred[:, :img_nrows - 1, 1:, :])
    return K.mean(K.pow(a + b, 0.5))

def l2_loss(yTrue, yPred):
    return K.mean(K.square(yTrue - yPred))

def l1_loss(yTrue, yPred):
    return K.mean(K.abs(yTrue - yPred))

def custom_loss(yTrue, yPred):
    # return total_variation_loss(yPred) + l1_loss(yTrue, yPred)
    return K.mean(K.square(yTrue - yPred))

def psnr(yTrue, yPred):
    return 10.0 * K.log(1.0 / K.mean(K.square(yTrue - yPred))) / CONV_FACTOR

def assemble(drpo_rate, enable_bn):
    # Encoder (VGG16)
    ldr_enc = ldr_encoder()

    # Latent repr. layers
    # x = latent_layers(ldr_enc['b5p'], drpo_rate, enable_bn)
    x = latent_layers(ldr_enc['b5c3'], drpo_rate, enable_bn)

    # Decoder
    x = hdr_decoder(x, ldr_enc, drpo_rate, enable_bn)

    model = Model(inputs=ldr_enc['img_inp'], outputs=x)

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
        callbacks=[history, early_stopping])

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
