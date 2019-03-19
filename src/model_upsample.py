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
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, \
    BatchNormalization, Dropout, Concatenate, LeakyReLU, UpSampling2D, Activation
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.optimizers import Adam, SGD
from keras.regularizers import l1, l2
from keras.callbacks import History
import keras.backend as K

CONV_FACTOR = np.log(10)
history = History()
VGG_MEAN = [103.939, 116.779, 123.68]


def ldr_encoder():
    vgg16_enc = {}

    vgg16 = VGG16(include_top=False, weights='imagenet',
        input_shape=(224, 224, 3))

    # for layer in vgg16.layers[:17]:
    for layer in vgg16.layers[:17]:
        layer.trainable = False

    vgg16_enc['inp'] = vgg16.layers[0].input
    vgg16_enc['b1c2'] = vgg16.layers[2].output
    vgg16_enc['b2c2'] = vgg16.layers[5].output
    vgg16_enc['b3c3'] = vgg16.layers[9].output
    vgg16_enc['b4c3'] = vgg16.layers[13].output
    vgg16_enc['b5c3'] = vgg16.layers[17].output

    vgg16.layers[2].trainable = True
    vgg16.layers[5].trainable = True
    vgg16.layers[9].trainable = True
    vgg16.layers[13].trainable = True
    vgg16.layers[17].trainable = True
    
    vgg16.layers[1].trainable = True
    vgg16.layers[4].trainable = True
    vgg16.layers[8].trainable = True
    vgg16.layers[12].trainable = True
    vgg16.layers[16].trainable = True
    
    return vgg16_enc


def upscale_layer(x, n_filters, filter_size, stride, drpo_rate,
        pad='same', acti='relu'):
    #x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Conv2DTranspose(n_filters, filter_size, bias_regularizer = l2(0.001), kernel_regularizer = l2(0.001),
        strides=2, padding=pad)(x)
    x = LeakyReLU(alpha = 0.3)(x)
    #x = BatchNormalization()(x)
    x = Dropout(rate = drpo_rate)(x)

    return x


def hdr_decoder(ldr_enc, drpo_rate):
    x = upscale_layer(ldr_enc['b5c3'], 256, 3, 2, drpo_rate)
    
    x = Concatenate(axis = -1)([x, ldr_enc['b4c3']])
    #x = Conv2D(256, 1, activation='relu')(x)
    x = Conv2D(256, 1, bias_regularizer = l2(0.001), kernel_regularizer = l2(0.001))(x)
    x = LeakyReLU(alpha = 0.3)(x)
    x = Dropout(rate = drpo_rate)(x)
    x = upscale_layer(x, 128, 3, 2, drpo_rate)

    x = Concatenate(axis = -1)([x, ldr_enc['b3c3']])
    #x = Conv2D(256, 1, activation='relu')(x)
    x = Conv2D(128, 1, bias_regularizer = l2(0.001), kernel_regularizer = l2(0.001))(x)
    x = LeakyReLU(alpha = 0.3)(x)
    x = Dropout(rate = drpo_rate)(x)
    x = upscale_layer(x, 64, 3, 2, drpo_rate)

    x = Concatenate(axis = -1)([x, ldr_enc['b2c2']])
    #x = Conv2D(128, 1, activation='relu')(x)
    x = Conv2D(64, 1, bias_regularizer = l2(0.001), kernel_regularizer = l2(0.001))(x)
    x = LeakyReLU(alpha = 0.3)(x)
    x = Dropout(rate = drpo_rate)(x)
    x = upscale_layer(x, 32, 3, 2, drpo_rate)

    x = Concatenate(axis = -1)([x, ldr_enc['b1c2']])
    #x = Conv2D(64, 1, activation='relu')(x)
    x = Conv2D(32, 1, bias_regularizer = l2(0.001), kernel_regularizer = l2(0.001))(x)
    x = LeakyReLU(alpha = 0.3)(x)
    #x = Dropout(drpo_rate)(x)
    #x = Conv2D(3, 1, activation='relu')(x)
    x = Conv2D(3, 1, bias_regularizer = l2(0.001), kernel_regularizer = l2(0.001))(x)
    x = LeakyReLU(alpha = 0.3)(x)
    #x = Dropout(drpo_rate)(x)

    x = Concatenate(axis = -1)([x, ldr_enc['inp']])
    x = Conv2D(3, 1, bias_regularizer = l2(0.001), kernel_regularizer = l2(0.001))(x)
    x = Activation('sigmoid')(x)
    #x = Concatenate(axis = -1)([x, ldr_enc['inp']])
    #x = Conv2D(3, 1, activation='sigmoid')(x)

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
    return K.mean(K.pow(a + b, 1.5))

def l2_loss(yTrue, yPred):
    return K.mean(K.square(yTrue - yPred))

def l1_loss(yTrue, yPred):
    return K.mean(K.abs(yTrue - yPred))

def custom_loss(yTrue, yPred):
    return (0.5 * total_variation_loss(yPred)) + l1_loss(yTrue, yPred) + l2_loss(yTrue, yPred)
    #return l1_loss(yTrue, yPred) + l2_loss(yTrue, yPred)

def psnr(yTrue, yPred):
    return 10.0 * K.log(1.0 / K.mean(K.square(yTrue - yPred))) / CONV_FACTOR

def ssim(yTrue, yPred):#may be wrong
    K1 = 0.04
    K2 = 0.06
    mu_x = K.mean(yPred)
    mu_y = K.mean(yTrue)
    sig_x = K.std(yPred)
    sig_y = K.std(yTrue)
    sig_xy = (sig_x * sig_y) ** 0.5
    L =  255
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    ssim = (2 * mu_x * mu_y + C1) * (2 * sig_xy * C2) * 1.0 / ((mu_x ** 2 + mu_y ** 2 + C1) * (sig_x ** 2 + sig_y ** 2 + C2))
    return ssim 

def assemble(drpo_rate, enable_bn):
    # Encoder (VGG16)
    ldr_enc = ldr_encoder()

    # Decoder
    x = hdr_decoder(ldr_enc, drpo_rate)

    model = Model(inputs=ldr_enc['inp'], outputs=x)

    optimi = Adam()

    model.compile(optimizer=optimi, loss=custom_loss, metrics=[psnr, ssim, l1_loss, l2_loss])

    return model


def train(model, XY_train, XY_dev, epochs, batch_size, es_fp):
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
