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
from keras.callbacks import History, EarlyStopping, ModelCheckpoint
from keras import regularizers
import keras.backend as K

CONV_FACTOR = np.log(10)
VGG_MEAN = [103.939, 116.779, 123.68]

history = History()

def ldr_encoder():
    vgg16_enc = {}

    vgg16 = VGG16(include_top=False, weights='imagenet',
        input_shape=(224, 224, 3))

    # for layer in vgg16.layers[:16]:
    vgg16_layers_trainable = [2, 5, 9, 13, 17]
    for i, layer in enumerate(vgg16.layers):
        layer.trainable = (i in vgg16_layers_trainable)

    vgg16_enc['img_inp'] = vgg16.input
    vgg16_enc['inp'] = vgg16.layers[0].input
    vgg16_enc['b1c2'] = vgg16.layers[2].output
    vgg16_enc['b2c2'] = vgg16.layers[5].output
    vgg16_enc['b3c3'] = vgg16.layers[9].output
    vgg16_enc['b4c3'] = vgg16.layers[13].output
    vgg16_enc['b5c3'] = vgg16.layers[17].output
    vgg16_enc['b5p'] = vgg16.layers[18].output

    return vgg16_enc

def latent_layer(x, drpo_rate=None, enable_bn=False,
        kernel_regularizer=regularizers.l2(0.0001),
        bias_regularizer=regularizers.l2(0.0001)):
    x = Conv2D(512, (1, 1), strides=(1, 1),
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer)(x)
    if enable_bn:
        x = BatchNormalization()(x)
    x = activation_layer(x)
    if drpo_rate is not None:
        x = Dropout(drpo_rate)(x)

    x = upscale_layer(x, 512,
        drpo_rate=drpo_rate, enable_bn=enable_bn)

    return x

def upscale_layer(x, num_filters, filter_size=(3, 3), strides=(2, 2),
        padding='same', drpo_rate=None, enable_bn=False,
        kernel_regularizer=regularizers.l2(0.0001),
        bias_regularizer=regularizers.l2(0.0001)):
    x = Conv2DTranspose(num_filters, filter_size,
        strides=strides, padding=padding,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer)(x)
    # x = UpSampling2D(strides)(x)
    if enable_bn:
        x = BatchNormalization()(x)
    x = activation_layer(x)
    if drpo_rate is not None:
        x = Dropout(drpo_rate)(x)

    return x

def concat_conv_layer(x, ldr_layer, num_filters, filter_size=(1, 1),
        strides=(1, 1), kernel_regularizer=regularizers.l2(0.0001),
        bias_regularizer=regularizers.l2(0.0001), output_layer=False,
        drpo_rate=None, enable_bn=False):
    if output_layer:
        kernel_regularizer = None
        bias_regularizer = None
    if ldr_layer is not None:
        x = Concatenate(axis=-1)([x, ldr_layer])
    x = Conv2D(num_filters, filter_size, strides=strides,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer)(x)
    if enable_bn:
        x = BatchNormalization()(x)
    x = activation_layer(x, output_layer)
    if drpo_rate is not None:
        x = Dropout(drpo_rate)(x)

    return x

def activation_layer(x, output_layer=False):
    if not output_layer:
        return LeakyReLU(alpha=0.4)(x)
    else:
        return Activation('sigmoid')(x)

def hdr_decoder(x, ldr_enc, drpo_rate=None, enable_bn=False):
    x = concat_conv_layer(x, ldr_enc['b5c3'], 512,
        drpo_rate=drpo_rate, enable_bn=enable_bn)
    x = upscale_layer(x, 512,
        drpo_rate=drpo_rate, enable_bn=enable_bn)

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

    x = concat_conv_layer(x, None, 3,
        drpo_rate=drpo_rate, enable_bn=enable_bn)

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
    return K.mean(K.pow(a + b, 1.25))

def l2_loss(yTrue, yPred):
    return K.mean(K.square(yTrue - yPred))

def l1_loss(yTrue, yPred):
    return K.mean(K.abs(yTrue - yPred))

def custom_loss(yTrue, yPred):
    # return l1_loss(yTrue, yPred)
    # return l2_loss(yTrue, yPred)
    return total_variation_loss(yPred) + l1_loss(yTrue, yPred)

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

def psnr(yTrue, yPred):
    return 10.0 * K.log(1.0 / K.mean(K.square(yTrue - yPred))) / CONV_FACTOR

def assemble(drpo_rate, enable_bn):
    # Encoder (VGG16)
    ldr_enc = ldr_encoder()

    # Latent repr. layers
    x = latent_layer(ldr_enc['b5p'], drpo_rate, enable_bn)

    # Decoder
    x = hdr_decoder(x, ldr_enc, drpo_rate, enable_bn)

    model = Model(inputs=ldr_enc['img_inp'], outputs=x)

    optimi = Adam()

    model.compile(optimizer=optimi, loss=custom_loss, metrics=[psnr, ssim])

    return model

def train(model, XY_train, XY_dev, epochs, batch_size, es_fp):
    X_train, Y_train = XY_train
    X_dev, Y_dev = XY_dev

    model.summary()

    model_cbs = []
    model_cbs.append(history)
    if es_fp is not None:
        # model_cbs.append(EarlyStopping(monitor='val_psnr', min_delta=0.1,
        #     patience=2))
        model_cbs.append(ModelCheckpoint(monitor='val_psnr', filepath=es_fp,
            save_best_only=True))

    model.fit(X_train, Y_train, batch_size, epochs,
        validation_data=(X_dev, Y_dev),
        verbose=1,
        shuffle=True,
        callbacks=model_cbs)

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

    plt.plot(range(0,len(history.history["ssim"])), history.history["ssim"])
    plt.plot(range(0,len(history.history["val_ssim"])), history.history["val_ssim"])
    plt.legend(["train","dev"], loc='center left')
    plt.ylabel('SSIM')
    plt.xlabel('Epoch')
    plt.savefig(os.path.join(out_fd, 'ssim_chart.png'), dpi=100)
    plt.clf()

def save_final(m, out_fd):
    m.save(os.path.join(out_fd, 'model-final.h5'))
