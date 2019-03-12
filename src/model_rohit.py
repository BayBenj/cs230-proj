#!/usr/bin/env python3

import argparse
import os

import numpy as np
import pprint as pp
import matplotlib.pyplot as plt

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

def main():
    global args
    args = parse_args()
    x, y = load_datasets()
    model = train_model((x, y))
    predict_model(model, x[0:1])
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
        default=100)
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        default=32)

    return parser.parse_args()


def train_model(dataset):
    model = assemble_model()
    X, Y = dataset
    model.summary()
    model.fit(X, Y, epochs=args.num_epochs, batch_size=args.batch_size,
        verbose=1, validation_split=0.2, shuffle=True, callbacks=[history])
    
    return model


def predict_model(model, img):
    out_img = model.predict(img[0:1])
    out_img = (out_img + 1) * 127.5
    print(out_img.shape)
    pred_img = image.array_to_img(out_img[0])
    inp_img = image.array_to_img((img[0] + 1) * 127.5)
    #print(pred_img.shape)
    pred_img.save('out_img.jpg')
    inp_img.save('inp_img.jpg')


def custom_loss(yTrue, yPred):
    return K.mean(K.square(yTrue - yPred))


def psnr(yTrue, yPred):
    return 10 * K.log(2 / K.mean(K.square(yTrue - yPred))) / CONV_FACTOR
#    print(mse)
#    if mse == 0:
#        return 100
#    PIXEL_MAX = 255.0
#    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


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


def assemble_model():
    # Encoder (VGG16)
    inp_img, skip1, skip2, skip3, skip4, latent_rep, vgg16_input = ldr_encoder()

    # Decoder
    x = hdr_decoder(inp_img, skip1, skip2, skip3, skip4, latent_rep)

    model = Model(inputs=vgg16_input.input, outputs=x)
    model.compile(optimizer='adam', loss=custom_loss,
                  metrics=[psnr])

    return model


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


def plot():
    for metric, vals in history.history.items():
        plt.plot(range(1,len(vals) + 1),vals)
    plt.legend(history.history.keys(), loc='center left')
    #plt.ylabel('Loss')
    # plt.ylabel('Mean Squared Error')
    plt.xlabel('Epoch')
    plt.savefig('figure.png', dpi=100)


if __name__ == '__main__':
    main()
