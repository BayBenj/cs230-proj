#!/usr/bin/env python3

import argparse
import os

import numpy as np
import pprint as pp
import matplotlib.pyplot as plt
import time as clock
import datetime
import pytz
import math

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.utils import plot_model

# import model_upsample as model
import model as model

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
    start_time = timestamp()
    args = parse_args()
    args.func(args, start_time)

def train_model(args, start_time):
    m = model.assemble(args.drpo_rate, args.enable_bn)
    plot_model(m, to_file='model.png')

    XY_train, XY_dev = load_trdev_datasets()

    start = clock.time()

    out_fd = os.path.join(args.output_dir, 'model_{}_epch{}_bs{}_trn{}_val{}' \
        .format(start_time, args.num_epochs, args.batch_size,
            len(XY_train[0]), len(XY_dev[0])))
    os.makedirs(out_fd)

    es_fp = os.path.join(out_fd, 'model-es.h5') if args.save_model else None

    model.train(m, XY_train, XY_dev, args.num_epochs, args.batch_size, es_fp)
    end = clock.time()
    print("Training model took {} seconds.".format(end - start))

    # Save outputs
    print('Processing: ' + out_fd)

    model.write_summary(m, out_fd)
    if args.save_model:
        model.save_final(m, out_fd)
    model.plot(out_fd)

    # Predict a few samples
    train_idx = [0, 60, 100, 400]
    for i in train_idx:
        model.predict_imgs(m, (XY_train[0][i:i+1], XY_train[1][i:i+1]),
            out_fd, 'train' + str(i))

    dev_idx = [0, 10, 20, 40]
    for i in dev_idx:
        model.predict_imgs(m, (XY_dev[0][i:i+1], XY_dev[1][i:i+1]),
            out_fd, 'dev' + str(i))

    print('  Complete.')


def predict_model(args, start_time):
    m = model.assemble(None, False)
    m.load_weights(args.model_h5)

    print('Processing: ' + args.input_dir)
    print('  Note: If you see a TF exception w/ BaseSession.__del__, that is ' \
        'a known and harmless bug')

    os.makedirs(args.output_dir, exist_ok=True)

    for i, img_fn in enumerate(os.listdir(args.input_dir)):
        img_fp = os.path.join(args.input_dir, img_fn)

        img_out_fn = os.path.splitext(img_fn)[0] + '.jpg'
        img_out_fp = os.path.join(args.output_dir, img_out_fn)

        print('  output: ' + img_out_fn)

        img_x = np.asarray(image.img_to_array(image.load_img(img_fp))) / 255.0

        img_pred = m.predict(np.array([img_x])) * 255.0
        img_out = image.array_to_img(img_pred[0])
        img_out.save(img_out_fp)

def evaluate_model(args, start_time):
    m = model.assemble(None, False)
    m.load_weights(args.model_h5)

    eval_fd = os.path.join(args.input_dir, args.eval_dataset)

    print('Processing: ' + eval_fd)

    X_eval, Y_eval = load_dataset(eval_fd, None)

    res = m.evaluate(X_eval, Y_eval, verbose=1)
    print('  model: ' + args.model_h5)
    for i, metric_name in enumerate(m.metrics_names):
        print('    {}: {}'.format(metric_name, res[i]))

def parse_args():
    parser = argparse.ArgumentParser(
        description='Script to train HDR inference model')

    subparsers = parser.add_subparsers(help='Select model operation')
    subparsers.dest = 'model_op'
    subparsers.required = True

    train_parser = subparsers.add_parser('train',
        help='Train neural network model model')
    train_parser.add_argument('--input-dir', dest='input_dir', type=str,
        default='/proj/data/split_nn_dataset_aligned')
    train_parser.add_argument('--output-dir', dest='output_dir', type=str,
        default='./model_out/')
    train_parser.add_argument('--save-model', dest='save_model',
        action='store_true', default=False)
    train_parser.add_argument('--val-split', dest='val_split', type=float,
        default=0.8)
    train_parser.add_argument('--max-samples', dest='max_samples', type=int,
        default=None)
    train_parser.add_argument('--epochs', dest='num_epochs', type=int,
        default=5)
    train_parser.add_argument('--batch_size', dest='batch_size', type=int,
        default=32)
    train_parser.add_argument('--drpo-rate', dest='drpo_rate', type=float,
        default=None)
    train_parser.add_argument('--enable-bn', dest='enable_bn',
        action='store_true', default=False)
    train_parser.set_defaults(func=train_model)

    predict_parser = subparsers.add_parser('predict',
        help='Predict HDR image using neural network model model')
    predict_parser.add_argument('--input-dir', dest='input_dir', type=str,
        default='./pred-in-imgs')
    predict_parser.add_argument('--output-dir', dest='output_dir', type=str,
        default='./pred-out-imgs')
    predict_parser.add_argument('model_h5', type=str,
        default='model.h5')
    predict_parser.set_defaults(func=predict_model)

    eval_parser = subparsers.add_parser('eval',
        help='Evaluate neural network model against input dataset')
    eval_parser.add_argument('--input-dir', dest='input_dir', type=str,
        default='/proj/data/split_nn_dataset_aligned')
    eval_parser.add_argument('--eval-dataset', dest='eval_dataset', type=str,
        choices=['train', 'dev', 'test'], default='test')
    eval_parser.add_argument('model_h5', type=str,
        default='model.h5')
    eval_parser.set_defaults(func=evaluate_model)

    args = parser.parse_args()

    if args.model_op == 'train':
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
        imgy_fn = imgx_fn[:-5] + 'y.jpg'
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

    x = np.asarray(x) / 255.0
    print(x.shape)
    y = np.asarray(y) / 255.0
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

    if args.max_samples is not None:
        max_train_samples = args.val_split * args.max_samples
        max_dev_samples = math.ceil((1.0 - args.val_split) * args.max_samples)
    else:
        max_train_samples = None
        max_dev_samples = None

    return load_dataset(train_fd, max_train_samples), \
        load_dataset(dev_fd, max_dev_samples)

if __name__ == '__main__':
    main()
