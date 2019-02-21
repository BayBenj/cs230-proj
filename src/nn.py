
import tensorlayer as tl
import tensorflow as tf


SAME = "SAME"


def conv(input_layer, size, name):
    nn = tl.layers.Conv2dLayer(input_layer,
            padding=SAME,
            act=tf.nn.relu,
            shape=[3, 3, size[0], size[1]],
            strides=[1, 1, 1, 1],
            name=name)
    return nn


def deconv(input_layer, size, name):
    nn = tl.layers.DeConv2dLayer(input_layer,
            padding=SAME,
            act=tf.nn.relu,
            shape=[3, 3, size[0], size[1]],
            strides=[1, 1, 1, 1],
            name=name)
    nn = tl.layers.BatchNormLayer(nn, is_train=True, name="deconv_batch_norm_{}".format(name))
    return nn


def max_pool(input_layer, name):
    nn = tl.layers.PoolLayer(input_layer,
            padding=SAME,
            pool=tf.nn.max_pool,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            name=name)
    return nn


def load_vgg_params(nn, _file, session):
    import numpy as np
    params = []
    if not _file.lower().endswith(".npy"):
        print("Error in loading numpy params in {}. See file extension.".format(_file))
    else:
        numpy_load = np.load(_file, encoding="latin1")
        for key, val in sorted(numpy_load.item().items()):
            if(key[:4] == "conv"):
                params.append(val["weights"])
                params.append(val["biases"])
    tl.files.assign_params(session, params, nn)
    return nn


def ldr_encoder(ldr_image):
    nn = tl.layers.InputLayer(ldr_image, name="input_layer")

    nn = conv(nn, [3, 64], "l1/conv_1")
    nn = conv(nn, [64, 64], "l1/conv_2")
    nn = max_pool(nn, "l1/pool")

    nn = conv(nn, [64, 128], "l2/conv_1")
    nn = conv(nn, [128, 128], "l2/conv_2")
    nn = max_pool(nn, "l2/pool")

    nn = conv(nn, [128, 256], "l3/conv_1")
    nn = conv(nn, [256, 256], "l3/conv_2")
    nn = max_pool(nn, "l3/pool")

    nn = conv(nn, [256, 512], "l4/conv_1")
    nn = conv(nn, [512, 512], "l4/conv_2")
    nn = max_pool(nn, "l4/pool")

    return nn


def hdr_decoder(latent_representation):
    pass


sess = tf.InteractiveSession()

"""
    Note: Because of its size, vgg.npy is in the .gitignore.
    You will have to first download it yourself before loading.
"""
load_vgg_params(nn, "vgg16.npy", sess)

