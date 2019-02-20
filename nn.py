
import tensorlayer as tl
import tensorflow as tf


def conv(input_layer, size, name):
    nn = tl.layers.Conv2dLayer(input_layer,
                    act=tf.nn.relu,
                    shape=[3, 3, size[0], size[1]],
                    strides=[1, 1, 1, 1],
                    padding='SAME',
                    name=name)
    return nn


def max_pool(input_layer, name):
    nn = tl.layers.PoolLayer(input_layer,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='SAME',
                    pool=tf.nn.max_pool,
                    name=name)
    return nn


# LDR image input should be 320 x 320 x 3
def ldr_encoder(ldr_image):

    nn = tl.layers.InputLayer(ldr_image, name='input_layer')

    nn = conv(nn, [3, 64], 'h1/conv_1')
    nn = conv(nn, [64, 64], 'h1/conv_2')
    nn = conv(nn, [64, 64], 'h1/conv_2')
    nn = max_pool(nn, 'h1/pool')

    nn = conv(nn, [64, 128], 'h2/conv_1')
    nn = conv(nn, [128, 128], 'h2/conv_2')
    nn = max_pool(nn, 'h2/pool')

    nn = conv(nn, [128, 256], 'h3/conv_1')
    nn = conv(nn, [256, 256], 'h3/conv_2')
    nn = conv(nn, [256, 256], 'h3/conv_3')
    nn = max_pool(nn, 'h3/pool')

    nn = conv(nn, [256, 512], 'h4/conv_1')
    nn = conv(nn, [512, 512], 'h4/conv_2')
    nn = conv(nn, [512, 512], 'h4/conv_3')
    nn = max_pool(nn, 'h4/pool')

    nn = conv(nn, [512, 512], 'h5/conv_1')
    nn = conv(nn, [512, 512], 'h5/conv_2')
    nn = conv(nn, [512, 512], 'h5/conv_3')
    nn = max_pool(nn, 'h5/pool')

    return nn


