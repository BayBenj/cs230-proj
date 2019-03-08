
import keras
from keras.applications.vgg16 import VGG16

model = VGG16()
print(model.summary())

# keras.layers.convolutional.Deconvolution2D(nb_filter, nb_row, nb_col, output_shape, init='glorot_uniform', activation=None, weights=None, border_mode='valid', subsample=(1, 1), dim_ordering='default', W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True)


