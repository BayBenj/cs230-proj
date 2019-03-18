#!/usr/bin/env python3

import tensorflow as tf

with tf.Session() as sess:
    devices = sess.list_devices()
