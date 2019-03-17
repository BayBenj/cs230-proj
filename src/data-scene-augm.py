#!/usr/bin/env python3

import argparse
import collections
import json
import math
import os
import re

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import rawpy
import imageio
import pyexiv2

import pprint as pp

def main():
    global args

    args = parse_args()

    gen_nn_dataset()

def parse_args():
    parser = argparse.ArgumentParser(
        description='Align input rgb ldr images to rendered scene image')
    parser.add_argument('--input-rgb-dir', dest='input_rgb_dir', type=str,
        default='/proj/data/fchdr_rgb-algned_dataset')
    # parser.add_argument('--input-rend-dir', dest='input_rend_dir', type=str,
        # default='/proj/data/proj/data/fchdr_rend_dataset')
    parser.add_argument('--output-nn-dir', dest='output_nn_dir', type=str,
        default='/proj/data/proc_nn_dataset_aligned')
    parser.add_argument('--output-size', dest='output_size', type=str,
        default='224x224',
        help='Output image dimensions')
    parser.add_argument('--output-format', dest='output_fm', type=str,
        choices=['png', 'jpeg'], default='png',
        help='Output image RGB format')
    parser.add_argument('--force', dest='force', action='store_true',
        default=False,
        help='Force overwrite existing output files')

    return parser.parse_args()

def parse_scene_rgb():
    with open('scene-rend-rgb-map.txt', 'r') as rrgb_file:
        return json.load(rrgb_file)

def parse_dims(dims):
    match = re.match(r'^(\d+)[Xx](\d+)$', dims)
    if not match:
        raise ValueError('parse_dims: invalid dimensions format' + dims)
    return int(match.group(1)), int(match.group(2))

def get_rsz_short_dims(in_w, in_h):
    out_w, out_h = parse_dims(args.output_size)

    if in_h <= in_w:
        rsz_w = math.ceil(in_w * out_h / in_h)
        rsz_h = int(out_h)
    else:
        rsz_w = int(out_w)
        rsz_h = math.ceil(in_h * out_w / in_w)

    return rsz_w, rsz_h

def get_img_crps(in_w, in_h):
    out_w, out_h = parse_dims(args.output_size)

    img_crps = []
    long_px = 0
    if in_h <= in_w:
        while long_px + out_w < in_w:
            img_crps.append((long_px, 0))
            long_px += out_w
        img_crps.append((in_w-out_w, 0))
    else:
        while long_px + out_h < in_h:
            img_crps.append((0, long_px))
            long_px += out_h
        img_crps.append((0, in_h-out_h))

    return img_crps

def get_img_augm_ops(img_filepath, rend_rsz_w, rend_rsz_h):
    out_w, out_h = parse_dims(args.output_size)

    img = plt.imread(img_filepath)
    img_tf = tf.convert_to_tensor(img)

    augm_ops = []

    # Scale image to output size on short edge
    if os.path.splitext(img_filepath)[1] in ['.jpg', '.jpeg']:
        img_rsz = tf.cast(tf.image.resize_images(img_tf,
                size=(rend_rsz_h, rend_rsz_w),
                method=tf.image.ResizeMethod.BILINEAR),
            tf.uint8)
    else:
        img_rsz = tf.image.resize_images(img_tf,
            size=(rend_rsz_h, rend_rsz_w),
            method=tf.image.ResizeMethod.BILINEAR)

    # Chop up image on long edge
    img_crps = []
    for crp_w, crp_h in get_img_crps(rend_rsz_w, rend_rsz_h):
        img_crp = tf.image.crop_to_bounding_box(img_rsz,
            crp_h, crp_w, out_h, out_w)
        img_crps.append(img_crp)
        augm_ops.append(img_crp)

    # Horizontal + vertical flips on crops (mirror)
    for img_crp in img_crps:
        h_flip = tf.image.flip_left_right(img_crp)
        v_flip = tf.image.flip_up_down(img_crp)

        augm_ops.append(h_flip)
        augm_ops.append(v_flip)
        augm_ops.append(tf.image.flip_left_right(v_flip))

    # 90 deg. rotations on crops
    for img_crp in img_crps:
        augm_ops.append(tf.image.rot90(img_crp, 1))
        augm_ops.append(tf.image.rot90(img_crp, 2))
        augm_ops.append(tf.image.rot90(img_crp, 3))

    return augm_ops

def gen_nn_dataset():
    scenes_obj = parse_scene_rgb()
    rend_rgb = sorted(scenes_obj, key=lambda sn: sn['rend_fp'])
    rend_rgb = [(sn['rend_fp'], sn['ldr_fps'], sn['ldr_fd']) \
        for sn in rend_rgb]
    pp.pprint(rend_rgb)

    x_filedir = os.path.join(args.output_nn_dir, 'x')
    y_filedir = os.path.join(args.output_nn_dir, 'y')

    scene_exmp_fp = os.path.join(args.output_nn_dir, 'scene_exmp_map.txt')

    if not os.path.exists(x_filedir):
        os.makedirs(x_filedir)
    if not os.path.exists(y_filedir):
        os.makedirs(y_filedir)

    nn_dataset = []
    out_w, out_h = parse_dims(args.output_size)

    scene_exmp_num = []

    samp_idx = 0

    for rend_fp, rgb_fps, rgb_fd in rend_rgb:
        # use aligned images
        rgb_fps = [os.path.join(args.input_rgb_dir, fp.split('/')[-2],
            fp.split('/')[-1]) \
                for fp in rgb_fps]

        print('Processing: ' + rend_fp)
        print('  len(rgb_fps): ' + str(len(rgb_fps)))
        samp_idx_init = samp_idx

        # Scene HDR render (ground truth)
        rend_h, rend_w, _ = plt.imread(rend_fp).shape
        rsz_w, rsz_h = get_rsz_short_dims(rend_w, rend_h)

        rend_rsz_w, rend_rsz_h = rsz_w, rsz_h

        rend_augm_ops = get_img_augm_ops(rend_fp, rsz_w, rsz_h)

        # Iterate through each LDR exposure (inputs)
        with tf.Session().as_default():
            print('  inputs:')
            for rgb_fp_idx, rgb_fp in enumerate(rgb_fps):
                print('    ldr_fp[{}]: {}'.format(rgb_fp_idx, rgb_fp))
                rgb_rend_ops = get_img_augm_ops(rgb_fp, rsz_w, rsz_h)

                # Write out x,y sample pairs
                for i in range(len(rgb_rend_ops)):
                    samp_idx_pd = '{:06d}'.format(samp_idx)
                    x_fp = os.path.join(x_filedir, '{}-x.jpg'.format(samp_idx_pd))
                    y_fp = os.path.join(y_filedir, '{}-y.jpg'.format(samp_idx_pd))
                    if not(os.path.isfile(x_fp) and os.path.isfile(y_fp)) \
                            or args.force:
                        plt.imsave(x_fp, rgb_rend_ops[i].eval())
                        plt.imsave(y_fp, rend_augm_ops[i].eval())
                    samp_idx += 1
        tf.reset_default_graph()

        # Log number of examples in scene
        num_added = samp_idx - samp_idx_init
        scene_exmp_num.append((rend_fp, num_added))

        print('  added {} samples'.format(num_added))

    print('gen_nn_dataset: added {} training samples', samp_idx)

    with open(scene_exmp_fp, 'w') as semp_file:
        print('Processing: ' + scene_exmp_fp)

        for (scene_fp, num_exmp) in scene_exmp_num:
            semp_file.write('{},{}\n'.format(scene_fp, num_exmp))

        print('  Complete.')

if __name__ == '__main__':
    main()
