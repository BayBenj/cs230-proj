#!/usr/bin/env python3

import argparse
import collections
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

    # Convert all raw images to 8-bit RGB
    snslugs_rgb = conv_raw_images()

    # Map RGB exposures w/ scene HDR render
    rend_rgb = link_test_pairs(snslugs_rgb)

    # Generate <LDR, HDR> samples dataset for NN
    gen_nn_dataset(rend_rgb)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocess HDR images for data pipeline to neural network')
    parser.add_argument('--input-raw-dir', dest='input_raw_dir', type=str,
        default='/proj/data/fchdr_raw_dataset')
    parser.add_argument('--input-rend-dir', dest='input_rend_dir', type=str,
        default='/proj/data/fchdr_rend_dataset')
    parser.add_argument('--output-rgb-dir', dest='output_rgb_dir', type=str,
        default='/proj/data/fchdr_rgb_dataset')
    parser.add_argument('--output-nn-dir', dest='output_nn_dir', type=str,
        default='/proj/data/proc_nn_dataset')
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

def parse_dims(dims):
    match = re.match(r'^(\d+)[Xx](\d+)$', dims)
    if not match:
        raise ValueError('parse_dims: invalid dimensions format' + dims)
    return int(match.group(1)), int(match.group(2))

def name_slugify(name):
    name = name.lower()
    name = name.replace('\'', '')
    name = re.sub(r'\W+', '-', name)
    if name.endswith('-'):
        name = name[0:-1]
    return name

def conv_raw_images():
    files_processed = False

    snslugs_rgb = []

    for rawds_root, scenes, _ in os.walk(args.input_raw_dir):
        num_scenes = len(str(len(scenes)))
        for scene_idx, scene in enumerate(sorted(scenes)):
            scene_prefix = 's{:0{num_scenes}}'.format(scene_idx+1,
                num_scenes=num_scenes)
            scene_name_slug = name_slugify(scene)
            scene_dirname = scene_prefix + '_' + scene_name_slug

            for scene_root, _, imgs in os.walk(os.path.join(rawds_root, scene)):
                rgb_filedir = os.path.join(args.output_rgb_dir, scene_dirname)

                raw_filenames = [img for img in imgs if \
                    img.lower().endswith('.nef')]

                raw_exp_times = []

                for raw_filename in sorted(raw_filenames):
                    raw_filepath = os.path.join(scene_root, raw_filename)
                    raw_suffix = os.path.splitext(raw_filename)[0].strip('_') \
                        .lower()

                    rgb_filename = '{}_{}.{}'.format(scene_prefix, raw_suffix,
                        args.output_fm)
                    rgb_filepath = os.path.join(rgb_filedir, rgb_filename)

                    exp_filename = scene_prefix + '_exp_times.txt'
                    exp_filepath = os.path.join(rgb_filedir, exp_filename)

                    # Cache the exposure times for HDR reconstruction
                    metadata = pyexiv2.ImageMetadata(raw_filepath)
                    metadata.read()
                    exp_time = float(metadata['Exif.Photo.ExposureTime'].value)
                    raw_exp_times.append((rgb_filepath, exp_time))

                    # Convert raw files to 8-bit RGB
                    if not(os.path.isfile(rgb_filepath)) or args.force:
                        if not os.path.exists(rgb_filedir):
                            os.makedirs(rgb_filedir)
                        print('Processing: ' + rgb_filepath)
                        print('  input_fn: ' + raw_filepath)
                        print('  exp_time: ' + str(exp_time))
                        with rawpy.imread(raw_filepath) as raw_file:
                            rgb = raw_file.postprocess()
                        imageio.imwrite(rgb_filepath, rgb)
                        file_processed = True

                    snslugs_rgb.append((scene_name_slug, rgb_filepath))

                # Write exp_times file
                if not(os.path.isfile(exp_filepath)) or args.force:
                    print('Processing: ' + exp_filepath)
                    print('  scene_root: ' + scene_root)
                    with open(exp_filepath, 'w') as exp_file:
                        for raw_exp_time in raw_exp_times:
                            exp_file.write('{},{}\n'.format(raw_exp_time[0],
                                raw_exp_time[1]))
                    file_processed = True

    if not files_processed:
        print('conv_raw_images: No files processed, use \'--force\' to ' \
            'overwrite existing outputs')

    return snslugs_rgb

def link_test_pairs(snslugs_rgb):
    snslugs = list(set([a for a, b in snslugs_rgb]))
    snslugs_norm = [re.sub(r'[()\-]', '', snslug.lower()) for snslug in snslugs]
    snslugs_norm_map = {}
    for i, snslug_norm in enumerate(snslugs_norm):
        snslugs_norm_map[snslugs[i]] = snslug_norm

    snslugs_norm_rend_map = {}

    rendds_root = args.input_rend_dir
    rend_filenames = [rf for rf in os.listdir(rendds_root) \
        if os.path.isfile(os.path.join(rendds_root, rf))]

    for rend_filename in rend_filenames:
        rend_norm = rend_filename.lower()
        rend_norm = re.sub(r'[()]', '', rend_norm)

        for snslug_norm in snslugs_norm:
            if rend_norm.startswith(snslug_norm):
                rend_filepath = os.path.join(rendds_root, rend_filename)
                snslugs_norm_rend_map[snslug_norm] = rend_filepath
                snslugs_norm.remove(snslug_norm)

    if len(snslugs_norm) > 0:
        print('link_test_pairs: Warning, unable to match the following raw ' \
            'scenes:')
        for snslug_norm in snslugs_norm:
            print('  ' + snslug_norm)

    rend_rgb = {}
    for i, (snslug, rgb_filepath) in enumerate(snslugs_rgb):
        snslug_norm = snslugs_norm_map[snslug]
        if snslug_norm in snslugs_norm_rend_map:
            rend_filepath = snslugs_norm_rend_map[snslug_norm]
            if rend_filepath not in rend_rgb:
                rend_rgb[rend_filepath] = []
            rend_rgb[rend_filepath].append(rgb_filepath)

    return rend_rgb

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

def gen_nn_dataset(rend_rgb=None):
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

    with tf.Session().as_default():
        samp_idx = 0

        for rend_fp, rgb_fps in sorted(rend_rgb.items()):
            print('Processing: ' + rend_fp)
            print('  len(rgb_fps): ' + str(len(rgb_fps)))
            samp_idx_init = samp_idx

            # Scene HDR render (ground truth)
            rend_h, rend_w, _ = plt.imread(rend_fp).shape
            rsz_w, rsz_h = get_rsz_short_dims(rend_w, rend_h)

            rend_rsz_w, rend_rsz_h = rsz_w, rsz_h

            rend_augm_ops = get_img_augm_ops(rend_fp, rsz_w, rsz_h)

            # Iterate through each LDR exposure (inputs)
            for rgb_fp in rgb_fps:
                rgb_rend_ops = get_img_augm_ops(rgb_fp, rsz_w, rsz_h)

                # Write out x,y sample pairs
                for i in range(len(rgb_rend_ops)):
                    x_fp = os.path.join(x_filedir, 'x{}.jpg'.format(samp_idx))
                    y_fp = os.path.join(y_filedir, 'y{}.jpg'.format(samp_idx))
                    if not(os.path.isfile(x_fp) and os.path.isfile(y_fp)) \
                            or args.force:
                        plt.imsave(x_fp, rgb_rend_ops[i].eval())
                        plt.imsave(y_fp, rend_augm_ops[i].eval())
                    samp_idx += 1

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
