#!/usr/bin/env python3

import argparse
import json
import os
import re
import math

import numpy as np
import cv2

import pprint as pp

def main():
    global args

    args = parse_args()

    scenes_psnr = calc_scene_hdr_psnr()

    calc_split_psnr(scenes_psnr)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate HDR images from LDR exposures using OpenCV and ' \
            'compute baseline PSNR against human renders')
    parser.add_argument('--scene-map', dest='scene_map_fp', type=str,
        default='scene-rend-rgb-map.txt')
    parser.add_argument('--split-map', dest='split_map_fp', type=str,
        default='scene-split-map.txt')
    parser.add_argument('--input-rend-dir', dest='input_rend_dir', type=str,
        default='/proj/data/fchdr_rend_dataset')
    parser.add_argument('--output-scene-bl-psnr', dest='output_scene_bl_psnr',
        type=str, default='scene-baseline-psnr.txt')
    parser.add_argument('--force', dest='force', action='store_true',
        default=False,
        help='Force overwrite existing output files')

    return parser.parse_args()

def parse_json_file(json_fp):
    with open(json_fp, 'r') as json_file:
        return json.load(json_file)

def get_scene_exptime_fp(scene_fd):
    scene_exptime_fn = [fn for fn in os.listdir(scene_fd) \
        if fn.endswith('.txt')]
    assert(len(scene_exptime_fn) == 1)
    return os.path.join(scene_fd, scene_exptime_fn[0])

def img_psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def calc_scene_hdr_psnr():
    scenes_obj = parse_json_file(args.scene_map_fp)

    scenes_psnr = {}

    for i, scene in enumerate(scenes_obj):
        scene_fd = os.path.dirname(scene['ldr_fps'][0])

        scene_rend_fp = scene['rend_fp']
        scene_cv_fp = os.path.join(scene_fd,
            '{}_cv-hdr.jpg'.format(scene['ldr_fd']))
        # scene_ldr_fps = scene['ldr_fps']
        scene_exptime_fp = get_scene_exptime_fp(scene_fd)

        print('Processing: ' + scene_fd)

        print('  scene_rend_fp (r): ' + scene_rend_fp)
        rend_hdr_img = cv2.imread(scene_rend_fp)

        # Render HDR image using OpenCV
        if args.force or not os.path.isfile(scene_cv_fp):
            print('  scene_cv_fp (w): ' + scene_cv_fp)
            ldr_imgs = []
            ldr_exptimes = []
            with open(scene_exptime_fp, 'r') as exptimes_file:
                for line in exptimes_file:
                    vals = line.split(',')
                    ldr_imgs.append(cv2.imread(os.path.join(scene_fd, vals[0].split('/')[-1])))
                    ldr_exptimes.append(float(vals[1]))

            ldr_exptimes = np.array(ldr_exptimes, dtype=np.float32)

            # Estimate Camera response function, merge exposures
            calib_debevec = cv2.createCalibrateDebevec()
            resp_debevec = calib_debevec.process(ldr_imgs, ldr_exptimes)
            merge_debevec = cv2.createMergeDebevec()
            hdr_debevec = merge_debevec.process(ldr_imgs, ldr_exptimes,
                resp_debevec)

            # Tone mapping
            tone_map = cv2.createTonemapReinhard(1.5, 0,0,0)
            cv_hdr_img = tone_map.process(hdr_debevec)
            cv2.imwrite(scene_cv_fp, cv_hdr_img * 255.0)
        else:
            print('  scene_cv_fp (r): ' + scene_cv_fp)
            cv_hdr_img = cv2.imread(scene_cv_fp)

        # Calculate PSNR
        cv_hdr_img = cv2.resize(cv_hdr_img,
            dsize=(rend_hdr_img.shape[1], rend_hdr_img.shape[0]),
            interpolation=cv2.INTER_CUBIC)
        scene_rend_fn = scene_rend_fp.split('/')[-1]
        scenes_psnr[scene_rend_fn] = img_psnr(cv_hdr_img, rend_hdr_img)

    return scenes_psnr

def calc_split_psnr(scenes_psnr):
    split_obj = parse_json_file(args.split_map_fp)

    with open(args.output_scene_bl_psnr, 'w') as psnr_res_file:
        for i, (split_name, rend_fns) in enumerate(sorted(split_obj.items())):
            split_str = 'split[{}] {} ({} items)'.format(i, split_name,
                len(rend_fns))
            print(split_str)
            psnr_res_file.write(split_str + '\n')

            split_psnr_avg = 0.0
            for rend_fn in rend_fns:
                scene_psnr = scenes_psnr[rend_fn]
                split_psnr_avg += scene_psnr
                scene_str = '  {:6.4f}: {}'.format(scene_psnr, rend_fn)
                print(scene_str)
                psnr_res_file.write(scene_str + '\n')
            split_psnr_avg /= len(rend_fns)

            split_psnr_stddev = 0.0
            for rend_fn in rend_fns:
                split_psnr_stddev += math.pow((scenes_psnr[rend_fn] - \
                    split_psnr_avg), 2)
            split_psnr_stddev /= len(rend_fns)
            split_psnr_stddev = math.sqrt(split_psnr_stddev)

            avg_str = '  split psnr avg.: {}'.format(split_psnr_avg)
            stddev_str = '  split psnr std. dev.: {}'.format(split_psnr_stddev)
            split_avg_stddev_str = '{}\n{}'.format(avg_str, stddev_str)
            print(split_avg_stddev_str)
            psnr_res_file.write(split_avg_stddev_str + '\n')

if __name__ == '__main__':
    main()
