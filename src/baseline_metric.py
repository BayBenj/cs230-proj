#!/usr/bin/env python3

import argparse
import os
import re
import math
import string

import numpy as np
import cv2
import rawpy
import imageio
import pyexiv2

import shutil

import pprint as pp

def main():
    global args

    args = parse_args()
#    fix_rendered_names()
    get_image_psnr()

def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocess HDR images for data pipeline to neural network')
    parser.add_argument('ground_truth_dir', type=str,
        default='../data/hdr_image_dataset')
    parser.add_argument('output_dir', type=str,
        default='../data/proc_image_dataset')
    parser.add_argument('--output-size', dest='output_size', type=str,
        default='300x300')
    parser.add_argument('--force', dest='force', action='store_true',
        default=False,
        help='Force overwrite existing output files')

    return parser.parse_args()

def name_slugify(name):
    name = name.lower()
    name = name.replace('\'', '')
    name = re.sub(r'\W+', '-', name)
    if name.endswith('-'):
        name = name[0:-1]
    return name

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    print(mse)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def fix_rendered_names():
    for ds_root, _, rend_files in os.walk(args.ground_truth_dir):
        for filename in rend_files:
            filepath = os.path.join(ds_root, filename)
            hdr_img = cv2.imread(filepath)
            out_filename = name_slugify(os.path.splitext(filename)[0]) + '.jpg'
            print(filename)
            print(out_filename)
            out_filepath = os.path.join(ds_root, out_filename)
            print(out_filepath)
            shutil.copy(filepath, out_filepath)
            #cv2.imwrite(out_filepath, hdr_img)

def get_image_psnr():
    for ds_root, scenes, _ in os.walk(args.output_dir):
        for scene_idx, scene in enumerate(sorted(scenes)):
            for sn_root, _, files in os.walk(os.path.join(ds_root, scene)):

                for filename in files:
                    filepath = os.path.join(sn_root, filename)
                    if re.match(r'_MDF.*.png', filename):
                        ldr_img = cv2.imread(filepath)
                        #print(filepath)
                    elif re.match(r's.*hdr.*.png', filename):
                        tnmp_img = cv2.imread(filepath)
                        #print(filepath)
            for gt_root, _, rend_files in os.walk(args.ground_truth_dir):
                for filename in rend_files:
                    filepath = os.path.join(gt_root, filename)
                    check_str = re.sub(r"s(\d+)_", "", scene)
                    check_str = re.sub(r"-", "", check_str)
                    #print(check_str)
                    #print(filename)
                    if re.match(check_str, filename):
                        hdr_img = cv2.imread(filepath)
                        #print(filepath)
            size_req = (hdr_img.shape[1], hdr_img.shape[0])
            tnmp_img_res = cv2.resize(tnmp_img, dsize=size_req, interpolation=cv2.INTER_CUBIC)
            ldr_img_res = cv2.resize(ldr_img, dsize=size_req, interpolation=cv2.INTER_CUBIC)
            print("PSNR for scene " + scene + ":" + str(psnr(tnmp_img_res, hdr_img)))
            #print("PSNR for single LDR: " + str(psnr(ldr_img_res, hdr_img)))

#    hdr_img = cv2.imread("507Rendered.jpg")
#    tnmp_img = cv2.imread("s1_hdr-reinhard.png")
#    ldr_img = cv2.imread("_MDF0001.png")
#    print(hdr_img.shape)
#    print(tnmp_img.shape)
#    print(ldr_img.shape)
#    tnmp_img_res = cv2.resize(tnmp_img, dsize=(600, 398), interpolation=cv2.INTER_CUBIC)
#    ldr_img_res = cv2.resize(ldr_img, dsize=(600, 398), interpolation=cv2.INTER_CUBIC)
#    print(tnmp_img_res.shape)
#    print(ldr_img_res.shape)
    

#    d = psnr(tnmp_img, ldr_img)
#    print(d)
#    d = psnr(tnmp_img_res, ldr_img_res)
#    print(d)
#    d = psnr(tnmp_img_res, hdr_img)
#    print(d)
#    d = psnr(ldr_img_res, hdr_img)
#    print(d)

#
#def proc_raw_images():
#    for ds_root, scenes, _ in os.walk(args.input_dir):
#        for scene_idx, scene in enumerate(sorted(scenes)):
#            proc_scenedir = 's{}_{}'.format(scene_idx+1, name_slugify(scene))
#            for sn_root, _, raw_files in os.walk(os.path.join(ds_root, scene)):
#                proc_filedir = os.path.join(args.output_dir, proc_scenedir)
#                exp_times = []
#
#                for raw_filename in raw_files:
#                    raw_filepath = os.path.join(sn_root, raw_filename)
#
#                    proc_filename = os.path.splitext(raw_filename)[0] + '.png'
#                    proc_filepath = os.path.join(proc_filedir, proc_filename)
#
#                    # Cache the exposure times for HDR reconstruction
#                    metadata = pyexiv2.ImageMetadata(raw_filepath)
#                    metadata.read()
#                    exp_time = float(metadata['Exif.Photo.ExposureTime'].value)
#                    exp_times.append((proc_filepath, exp_time))
#
#                    # Convert raw files to 8-bit RGB
#                    if not(os.path.isfile(proc_filepath)) or args.force:
#                        if not os.path.exists(proc_filedir):
#                            os.makedirs(proc_filedir)
#                        print('Processing: {}'.format(proc_filepath))
#                        with rawpy.imread(raw_filepath) as raw_file:
#                            rgb = raw_file.postprocess()
#                        imageio.imwrite(proc_filepath, rgb)
#
#                # Generate HDR image using OpenCV
#                if len(exp_times) == 0:
#                    print('Unable to retrieve exposure times for raws in: {}' \
#                        .format(proc_filedir))
#                    print('Skipping HDR image reconstruction')
#                    break
#
#                proc_hdr_filepath = os.path.join(proc_filedir,
#                    's{}_hdr-{}.png'.format(scene_idx+1, args.tone_map_algo))
#
#                if os.path.isfile(proc_hdr_filepath) and not(args.force):
#                    return
#
#                print('Processing: {}'.format(proc_hdr_filepath))
#
#                ldr_images = []
#                ldr_exp_times = []
#                for rgb_filepath, exp_time in exp_times:
#                    ldr_images.append(cv2.imread(rgb_filepath))
#                    ldr_exp_times.append(exp_time)
#                ldr_exp_times = np.array(ldr_exp_times, dtype=np.float32)
#
#                # Estimate Camera response function, merge exposures
#                calib_debevec = cv2.createCalibrateDebevec()
#                resp_debevec = calib_debevec.process(ldr_images, ldr_exp_times)
#                merge_debevec = cv2.createMergeDebevec()
#                hdr_debevec = merge_debevec.process(ldr_images, ldr_exp_times,
#                    resp_debevec)
#
#                # Tone mapping
#                if args.tone_map_algo == 'reinhard':
#                    tone_map = cv2.createTonemapReinhard(1.5, 0,0,0)
#                    ldr_output = tone_map.process(hdr_debevec)
#                    cv2.imwrite(proc_hdr_filepath, ldr_output * 255)
#                elif args.tone_map_algo == 'durand':
#                    tone_map = cv2.createTonemapDurand(1.5,4,1.0,1,1)
#                    ldr_output = 3 * tone_map.process(hdr_debevec)
#                    cv2.imwrite(proc_hdr_filepath, ldr_output * 255)
#
#def resize_proc_images():
#    pass
#
#def write_test_pairs():
#    pass

if __name__ == '__main__':
    main()
