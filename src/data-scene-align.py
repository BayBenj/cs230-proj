#!/usr/bin/env python3

import argparse
import numpy as np
import cv2
import json
import os

import pprint as pp

def main():
    global args

    args = parse_args()

    align_scene_images()

def parse_args():
    parser = argparse.ArgumentParser(
        description='Align input rgb ldr images to rendered scene image')
    parser.add_argument('--input-rgb-dir', dest='input_rgb_dir', type=str,
        default='/proj/data/fchdr_rgb_dataset')
    parser.add_argument('--input-rend-dir', dest='input_rend_dir', type=str,
        default='/proj/data/proj/data/fchdr_rend_dataset')
    parser.add_argument('--output-rgb-algnd-dir', dest='output_rgb_algnd_dir', type=str,
        default='/proj/data/fchdr_rgb-algned_dataset')
    parser.add_argument('--force', dest='force', action='store_true',
        default=False,
        help='Force overwrite existing output files')

    return parser.parse_args()

def parse_scene_rgb():
    with open('scene-rend-rgb-map.txt', 'r') as rrgb_file:
        return json.load(rrgb_file)

def align_scene_images():
    scenes_obj = parse_scene_rgb()

    # Define the motion model
    warp_mode = cv2.MOTION_AFFINE

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
            warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
            warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 5000

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    # termination_eps = 1e-10
    termination_eps = 1e-8

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    converge_error_files = []

    for i, scene in enumerate(scenes_obj):
        algnd_fd = os.path.join(args.output_rgb_algnd_dir, scene['ldr_fd'])
        os.makedirs(algnd_fd, exist_ok=True)

        print('Processing: ' + algnd_fd)
        print('  rend_fp: ' + scene['rend_fp'])

        rend_img = cv2.imread(scene['rend_fp'])

        # rend_w = rend_img.shape[]
        # rend_h = rend_img.shape[]
        # size_req = (rend_img.shape[1], rend_img.shape[0])
        sz = rend_img.shape

        for ldr_fp in scene['ldr_fps']:
            rgb_algnd_fp = os.path.join(algnd_fd, ldr_fp.split('/')[-1])
            if os.path.isfile(rgb_algnd_fp):
                continue
            print('  ldr_fp[{}]: {}'.format(i, rgb_algnd_fp))

            rgb_img = cv2.imread(ldr_fp)
            rgb_img = cv2.resize(rgb_img, dsize=(sz[1], sz[0]), interpolation=cv2.INTER_CUBIC)

            rend_gimg = cv2.cvtColor(rend_img, cv2.COLOR_RGB2GRAY)
            rgb_gimg = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

            try:
                cc, warp_matrix = cv2.findTransformECC(rend_gimg, rgb_gimg, warp_matrix, warp_mode, criteria)
                print(warp_matrix)
            except:
                print('    findTransformECC error:' + rgb_algnd_fp)
                print('      input: ' + ldr_fp)
                print('      target: ' + rgb_algnd_fp)
                print('      Skipping.')
                converge_error_files.append(rgb_algnd_fp)
                continue

            if warp_mode == cv2.MOTION_HOMOGRAPHY :
                # Use warpPerspective for Homography
                rgb_algnd_img = cv2.warpPerspective(rgb_img, warp_matrix, (sz[1],sz[0]),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            else :
                # Use warpAffine for Translation, Euclidean and Affine
                rgb_algnd_img = cv2.warpAffine(rgb_img, warp_matrix, (sz[1], sz[0]),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)


            cv2.imwrite(rgb_algnd_fp, rgb_algnd_img)
            
    if len(converge_error_files) > 0:
        print('The following files failed to converge or otherwise errored:')
        for rgb_fp in converge_error_files:
            print('  ' + rgb_fp)
            
    print('Complete.')

if __name__ == '__main__':
    main()
