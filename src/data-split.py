#!/usr/bin/env python3

import argparse
import math
import os
import random
import re
import shutil

import pprint as pp

def main():
    global args

    args = parse_args()

    split_datasets()

def parse_args():
    parser = argparse.ArgumentParser(
        description='Split data into into train/dev/test sets')
    parser.add_argument('--input-nn-dir', dest='input_nn_dir', type=str,
        default='/proj/data/proc_nn_dataset')
    parser.add_argument('--input-format', dest='input_fm', type=str,
        choices=['jpg', 'png'], default='jpg')
    parser.add_argument('--split-ratio', dest='split_ratio', type=str,
        default='80,10,10')
    parser.add_argument('--output-split-dir', dest='output_split_dir', type=str,
        default='/proj/data/split_nn_dataset')

    args = parser.parse_args()

    ratios = [int(ra) for ra in args.split_ratio.split(',')]
    assert(len(ratios) == 3)
    assert(sum(ratios) == 100)
    args.split_ratio = ratios

    return args

def split_datasets():
    print('Processing train/dev/test dataset split')
    print('  Split: {}/{}/{}'.format(args.split_ratio[0], args.split_ratio[1],
        args.split_ratio[2]))

    scene_exmp_fp = os.path.join(args.input_nn_dir, 'scene_exmp_map.txt')
    print('  Scene-No. of Examples: ' + scene_exmp_fp)

    scene_exmp_num = []

    with open(scene_exmp_fp, 'r') as semp_file:
        for line in semp_file:
            vals = line.split(',')
            assert(len(vals) == 2)

            scene_exmp_num.append((vals[0], int(vals[1])))

    scene_exmp_rng = {}
    sns_idx = 0
    for (scene_fp, num_exmp) in scene_exmp_num:
        sne_idx = sns_idx + num_exmp
        scene_exmp_rng[scene_fp] = (sns_idx, sne_idx)
        sns_idx = sne_idx

    # Shuffle examples and copy files
    RANDOM_SEED = 3141516
    random.seed(RANDOM_SEED)

    print('  Shuffling examples w/ seed: ' + str(RANDOM_SEED))
    random.shuffle(scene_exmp_num)

    split_name = ['train', 'dev', 'test']
    split_parts = [round((ra/100.0) * len(scene_exmp_num)) \
        for ra in args.split_ratio]
    if sum(split_parts) != len(scene_exmp_num):
        split_parts[0] += 1
    assert(sum(split_parts) == len(scene_exmp_num))

    sns_idx = 0
    for sn, split in enumerate(args.split_ratio):
        scene_dir = os.path.join(args.output_split_dir, split_name[sn])

        scene_x_dir = os.path.join(scene_dir, 'x')
        scene_y_dir = os.path.join(scene_dir, 'y')
        os.makedirs(scene_x_dir, exist_ok=True)
        os.makedirs(scene_y_dir, exist_ok=True)

        print('  Copying {} split'.format(split_name[sn]))
        print('    output: ' + str(scene_dir))
        print('    num_scenes: ' + str(split_parts[sn]))

        sne_idx = sns_idx + split_parts[sn]
        for sn_idx in range(sns_idx, sne_idx):
            eis_idx, eie_idx = scene_exmp_rng[scene_exmp_num[sn_idx][0]]
            for ei in range(eis_idx, eie_idx):
                exmp_x_ofp = os.path.join(args.input_nn_dir, 'x',
                    'x{}.{}'.format(ei, args.input_fm))
                exmp_y_ofp = os.path.join(args.input_nn_dir, 'y',
                    'y{}.{}'.format(ei, args.input_fm))
                shutil.copy(exmp_x_ofp, scene_x_dir)
                shutil.copy(exmp_y_ofp, scene_y_dir)

        sns_idx = sne_idx

    print('  Complete.')

if __name__ == '__main__':
    main()
