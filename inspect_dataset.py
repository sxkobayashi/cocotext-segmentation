"""
Inspect masks for statistics.
"""
import argparse
import os, sys, glob, random, csv
import numpy as np
from PIL import Image


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Visualize dataset.")
    parser.add_argument('mask_dir', help="Root path to mask dataset")
    parser.add_argument('--max_images', nargs='?', type=int, default=-1,
        help="Maximum number of images to visualize. Default is visualize all images.")
    parser.add_argument('--random_seed', nargs='?', type=int, default=-1, 
        help="Random seed. Randomly selects max_images to visualize")

    # Argument checks
    args = parser.parse_args()
    if not os.path.exists(args.mask_dir):
        raise ValueError(f'{mask_dir} does not exist.')

    # Get masks' corresponding image file
    mask_files = glob.glob(os.path.join(args.mask_dir, '*.jpg'))
    print(f'{len(mask_files)} patches are found.')

    image_files = []
    for mask_file in mask_files:
        bname = os.path.basename(mask_file)
        image_files.append(mask_file)

    # Random shuffle
    if args.max_images > 0:
        if args.random_seed > 0:
            random.Random(args.random_seed).shuffle(image_files)
        image_files = image_files[0:args.max_images]

    threshold = 0.5
    ratios = []
    for mask_file in image_files:
        mask = np.asarray(Image.open(mask_file)).astype('float')[:,:,0] / 255.0
        pos_samples = np.abs(np.sum(1.0 * (mask >= threshold)))
        neg_samples = mask.size - pos_samples
        if neg_samples < 1:
            ratio = 100
        else:
            ratio = pos_samples / neg_samples
        ratios.append(ratio)
        print(f'pos = {pos_samples}, neg = {neg_samples}, ratio = {ratio}')
    print('Pos neg ratio = {}'.format(np.mean(ratios)))
