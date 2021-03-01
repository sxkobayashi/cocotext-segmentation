"""
Visualize the segmentation masks.
Assumes for each file in mask_dir, there's a matching file in dataset_dir.
"""
import argparse
import os, sys, glob, random, csv
import numpy as np
from PIL import Image

from visutils import draw_overlay_image, draw_boxes
from box_utils import read_boxes_from_csv


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Visualize dataset.")
    parser.add_argument('mask_dir', help="Root path to mask dataset")
    parser.add_argument('dataset_dir', help="Root path to training dataset")
    parser.add_argument('output_dir', help="Root path to visualization output")
    parser.add_argument('--box_file', nargs='?', type=str, default="",
        help="CSV file containing all boxes.")
    parser.add_argument('--max_images', nargs='?', type=int, default=-1,
        help="Maximum number of images to visualize. Default is visualize all images.")
    parser.add_argument('--random_seed', nargs='?', type=int, default=-1, 
        help="Random seed. Randomly selects max_images to visualize")

    # Argument checks
    args = parser.parse_args()
    if not os.path.exists(args.mask_dir):
        raise ValueError(f'{mask_dir} does not exist.')
    if not os.path.exists(args.dataset_dir):
        raise ValueError(f'{dataset_dir} does not exist.')

    # Get masks' corresponding image file
    mask_files = glob.glob(os.path.join(args.mask_dir, '*.jpg'))
    print(f'{len(mask_files)} patches are found.')

    image_files = []
    for mask_file in mask_files:
        bname = os.path.basename(mask_file)
        image_file = os.path.join(args.dataset_dir, bname)
        if not os.path.exists(image_file):
            print(f"{image_file} does not exist. Skipping.")
        image_files.append((image_file, mask_file))

    # Random shuffle
    if args.max_images > 0:
        if args.random_seed > 0:
            random.Random(args.random_seed).shuffle(image_files)
        image_files = image_files[0:args.max_images]

    box_dict = None
    if len(args.box_file) > 0:
        box_dict = read_boxes_from_csv(args.box_file)

    os.makedirs(args.output_dir, exist_ok=True)
    for img_file, mask_file in image_files:
        img = np.asarray(Image.open(img_file)) # uint8
        mask = np.asarray(Image.open(mask_file)).astype('float')[:,:,0] / 255.0
        overlay_img = draw_overlay_image(img, mask, threshold=0.5)
        overlay_image = Image.fromarray(overlay_img)
        base_name = os.path.basename(img_file)
        if box_dict is not None:
            box = box_dict.get(base_name)
            if box is not None:
                draw_boxes(overlay_image, [box])
        out_filename = os.path.join(args.output_dir, base_name)
        print(f'Writing to {out_filename}...')
        overlay_image.save(out_filename)
