"""
Visualize the segmentation masks.
Assumes for each file in mask_dir, there's a matching file in dataset_dir.
"""
import argparse
import os, sys, glob, random
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from visutils import draw_overlay_image, draw_boxes
from box_utils import read_boxes_from_csv
from dataset import COCOTextSegmentationDataset
from config import Configuration

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Visualize the data loader with augmented image and segmentation.")
    parser.add_argument('mask_dir', help="Root path to mask dataset")
    parser.add_argument('dataset_dir', help="Root path to training dataset")
    parser.add_argument('box_file', help="CSV file containing bounding boxes")
    parser.add_argument('output_dir', help="Root path to visualization output")
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

    box_dict = read_boxes_from_csv(args.box_file)
    image_files = []
    for mask_file in mask_files:
        bname = os.path.basename(mask_file)
        box = box_dict.get(bname)
        if box is None:
            print(f"Box for {bname} does not exist. Skipping.")
            continue
        image_file = os.path.join(args.dataset_dir, bname)
        if not os.path.exists(image_file):
            print(f"{image_file} does not exist. Skipping.")
            continue
        image_files.append((image_file, mask_file, box))

    # Random shuffle
    if args.max_images > 0:
        if args.random_seed > 0:
            random.Random(args.random_seed).shuffle(image_files)
        image_files = image_files[0:args.max_images]

    configs = Configuration()
    configs.batch_size = 1  # For memory considerations

    dataset = COCOTextSegmentationDataset(image_files, configs.im_size, configs.random_scale,
            configs.random_displacement, configs.random_flip)
    dataloader = DataLoader(dataset, batch_size=configs.batch_size,
            shuffle=False, num_workers=0)
    os.makedirs(args.output_dir, exist_ok=True)

    for i_batch, sample_batched in enumerate(dataloader):
        for i in range(configs.batch_size):
            if i >= sample_batched['image'].shape[0]:
                break
            img = sample_batched['image'][i].numpy().transpose((1,2,0))
            mask = sample_batched['mask'][i].numpy().transpose((1,2,0))

            overlay_img = draw_overlay_image(img, mask, threshold=0.5)
            overlay_image = Image.fromarray(overlay_img)
            out_filename = os.path.join(args.output_dir, f'b{i_batch}_s{i}.jpg')
            print(f'Writing to {out_filename}...')
            overlay_image.save(out_filename)
