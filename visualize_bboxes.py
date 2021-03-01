"""
Visualize the segmentation masks.
Assumes for each file in mask_dir, there's a matching file in dataset_dir.
"""
import argparse
import os, sys, glob, random

import coco_text
from visutils import draw_bounding_boxes

MINIMUM_BOX_AREA = 200

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Visualize dataset.")
    parser.add_argument('dataset_dir', help="Root path to training dataset")
    parser.add_argument('cocotext', help="Path to cocotext json file")
    parser.add_argument('output_dir', help="Root path to visualization output")
    parser.add_argument('--train_or_val', nargs='?', default='train', const='train', choices=('train','val'),
            help="Whether to generate from training set or validation set.")
    parser.add_argument('--max_images', nargs='?', type=int, default=-1,
        help="Maximum number of images to visualize. Default is visualize all images.")
    parser.add_argument('--random_seed', nargs='?', type=int, default=-1, 
        help="Random seed. Randomly selects max_images to visualize")

    # Argument checks
    args = parser.parse_args()
    if not os.path.exists(args.dataset_dir):
        raise ValueError('Dataset path {args.dataset_dir} does not exist.')

    ct = coco_text.COCO_Text(args.cocotext)

    if args.train_or_val == 'train':
        img_id_set = ct.train
    elif args.train_or_val == 'val':
        img_id_set = ct.val
    else:
        print("'train_or_val' should be either 'train' or 'val'")
        sys.exit(0)

    img_ids = ct.getImgIds(imgIds=img_id_set, catIds=[('legibility','legible')])
    print(f"Found {len(img_ids)} images.")
    img_info = ct.loadImgs(img_ids)
    
    # Random shuffle
    if args.max_images > 0:
        if args.random_seed > 0:
            random.Random(args.random_seed).shuffle(img_info)
        img_info = img_info[0:args.max_images]

    os.makedirs(args.output_dir, exist_ok=True)

    for info in img_info:
        print(f"Visualizing {info['file_name']} ...")
        annotationIds = ct.getAnnIds(imgIds=info['id'])
        annotations = ct.loadAnns(annotationIds)
        image_path = os.path.join(args.dataset_dir, info['file_name'])
        image = Image.open(image_path)
        vis_image = draw_bounding_boxes(image, annotations, MINIMUM_BOX_AREA)
        vis_image_path = os.path.join(args.output_dir, info['file_name'])
        vis_image.save(vis_image_path)

