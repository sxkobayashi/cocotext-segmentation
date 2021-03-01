import argparse
import os, sys, csv
import numpy as np
from PIL import Image, ImageDraw

import coco_text
import box_utils

def create_annotaion_mask(annotations, height, width, skip_illegible=False, minimum_area=100):
    """ Create a RGB image array for text mask.
    """
    mask = np.zeros((height, width, 3), dtype='uint8')
    mask_image = Image.fromarray(mask)
    draw = ImageDraw.Draw(mask_image)

    if skip_illegible:
        annos = [anno for anno in annotations if anno['legibility'] == 'legible']
    else:
        annos = [anno for anno in annotations]

    if minimum_area is not None:
        annos = [anno for anno in annos if anno['area'] >= minimum_area]

    polygons = [anno['mask'] for anno in annos]
    for polygon in polygons:
        poly_x = polygon[0::2]
        poly_y = polygon[1::2]
        draw.polygon(list(zip(poly_x, poly_y)), fill=255)

    return mask_image


def positive_ratio(mask):
    return np.sum(mask / 255.0) / mask.size


def generate_3x3_patches(image, mask_image, output_img_dir, output_anno_dir, filename, min_area_ratio=0.05):
    # This is not used because it doesn't generate high quality patches
    img = np.asarray(image)
    if len(img.shape) == 2:
        img = np.repeat(img[:,:,np.newaxis], 3, axis=-1)

    mask = np.asarray(mask_image)
    height, width = img.shape[0:2]
    h_step = height // 3
    w_step = width // 3

    ratios = []
    for i in range(3):
        for j in range(3):
            img_patch = img[i*h_step: (i + 1) * h_step,
                            j*w_step: (j + 1) * w_step, :]
            mask_patch = mask[i*h_step: (i + 1) * h_step,
                              j*w_step: (j + 1) * w_step, :]
            ratio = positive_ratio(mask_patch)
            ratios.append(ratio)
            if ratio < min_area_ratio:
                continue
            img_patch_path = os.path.join(output_img_dir, f"{filename[:-4]}_x{i}_y{j}.jpg")
            anno_patch_path = os.path.join(output_anno_dir, f"{filename[:-4]}_x{i}_y{j}.jpg")
            Image.fromarray(img_patch).save(img_patch_path)
            Image.fromarray(mask_patch).save(anno_patch_path)

    return ratios


def generate_box_centered_patches(image, mask_image, annotations, output_img_dir, output_anno_dir, filename,
                                  min_crop_width=256, min_crop_height=256, minimum_area=100, skip_illegible=False):

    img = np.asarray(image)
    if len(img.shape) == 2:
        img = np.repeat(img[:,:,np.newaxis], 3, axis=-1)

    mask = np.asarray(mask_image)
    im_height, im_width = img.shape[0:2]

    if skip_illegible:
        annos = [anno for anno in annotations if anno['legibility'] == 'legible']
    else:
        annos = [anno for anno in annotations]

    boxes = [anno['bbox'] for anno in annos]

    box_utils.merge_bounding_boxes(boxes) # boxes will be changed
    boxes = box_utils.filter_small_boxes(boxes, minimum_area)

    boxes_patch = []
    for i, box0 in enumerate(boxes):
        box = box_utils.cut_within_frame(box0, im_width, im_height)
        x,y,w,h = box
        if x < 0 or y < 0 or w > im_width or h > im_height:
            raise ValueError('Original box out of original image bound. This should never happen.')

        x_center, y_center = x + w/2, y + h/2
        min_crop_box = box_utils.int_box([x_center - min_crop_width / 2, y_center - min_crop_height / 2, 
                                          min_crop_width,                min_crop_height])
        crop_box = box_utils.int_box(box_utils.rescale_box(box, 2.0))
        crop_box = box_utils.merge_box(min_crop_box, crop_box)
        crop_box = box_utils.cut_within_frame(crop_box, im_width, im_height)  # cut out over-the-limit box area

        x,y,w,h = crop_box
        img_patch = img[y:y+h, x:x+w, :]
        mask_patch = mask[y:y+h, x:x+w, :]

        box_patch = [box[0] - x, box[1] - y, box[2], box[3]]
        if not ( box_patch[0] >= 0 and box_patch[1] >= 0 and
                 box_patch[0] + box_patch[2] <= x + w and box_patch[1] + box_patch[3] <= y + h):
            raise ValueError('Original box out of crop bound. This should never happen.')

        basename = f"{filename[:-4]}_b{i}.jpg"
        boxes_patch.append((basename, box_patch))
        img_patch_path = os.path.join(output_img_dir, basename)
        anno_patch_path = os.path.join(output_anno_dir, basename)
        Image.fromarray(img_patch).save(img_patch_path, quality=100)
        Image.fromarray(mask_patch).save(anno_patch_path, quality=100)

    return boxes_patch


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Generate COCOText segementation dataset for text segmentation.")
    parser.add_argument('cocotext', help="Path to cocotext json file")
    parser.add_argument('dataset_dir', help="Path to MS COCO dataset train / val dataset")
    parser.add_argument('output_dir', help="Root path to output dataset (masks)")
    parser.add_argument('output_boxes', help="CSV file for output boxes")
    parser.add_argument('--train_or_val', nargs='?', default='train', const='train', choices=('train','val'),
            help="Whether to generate from training set or validation set.")
    parser.add_argument('--skip_writing', dest='skip_writing', action='store_true',
            help='If true, skip writing images out.')
    parser.set_defaults(skip_writing=False)
    args = parser.parse_args()

    if not os.path.exists(args.dataset_dir):
        raise ValueError('Dataset path {args.dataset_dir} does not exist.')
    ct = coco_text.COCO_Text(args.cocotext)

    output_img_dir = os.path.join(args.output_dir, args.train_or_val)
    output_anno_dir = os.path.join(args.output_dir, args.train_or_val + '_anno')
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_anno_dir, exist_ok=True)

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

    all_patch_boxes = []
    num_patches = 0
    for info in img_info:
        print(f"processing {info['file_name']} ...")
        annotationIds = ct.getAnnIds(imgIds=info['id'])
        annotations = ct.loadAnns(annotationIds)
        mask_image = create_annotaion_mask(annotations, info['height'], info['width'], skip_illegible=False, 
                                           minimum_area=0)
        image_path = os.path.join(args.dataset_dir, info['file_name'])
        image = Image.open(image_path)
        patch_boxes = generate_box_centered_patches(image, mask_image, annotations, output_img_dir, output_anno_dir,
                                                info['file_name'], min_crop_width=256, min_crop_height=256,
                                                minimum_area=200, skip_illegible=True)
        num_patches += len(patch_boxes)
        all_patch_boxes.extend(patch_boxes)

    print(f'{num_patches} patches are generated.')

    with open(args.output_boxes, 'w') as csvfile:
        wr = csv.writer(csvfile, delimiter=',')
        for line in all_patch_boxes:
            fname, box = line
            x,y,w,h = box
            wr.writerow([fname, x, y, w, h])

    print('Done.')
