""" Visualization utilities
"""
import numpy as np
from PIL import Image, ImageDraw

from box_utils import merge_bounding_boxes, filter_small_boxes

def draw_overlay_image(image, mask, threshold=0.5):
    """ image: numpy array of shape (H,W,3), uint8 0 ~ 255 or float 0 ~ 1
    mask: numpy array of shape (H,W) or (H,W,1), float 0 ~ 1
    return: numpy array of shape (H,W,3), uint8 0 ~ 255 for display
    """
    img = image.astype('float')
    if image.dtype == np.uint8:
        img /= 255.0

    msk = mask.squeeze()
    overlay_img = img.copy()
    overlay_img[msk >= threshold, 1] = 0.8
    overlay_img[msk >= threshold, 0] *= 0.8
    overlay_img[msk >= threshold, 2] *= 0.8

    overlay_img = np.clip(255 * overlay_img, 0, 255).astype('uint8')
    return overlay_img


def draw_boxes(image, boxes):
    """ image: PIL Image
    boxes: a list of 4-tuples [(x,y,width,height)]
    """
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x,y,width,height = box # x,y,width,height
        draw.rectangle([x,y,x+width,y+height], fill=None, outline=(255,0,0), width=2)


def draw_bounding_boxes(image, annotations, minimum_box_area):
    """ image: PIL Image
    """
    img_vis = image.copy()
    draw = ImageDraw.Draw(img_vis)

    for anno in annotations:
        x,y,width,height = anno['bbox'] # x,y,width,height
        draw.rectangle([x,y,x+width,y+height], fill=None, outline=(0,255,0), width=2)

    boxes = [anno['bbox'] for anno in annotations]
    merge_bounding_boxes(boxes) # boxes will be changed
    boxes = filter_small_boxes(boxes, minimum_box_area)

    for box in boxes:
        x,y,width,height = box # x,y,width,height
        draw.rectangle([x,y,x+width,y+height], fill=None, outline=(255,0,0), width=2)

    return img_vis
