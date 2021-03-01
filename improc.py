"""
Image processing utilities
"""
import numpy as np
from skimage.transform import resize

def crop_images(images, box):
    x,y,w,h = box
    return [img[x:x+w, y:y+h, :] for img in images]

def resize_images(images, output_shape):
    return [resize(img, output_shape) for img in images]

def rotate_90(images, k):
    return [np.rot90(m, k, axes=(0, 1)) for m in images]

# def random_crop(img):
