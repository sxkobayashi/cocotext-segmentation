import os,random,glob
import numpy as np
import torch
import torchvision.transforms.functional as ttf

from PIL import Image
from torch.utils.data import Dataset, DataLoader

import box_utils


def read_instances_without_box(mask_dir, dataset_dir):
    mask_files = glob.glob(os.path.join(mask_dir, '*.jpg'))
    print(f'{len(mask_files)} patches are found for validation.')

    image_files = []
    for mask_file in mask_files:
        bname = os.path.basename(mask_file)
        image_file = os.path.join(dataset_dir, bname)
        if not os.path.exists(image_file):
            print(f"{image_file} does not exist. Skipping.")
            continue
        image_files.append((image_file, mask_file))
    return image_files


def read_instances_with_box(mask_dir, dataset_dir, box_file):

    mask_files = glob.glob(os.path.join(mask_dir, '*.jpg'))
    print(f'{len(mask_files)} patches are found for training.')
    box_dict = box_utils.read_boxes_from_csv(box_file)

    image_files = []
    for mask_file in mask_files:
        bname = os.path.basename(mask_file)
        box = box_dict.get(bname)
        if box is None:
            print(f"Box for {bname} does not exist. Skipping.")
            continue
        image_file = os.path.join(dataset_dir, bname)
        if not os.path.exists(image_file):
            print(f"{image_file} does not exist. Skipping.")
            continue
        image_files.append((image_file, mask_file, box))

    return image_files


def augment_image(img, mask, box, net_input_shape, random_scale, random_displacement, random_flip):
    """
    img, mask: numpy array of (height, width, 3)
    box: a [x,y,w,h] box at the center
    net_input_shape: input height and width of network (height, width)
    """
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError('Expecting image shape to be [H,W,3].')
    if not (len(mask.shape) == 3 and mask.shape[2] == 1):
        raise ValueError('Expecting mask shape to be [H,W,1].')

    img_height = img.shape[0]
    img_width = img.shape[1]
    net_height = net_input_shape[0]
    net_width = net_input_shape[1]

    # random scale
    if random_scale is not None:
        scale_factor = np.random.uniform(random_scale[0], random_scale[1])
    else:
        scale_factor = 1.0
    scaled_height = int(net_height * scale_factor)
    scaled_width = int(net_width * scale_factor)

    # randomly displace a little
    if random_displacement is not None:
        displacement_x = int(np.random.uniform(-random_displacement, random_displacement) * scaled_width)
        displacement_y = int(np.random.uniform(-random_displacement, random_displacement) * scaled_height)
    else:
        displacement_x = 0
        displacement_y = 0

    x, y, w, h = box
    x_center, y_center = x + w/2, y + h/2
    crop_box = box_utils.int_box([x_center - net_width / 2, y_center - net_height / 2, 
                                  net_width,                net_height])
    crop_box = box_utils.int_box(box_utils.rescale_box(crop_box, scale_factor))
    crop_box = box_utils.shift_box(crop_box, displacement_x, displacement_y, img_width, img_height)

    # randomly rotate by 90k degrees, k = 0,1,2,3
    random_hflip = random.randrange(2)
    random_vflip = random.randrange(2)

    # Begin transforms
    # numpy to Torch tensor
    img_aug = torch.from_numpy(img)
    mask_aug = torch.from_numpy(mask)

    # (H,W,C) -> (C,H,W)
    img_aug = img_aug.permute(2,0,1)
    mask_aug = mask_aug.permute(2,0,1)

    # Crop out the randomly scaled / displaced box
    x,y,w,h = crop_box
    img_aug = ttf.crop(img_aug, top=y, left=x, height=h, width=w)
    mask_aug = ttf.crop(mask_aug, top=y, left=x, height=h, width=w)

    # Resize to network size
    img_aug = ttf.resize(img_aug, [net_height, net_width])
    mask_aug = ttf.resize(mask_aug, [net_height, net_width])

    # Flip if needed
    if random_flip:
        if random_hflip > 0:
            img_aug = ttf.hflip(img_aug)
            mask_aug = ttf.hflip(mask_aug)
        if random_vflip > 0:
            img_aug = ttf.vflip(img_aug)
            mask_aug = ttf.vflip(mask_aug)

    return img_aug, mask_aug


class COCOTextSegmentationDataset(Dataset):

    def __init__(self, all_instances, im_size,
                 random_scale=(0.8,1.2), random_displacement=0.2, random_flip=True):
        """Initialize dataset.
        all_instances: a list of 2-tuple (image_path, mask_path)
        im_size: (height, width) of the generated mini-batch
        random_scale: random scale minimum and maximum
        random_displacement: ratio of maximum random shift relative to x and y direction.
        """
        self.all_instances = all_instances
        self.im_size = im_size
        self.random_scale = random_scale
        self.random_displacement = random_displacement
        self.random_flip = random_flip

    def __len__(self):
        return len(self.all_instances)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image_path, mask_path, box = self.all_instances[idx]
        
        img = np.asarray(Image.open(image_path))
        mask = np.asarray(Image.open(mask_path))
        img = img.astype('float32') / 255.0
        mask = mask.astype('float32') / 255.0
        mask = mask[:,:,0:1]

        img_aug, mask_aug = augment_image(img, mask, box, self.im_size, self.random_scale, self.random_displacement,
                                          self.random_flip)

        sample = {'image': img_aug, 'mask': mask_aug}

        return sample

