"""
input size should be 192, to leave spaces for scaling and displacement a patch should be 256x256
"""
import argparse
import os, glob, random
import torch
import numpy as np
from PIL import Image
from PIL import ImageFilter

from config import Configuration
from net import UNet
from visutils import draw_overlay_image


def process_single_image(img_path, out_img_path, pos_threshold=0.5, blur=True, blur_kernel_len=8):
    """Process a single image.
    img_path: path of input image.
    out_img_path: path of output image.
    pos_threshold: positive threshold, default to 0.5
    blur: If True, generate blurred image; otherwise, draw mask on the image.
    blur_kernel_len: gaussian blur kernel radius. Defaults to 8
    """

    img = Image.open(img_path)
    orig_width, orig_height = img.size
    
    width = round(orig_width / 32) * 32
    height = round(orig_height / 32) * 32
    img = img.resize((width, height))

    if blur:
        blurred_img = img.filter(ImageFilter.GaussianBlur(blur_kernel_len))
        blurred_img = np.asarray(blurred_img, dtype='float32') / 255.0

    img = np.asarray(img, dtype='float32') / 255.0
    if len(img.shape) == 2:
        img = np.repeat(img[:,:,np.newaxis], 3, axis=-1)
    input_img = np.transpose(img, (2,0,1))
    batched_input = np.expand_dims(input_img, axis=0)

    batched_input_tensor = torch.from_numpy(batched_input)
    batched_input_tensor = batched_input_tensor.to(device)

    outputs = net(batched_input_tensor)
    predictions = outputs.cpu().detach().numpy()

    mask = 1.0 * (predictions > pos_threshold)
    msk = np.transpose(mask[0], (1,2,0))

    if blur:
        masked_blur = msk * blurred_img + (1 - msk) * img
        blurred_img = np.clip(255 * masked_blur, 0, 255).astype('uint8')
        blurred_img = Image.fromarray(blurred_img)
        blurred_img = blurred_img.resize((orig_width, orig_height))
        blurred_img.save(out_img_path)
    else:
        out_img = draw_overlay_image(img, msk, threshold=pos_threshold)
        out_image = Image.fromarray(out_img)
        out_image = out_image.resize((orig_width, orig_height))
        out_image.save(out_img_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Running inference on some images.")
    parser.add_argument('img_path', help="Path to an image file, or a folder containing a bunch of images or a text file of a list of images.")
    parser.add_argument('checkpoint', help="Path to model checkpoint")
    parser.add_argument('--arch_params', nargs='?',
                        help="Architecture parameters stored in json file. If none, use all defaults.")
    parser.add_argument('--output', nargs='?', default='output.jpg',
                        help="Path to output image or a folder containing output images.")
    parser.add_argument('--pos_threshold', nargs='?', type=float, default=0.5,
                        help="Positive threshold. Default to 0.5.")
    parser.add_argument('--blur', dest='blur', action='store_true',
                        help="If true, blur the detected mask instead of coloring it.")
    parser.set_defaults(blur=False)
    parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                        help="If true, overwrite existing files.")
    parser.set_defaults(overwrite=False)
    args = parser.parse_args()

    if torch.cuda.device_count() == 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")

    cf = Configuration()
    if args.arch_params is not None:
        cf.load(args.arch_params)

    if os.path.isdir(args.img_path) or args.img_path.endswith('.txt'):
        multiple_files = True
        if os.path.isdir(args.img_path):
            img_list = glob.glob(os.path.join(args.img_path, '*.jpg'))
        else:
            with open(args.img_path, 'r') as fp:
                img_list = fp.readlines()
            img_list = [x.strip() for x in img_list]
        print('Found {} images.'.format(len(img_list)))

        if args.output.endswith('.jpg'):
            raise ValueError('Input is an image directory, please provide output directory name.')
        if not os.path.exists(args.output):
            os.makedirs(args.output, exist_ok=True)
    else:
        multiple_files = False

    print('Loading network...')
    net = UNet(has_sigmoid=True, multiplier=cf.multiplier).float()
    net.load_state_dict(torch.load(args.checkpoint, map_location=device))
    net.to(device)

    if multiple_files:
        for img_path in img_list:
            out_img_path = os.path.join(args.output, os.path.basename(img_path))
            if os.path.exists(out_img_path) and args.overwrite is False:
                print(f'{out_img_path} exists. Skipping. ')
                continue
            print(img_path)
            try:
                process_single_image(img_path, out_img_path, pos_threshold=args.pos_threshold, blur=args.blur)
            except Exception as e:
                print(e)
    else:
        print(img_path)
        process_single_image(img_path, args.output, pos_threshold=args.pos_threshold, blur=args.blur)

    print('Done.')

