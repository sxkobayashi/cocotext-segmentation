"""
input size should be 192, to leave spaces for scaling and displacement a patch should be 256x256
"""
import argparse
import os, glob, random
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image

from config import Configuration
from net import UNet
from dataset import COCOTextSegmentationDataset, read_instances_with_box
from visutils import draw_overlay_image


def sigmoid_numpy(x):
    return 1 / (1 + np.exp(-x))


def segmentation_metrics(y_true, y_pred, smooth=1):
    """ Return Intersection-Over-Union (IOU), precision and recall
    """
    assert len(y_true.shape) == 4
    assert len(y_pred.shape) == 4

    intersection = np.sum(np.abs(y_true * y_pred), axis=(1,2,3))  # also true positive
    gt_pos = np.sum(y_true, axis=(1,2,3)) # true positive + false negative
    pred_pos = np.sum(y_pred, axis=(1,2,3)) # true positive + false positive
#     union = np.sum(y_true, axis=(1,2,3)) + np.sum(y_pred, axis=(1,2,3)) - intersection
    union = gt_pos + pred_pos - intersection
    iou = np.mean((intersection + smooth) / (union + smooth), axis=0)
    precision = (intersection + smooth) / (pred_pos + smooth)
    recall = (intersection + smooth) / (gt_pos + smooth)
    return iou, precision, recall
    

def evaluate(dataloader, net, device, has_sigmoid=False, pos_threshold=0.5, 
             visualize=False, output_dir=None, img_files=None, verbose=False):
    """ Returns 3 metrics: iou, precision and recall.
    If the network has no sigmoid function, then set has_sigmoid = False, and this method will convert output 
        logits to probability.
    If visualize=True, dataloader must have batch_size = 1
    """
    metrics = [] # iou, precision ,recall
    for i, data in enumerate(dataloader):
        batch_inputs, batch_labels = data['image'].to(device), data['mask'].to(device)
        outputs = net(batch_inputs)
        predictions = outputs.cpu().detach().numpy()

        if not has_sigmoid:
            predictions = sigmoid_numpy(predictions)

        mask = 1.0 * (predictions > pos_threshold)

        batch_ious, batch_precisions, batch_recalls = segmentation_metrics(
                batch_labels.cpu().detach().numpy(), mask, smooth=1)
        mean_iou = np.mean(batch_ious)
        mean_precision = np.mean(batch_precisions)
        mean_recall = np.mean(batch_recalls)
        metrics.append([mean_iou, mean_precision, mean_recall])
        if verbose:
            print('[%d] iou = %f, precision = %f, recall = %f' %(i+1, mean_iou, mean_precision, mean_recall))

        if visualize:
            if output_dir is None or img_files is None:
                raiseValueError('output_dir and img_fiels must be provided when visualization is True.')
            if not os.path.exists(output_dir):
                raise ValueError(f'{output_dir} does not exist.')
            # if mean_iou > 0.4:
            #     continue

            img = batch_inputs.cpu().detach().numpy()[0]
            img = np.transpose(img, (1,2,0)) # [N=1,C,H,W] -> [H,W,C]
            msk = np.transpose(mask[0], (1,2,0)) # [N=1,C,H,W] -> [H,W,C]
            overlay_img = draw_overlay_image(img, msk, threshold=pos_threshold)
            overlay_image = Image.fromarray(overlay_img)
            img_file = image_files[i][0]
            out_filename = os.path.join(output_dir, os.path.basename(img_file))
            if verbose:
                print(f'Writing to {out_filename}...')
            overlay_image.save(out_filename)

    metrics = np.array(metrics)
    return metrics


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Evaluate the model by computing IOU and visualizing result.")
    parser.add_argument('mask_dir', help="Root path to mask dataset")
    parser.add_argument('dataset_dir', help="Root path to validation dataset")
    parser.add_argument('box_file', help="CSV file containing bounding boxes")
    parser.add_argument('checkpoint', help="Path to model checkpoint")
    parser.add_argument('output_dir', help="Root path to evaulation results")
    parser.add_argument('--visualize', dest='visualize', action='store_true', 
                        help="If True, visualize the result in output_dir.")
    parser.set_defaults(visualize=False)

    args = parser.parse_args()
    if not os.path.exists(args.mask_dir):
        raise ValueError(f'{args.mask_dir} does not exist.')
    if not os.path.exists(args.dataset_dir):
        raise ValueError(f'{args.dataset_dir} does not exist.')

    if torch.cuda.device_count() == 0:
        device = torch.device("cpu")
        print('[Training] Running on CPU.')
    else:
        device = torch.device("cuda:0")
        print('[Training] Running on GPU.')

    image_files = read_instances_with_box(args.mask_dir, args.dataset_dir, args.box_file)

    print('Instantiating neural network...')
    cf = Configuration()
    net = UNet(has_sigmoid=True, multiplier=cf.multiplier).float()

    net.load_state_dict(torch.load(args.checkpoint, map_location=device))
    net.to(device)

    dataset = COCOTextSegmentationDataset(image_files, cf.im_size, None, None, False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    os.makedirs(args.output_dir, exist_ok=True)

    evaluate(dataloader, net, device, has_sigmoid=True, pos_threshold=cf.pos_threshold, 
             visualize=args.visualize, output_dir=args.output_dir, img_files=image_files, verbose=True)

    try:
        np.save('metrics.npy', metrics)
    except:
        pass
    mean_metrics = np.mean(metrics, axis=0)
    print('IOU = %f, precision = %f, recall = %f' %(mean_metrics[0], mean_metrics[1], mean_metrics[2]))
    print('Finished Evaluating.')

