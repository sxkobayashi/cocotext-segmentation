"""
input size should be 192, to leave spaces for scaling and displacement a patch should be 256x256
"""
import argparse
import os, glob
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from config import Configuration
from net import UNet, createDeepLabv3
from dataset import COCOTextSegmentationDataset, read_instances_with_box



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Visualize the data loader with augmented image and segmentation.")
    parser.add_argument('mask_dir', help="Root path to mask dataset")
    parser.add_argument('dataset_dir', help="Root path to training dataset")
    parser.add_argument('box_file', help="CSV file containing bounding boxes")
    parser.add_argument('output_dir', help="Root path to models and evaulation results")
    parser.add_argument('--test_run', dest='test_run', action='store_true',
                        help="If specified, only go through code for one batch")
    parser.set_defaults(test_run=False)

    # Error checking
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
    if args.test_run:
        cf.batch_size = 2
        cf.batches_per_print = 1
        cf.epoches_per_save = 1

    net = UNet(has_sigmoid=cf.has_sigmoid, multiplier=cf.multiplier).float()

    # criterion = torch.nn.BCELoss()
    # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cf.pos_weight]))
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cf.pos_weight]).to(device))
    if cf.optimizer == 'adam':
        print('Using Adam optimizer.')
        optimizer = optim.Adam(net.parameters(), lr=cf.learning_rate)
    else:
        print('Using SGD optimizer.')
        optimizer = optim.SGD(net.parameters(), lr=cf.learning_rate, momentum=cf.momentum)

    scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 0.95)
    net.to(device)
    dataset = COCOTextSegmentationDataset(image_files, cf.im_size, cf.random_scale, cf.random_displacement,
                                          cf.random_flip)
    dataloader = DataLoader(dataset, batch_size=cf.batch_size, shuffle=True, num_workers=0)

    print('Begin training...')
    os.makedirs(args.output_dir, exist_ok=True)
    config_filename = os.path.join(args.output_dir, 'config.json')
    cf.save(config_filename)
    checkpoint_path = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_path, exist_ok=True)

    for epoch in range(cf.epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            batch_inputs, batch_labels = data['image'].to(device), data['mask'].to(device)
            optimizer.zero_grad()
            
            output = net(batch_inputs)                

            loss = criterion(output, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i > 0 and i % cf.batches_per_print == 0:
                print('[%d, %5d] loss: %.5f' %(epoch + 1, i+1, running_loss / cf.batches_per_print))
                running_loss = 0

            if args.test_run and i > 1:
                break

        scheduler.step()

        if epoch % cf.epoches_per_save == 0:
            checkpoint_filename = os.path.join(checkpoint_path, 'epoch_{0:03d}.pickle'.format(epoch))
            print('Saving network to {}'.format(checkpoint_filename))
            torch.save(net.state_dict(), checkpoint_filename)
        if args.test_run:
            break

    print('Finished Training.')
