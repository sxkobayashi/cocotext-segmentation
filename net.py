import torch.nn as nn
import torch
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models

class ConvBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.relu_1 = nn.LeakyReLU(0.1)

        self.conv_2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(out_channels)
        self.relu_2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu_1(x)

        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.relu_2(x)

        return x
        

class UpConvBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(UpConvBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        x = self.upsample(x)
        x = self.conv(x)
        x = self.bn(x)

        return x


class UNet(nn.Module):
    def __init__(self, has_sigmoid=False, multiplier=1):
        super(UNet, self).__init__()

        self.has_sigmoid = has_sigmoid

        # Downsample
        self.conv_block_d0 = ConvBlock(3, int(64 * multiplier))

        self.pool_d1 = nn.MaxPool2d(2)
        self.conv_block_d1 = ConvBlock(int(64 * multiplier), int(128 * multiplier))

        self.pool_d2 = nn.MaxPool2d(2)
        self.conv_block_d2 = ConvBlock(int(128 * multiplier), int(256 * multiplier))

        self.pool_d3 = nn.MaxPool2d(2)
        self.conv_block_d3 = ConvBlock(int(256 * multiplier), int(512 * multiplier))

        self.pool_d4 = nn.MaxPool2d(2)
        self.conv_block_d4 = ConvBlock(int(512 * multiplier), int(512 * multiplier))

        # Upsample
        self.up_conv_u4 = UpConvBlock(int(512 * multiplier), int(512 * multiplier))
        self.conv_block_u4 = ConvBlock(int(1024 * multiplier), int(512 * multiplier))

        self.up_conv_u3 = UpConvBlock(int(512 * multiplier), int(256 * multiplier))
        self.conv_block_u3 = ConvBlock(int(512 * multiplier), int(256 * multiplier))

        self.up_conv_u2 = UpConvBlock(int(256 * multiplier), int(128 * multiplier))
        self.conv_block_u2 = ConvBlock(int(256 * multiplier), int(128 * multiplier))

        self.up_conv_u1 = UpConvBlock(int(128 * multiplier), int(64 * multiplier))
        self.conv_block_u1 = ConvBlock(int(128 * multiplier), int(64 * multiplier))

        self.conv_r2 = nn.Conv2d(int(64 * multiplier), 1, 3, padding=1)

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        
        x = self.conv_block_d0(x)  # x: (N, 64, H, W)
        x1 = self.conv_block_d1(self.pool_d1(x))  # x1: (N, 128, H/2, W/2)
        x2 = self.conv_block_d2(self.pool_d2(x1))  # x2: (N, 256, H/4, W/4)
        x3 = self.conv_block_d3(self.pool_d3(x2))  # x3: (N, 512, H/8, W/8)
        x4 = self.conv_block_d4(self.pool_d4(x3))  # x4: (N, 512, H/16, W/16)

        y4 = self.up_conv_u4(x4)
        y4 = torch.cat((y4, x3), dim=1)  # y4: (N, 1024, H/8, W/8)
        y3 = self.conv_block_u4(y4)  # y3: (N, 512, H/8, W/8)

        y3 = self.up_conv_u3(y3)  # y3: (N, 256, H/4, W/4)
        y3 = torch.cat((y3, x2), dim=1)  # y3: (N, 512, H/4, W/4)
        y2 = self.conv_block_u3(y3) # y2: (N, 256, H/4, W/4)

        y2 = self.up_conv_u2(y2) # y2: (N, 128, H/2, W/2)
        y2 = torch.cat((y2, x1), dim=1) # y2: (N, 256, H/2, W/2)
        y1 = self.conv_block_u2(y2) # y2: (N, 128, H/2, W/2)

        y = self.up_conv_u1(y1) # y: (N, 64, H, W)
        y = torch.cat((y, x), dim=1) # y: (N, 128, H, W)
        y = self.conv_block_u1(y) # y: (N, 64, H, W)

        y_out = self.conv_r2(y)

        if self.has_sigmoid:
            y_out = self.sigmoid(y_out)

        return y_out


def createDeepLabv3(outputchannels=1, mode='train'):
    """DeepLabv3 class with custom head
    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.
    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """
    model = models.segmentation.deeplabv3_resnet50(pretrained=True,
                                                   progress=True)
    model.classifier = DeepLabHead(2048, outputchannels)

    # Set the model in training mode
    if mode == 'train':
        model.train()
    else:
        model.eval()

    return model.float()


if __name__ == '__main__':

    from torch.utils.tensorboard import SummaryWriter
    
    net = UNet(has_sigmoid=False, multiplier=0.5)
    input_to_model=torch.rand((1,3,192,192))
    output = net(input_to_model)

    writer = SummaryWriter('runs/net_experiment_1')
    writer.add_graph(net, input_to_model=input_to_model)
    writer.close()

    # tensorboard --logdir=runs
