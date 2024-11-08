#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn #CUDA Deep Neural Network Library
import numpy as np
import torchvision

def conv_3x3(in_channels, out_channels, stride, padding):
    return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=False,
        )
    
class ConvBNReLu6(nn.Sequential):
    """ Conv + Batch Normalization + ReLu6
    - groups will separate the kernels into multiple groups, each group sees a portion of the input channels. Outputs are finally
        concatenated together
    - dilation convolution is also used
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, groups = 1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
    
class InvertedBottleneckResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor, have_relu) -> None: 
        # first conv layer is in charge of the actual stride
        super().__init__()
    
     
class InvertedBlock(nn.Module):
   
        self.conv1 = conv_3x3(in_channels=in_channels, 
                              out_channels=out_channels, stride=stride, padding=1,)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # second conv does same convolution
        self.conv2 = conv_3x3(in_channels=out_channels, 
                              out_channels=out_channels, stride=1, padding=1,)
        # Is it going to be depthwise multiplication?

class MobileNetV2(nn.Module):
    def __init__(self):
        self.layers = nn.Sequential([
            conv_3x3(in_channels=3, out_channels=32, stride=2)
        ])
    