#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables train_val_split,
batch_size as well as the transform function.
You are encouraged to modify these to improve the performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

"""
   Answer to Question:

Briefly describe how your program works, and explain any design and training
decisions you made along the way.

"""

############################################################################
######     Specify transform(s) to be applied to the input images     ######
############################################################################
def transform(mode):
    """
    Called when loading the data. Visit this URL for more information:
    https://pytorch.org/vision/stable/transforms.html
    You may specify different transforms for training and testing
    """
    if mode == 'train':
        return transforms.ToTensor()
    elif mode == 'test':
        return transforms.ToTensor()

############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.resize_dim = None
        if in_channels != out_channels:  # or stride != 1
            self.resize_dim = nn.Sequential(
                Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                BatchNorm2d(out_channels)
        )

    def forward(self, t):
        if self.resize_dim is not None:
            identity = self.resize_dim(t)
        else:
            identity = t
        t = self.conv1(t)
        t = self.bn1(t)
        t = self.relu(t)
        t = self.conv2(t)
        t = self.bn2(t)
        t += identity
        t = self.relu(t)
        return t

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 128
        self.conv = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3)
        self.bn = BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(3, 2, 1)
        
        # Residual Architecture
        self.layer1 = self.create_layers(3, 32, 1)
        self.layer2 = self.create_layers(4, 64, 1)
        self.layer3 = self.create_layers(6, 128, 2)
        self.layer4 = self.create_layers(3, 256, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 14)    
        
    def create_layers(self, num_blocks, out_channels, stride):
        layers = []
        layers.append(Block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(num_blocks - 1):
            layers.append(Block(self.in_channels, out_channels))
        return (nn.Sequential(*layers))
    
    def forward(self, t):
        t = self.conv(t)
        t = self.bn(t)
        t = self.relu(t)
        t = self.pool(t)
        t = self.layer1(t)
        t = self.layer2(t)
        t = self.layer3(t)
        t = self.layer4(t)
        t = self.avgpool(t)
        t = torch.flatten(t, 1)
        t = self.fc(t)
        return t


class loss(nn.Module):
    """
    Class for creating a custom loss function, if desired.
    If you instead specify a standard loss function,
    you can remove or comment out this class.
    """
    def __init__(self):
        super(loss, self).__init__()

    def forward(self, output, target):
        pass


net = Network()
lossFunc = nn.CrossEntropyLoss()
############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"
train_val_split = 0.8
batch_size = 256
epochs = 150
optimiser = optim.Adam(net.parameters(), lr=0.001)
