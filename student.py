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

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1=nn.Conv2d(in_channels=3, out_channels=6,kernel_size=5, padding=1) 
        self.conv2=nn.Conv2d(in_channels=6,out_channels=12,kernel_size=5, padding=1) 
        self.fc_layer_1=nn.Linear(12*60*60,250)
        self.fc_layer_2=nn.Linear(250,14)

    def forward(self, x):        

        # (1) Convolutional Layer
        x = F.relu(self.conv1(x))
        # x = F.max_pool2d(x, kernel_size=2, stride=2)

        # (2) Convolutional Layer
        x = F.relu(self.conv2(x))
        # x = F.max_pool2d(x, kernel_size=2, stride=2)

        # (3) Linear Layer
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc_layer_1(x))

        # (4) Linear Layer
        x = F.relu(self.fc_layer_2(x))

        # (5) Output Layer
        x = F.log_softmax(x, dim=1)  

        return x  



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
epochs = 60
optimiser = optim.SGD(net.parameters(), lr=0.01, momentum=0.4)
