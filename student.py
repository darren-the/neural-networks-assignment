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

Architecture:
A 34-layer residual learning framework was implemented similar to that 
suggested by (He and others, 2015) with the idea that larger numbers of 
convolutional layers could improve image-recognition accuracy. Firstly, the
base architecture used for this task was inspired by VGG nets (Simonyan and 
Zisserman, 2015) where blocks containing multiple conv layers are stacked, 
filters are small (3x3) and the number of filters are doubled every few
blocks while halving the spatial dimension. However, on its own the model
had subpar performance on the validation set, with accuracy fluctuating
betwween 70% to 80% after 100 epochs. 
To counteract this, skip connections were added every two layers which
essentially performed an identity mapping. This was the solution to the 
potential degradation problem suggested by (He and others, 2015) and allowed
the model to maintain an accuracy above 80%.

An additional enhancement used was batch normalization. This
prevents the distributions of activations at each layer from varying too
much, allowing the network to train significantly faster (10% longer without batch
normalization) at a consistent rate. It also is said to somewhat act as a 
regularizer, eliminating the need for common regularization methods such as 
dropout.

Loss function and optimiser:
Being faced with a classification problem, softmax activations were applied 
to the outputs which converts them into probabilities. Hence, a cross
entropy loss function was ideal during training as it is able to capture the
distance between these probabilities and the truth values for each category.
Additionally, cross entropy loss heavily penalises misclassified inputs
making it a preferred loss function over other common ones such as MSE.

For the optimizer, Adam was used due to its two primary properties. One is 
that it utilises the idea of momentum which speeds up convergence to minima 
and two was that it incorporates an adaptive learning rate so manual tuning 
is not as important.

Transformations:
Various transformations were tested with the hypothesis that it may help the
model generalise with other validation sets.
Random horizontal and vertical flips were the first transformations tested
as it seemed plausible that characters could be portrayed facing several
different directions. These however did not improve the validation accuracy and 
limited it to roughly 81%. With similar reasoning, random rotations were also 
tested which again reduced the model's validation performance. It seemed as
though the model could already generalise sufficiently without the need of 
any transformations.

Tuning metaparameters and other parameters: 
The batch size used for training the model only slightly affected the model 
accuracy, with smaller batches resulting in much longer training times. 
The batch size used for the final model was 256.

Our model was based off of the ResNet-34 as described in the 
paper, however to fine tune it for the application on hand, we measured the
change in eval accuracy as we changed the number of hidden units in the first
convolutional layer (before the residual blocks) and the properties of
the residual blocks. After a grid search through these parameters, it was
found that a stride of 1 applied to the first two residual blocks and a primary
convolutional layer with 128 hidden units maximised accuracy on the validation set.
This most likely allowed the model, which is already sufficiently deep, to extrapolate
significant features that distinguish each of the characters.

Use of validation set:
The validation set was used to evaluate the model???s performance during 
training which also may have helped to avoid overfitting. This was achieved by 
identifying that 100 epochs on average was suitable for training till the
convergence of validation accuracy. After training the model too far beyond
this point, the validation accuracy could decline due to overfitting. 
So as a precaution, we stopped training during the period of time that the 
accuracy appeared to converge and stop improving. 

However, it is worth noting that overfitting was unlikely to occur anyway due
to the skip connections and batch normalizations that were already implemented.

References:

Simonyan, K. and Zisserman, A. 2014, Very deep convolutional networks for large-scale 
image recognition. Available from https://arxiv.org/abs/1409.1556. (Accessed
4 Aug 2021)

He, K., Zhang, X., Ren, S. and Sun, J. 2015, Deep Residual Learning for Image
Recognition. Available from https://arxiv.org/abs/1512.03385. (Accessed 4 Aug 2021)
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
    '''
    A residual block with 2 conv layers with relu activations. Also contains skip connection
    that performs an identity mapping.
    '''

    def __init__(self, in_channels, out_channels, stride=1):
        super(Block, self).__init__()

        # 1st Convolutional Layer (3x3)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 2nd Convolutional Layer (3x3)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
        self.resize_dim = None

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if in_channels != out_channels:  # or stride != 1
            self.resize_dim = nn.Sequential(
                Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                BatchNorm2d(out_channels)
        )

    def forward(self, t):

        if self.resize_dim is not None:
            identity = self.resize_dim(t) # Remapping identity to match output channels
        else:
            identity = t    # Otherwise, identity should be recalled from input

        t = self.conv1(t)   # 1st Convolutional Layer
        t = self.bn1(t)     # Batch Normalization
        t = self.relu(t)    # ReLU activation
        t = self.conv2(t)   # 2nd Convolutional Layer
        t = self.bn2(t)     # Batch Normalization
        t += identity       # Identity mapping
        t = self.relu(t)    # ReLU activation
        return t

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 128
        self.relu = nn.ReLU()

        # 1st Convolutional Layer
        self.conv = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3)
        
        # Batch Normalization
        self.bn = BatchNorm2d(self.in_channels)

        # Max Pooling
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual Architecture
        self.layer1 = self.create_blocks(num_blocks=3, out_channels=32, stride=1)
        self.layer2 = self.create_blocks(num_blocks=4, out_channels=64, stride=1)
        self.layer3 = self.create_blocks(num_blocks=6, out_channels=128, stride=2)
        self.layer4 = self.create_blocks(num_blocks=3, out_channels=256, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 14)    # Output channels to prediction classes
        
    
    def create_blocks(self, num_blocks, out_channels, stride):
        '''
        Function to create a residual block, with skip connection
        '''
        blocks = []
        # Downsampling may only occur on the first block of each residual layer
        blocks.append(Block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels

        # No downsampling on the other blocks
        for _ in range(num_blocks - 1):
            blocks.append(Block(self.in_channels, out_channels))

        return (nn.Sequential(*blocks))
    
    def forward(self, t):
        t = self.conv(t)            # 1st Convolutional Layer
        t = self.bn(t)              # Batch Normalization
        t = self.relu(t)            # Apply ReLU activation
        t = self.pool(t)            # Apply Max Pooling
        t = self.layer1(t)          # Residual Layer 1 (Size 3)
        t = self.layer2(t)          # Residual Layer 2 (Size 4)
        t = self.layer3(t)          # Residual Layer 3 (Size 6)
        t = self.layer4(t)          # Residual Layer 4 (Size 3)
        t = self.avgpool(t)         # Average Pooling
        t = torch.flatten(t, 1)     # Flatten Output
        t = self.fc(t)              # Fully Connected Layer
        return t


# class loss(nn.Module):
#     """
#     Class for creating a custom loss function, if desired.
#     If you instead specify a standard loss function,
#     you can remove or comment out this class.
#     """
#     def __init__(self):
#         super(loss, self).__init__()

#     def forward(self, output, target):
#         pass


net = Network()
lossFunc = nn.CrossEntropyLoss()
############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"
train_val_split = 0.8
batch_size = 256
epochs = 100
optimiser = optim.Adam(net.parameters(), lr=0.001)
