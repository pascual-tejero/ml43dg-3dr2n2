import numpy as np

from models.base_gru_net import BaseGRUNet
from lib.layers import FCConv3DLayer_torch, Unpool3DLayer, SoftmaxWithLoss3D

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Linear, Conv2d, MaxPool2d, \
                     LeakyReLU, Conv3d, Tanh, Sigmoid



class encoder(nn.Module):
    def __init__(self):
        # conv7_kernel_size = 7 --> padding = (7-1)/2 = 3
        # conv3_kernel_size = 3 --> padding = (3-1)/2 = 1

        self.conv_7 = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 7, stride = 1, padding = 3)
        self.conv_3 = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 3, stride = 1, padding = 1)
        self.conv_1 = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 1, stride = 1, padding = 0)

        self.fc = nn.Linear(in_features = 1, out_features = 1024)

        self.pool = MaxPool2d(kernel_size= 2, padding= 1)

        self.leaky_relu = LeakyReLU(negative_slope= 0.01)

    def forward(self, x, type):
        if type == 'simple':
            x_conv = nn.Sequential(self.conv_7, self.pool, self.leaky_relu,
                                   self.conv_3, self.pool, self.leaky_relu,
                                   self.conv_3, self.pool, self.leaky_relu,
                                   self.conv_3, self.pool, self.leaky_relu,
                                   self.conv_3, self.pool, self.leaky_relu,
                                   self.conv_3, self.pool, self.leaky_relu,
                                   self.fc, self.leaky_relu)
            
            return x_conv

        if type == 'deep':
            x_conv_111 = self.conv_7(self.conv_3(x))
            x_conv_112 = self.conv_1(x)
            x_conv_12 = self.leaky_relu(self.pool(x_conv_111 + x_conv_112))

            x_conv_211 = self.conv_3(self.conv_3(x_conv_12))
            x_conv_212 = self.conv_1(x_conv_12)
            x_conv_22 = self.leaky_relu(self.pool(x_conv_211 + x_conv_212))

            x_conv_311 = self.conv_3(self.conv_3(x_conv_22))
            x_conv_312 = self.conv_1(x_conv_22)
            x_conv_32 = self.leaky_relu(self.pool(x_conv_311 + x_conv_312))

            x_conv_411 = self.conv_3(self.conv_3(x_conv_32))
            x_conv_42 = self.leaky_relu(self.pool(x_conv_411))      

            x_conv_511 = self.conv_3(self.conv_3(x_conv_42))
            x_conv_512 = self.conv_1(x_conv_42)
            x_conv_52 = self.leaky_relu(self.pool(x_conv_511 + x_conv_512))      

            x_conv_611 = self.conv_3(self.conv_3(x_conv_52))
            x_conv_612 = self.conv_1(x_conv_52)
            x_conv_62 = self.leaky_relu(self.pool(x_conv_611 + x_conv_612))      

            x_conv = self.leaky_relu(self.fc(x_conv_62))
            
            return x_conv
