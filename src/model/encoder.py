import torch.nn as nn
import numpy as np

class Encoder(nn.Module):
    def __init__(self, type):
        super(Encoder, self).__init__()
        self.type = type
        n_convfilter = [96, 128, 256, 256, 256, 256]
        self.conv1a = nn.Conv2d(in_channels=3, out_channels=n_convfilter[0], kernel_size=7, padding=3)
        self.conv2a = nn.Conv2d(in_channels=n_convfilter[0], out_channels=n_convfilter[1], kernel_size=3, padding=1)
        self.conv3a = nn.Conv2d(in_channels=n_convfilter[1], out_channels=n_convfilter[2], kernel_size=3, padding=1)
        self.conv4a = nn.Conv2d(in_channels=n_convfilter[2], out_channels=n_convfilter[3], kernel_size=3, padding=1)
        self.conv5a = nn.Conv2d(in_channels=n_convfilter[3], out_channels=n_convfilter[4], kernel_size=3, padding=1)
        self.conv6a = nn.Conv2d(in_channels=n_convfilter[4], out_channels=n_convfilter[5], kernel_size=3, padding=1)
        if self.type == 'residual':
            self.conv1b = nn.Conv2d(in_channels=n_convfilter[0], out_channels=n_convfilter[0], kernel_size=3, padding=1)
            self.conv2b = nn.Conv2d(in_channels=n_convfilter[1], out_channels=n_convfilter[1], kernel_size=3, padding=1)
            self.conv2c = nn.Conv2d(in_channels=n_convfilter[0], out_channels=n_convfilter[1], kernel_size=1, padding=0)
            self.conv3b = nn.Conv2d(in_channels=n_convfilter[2], out_channels=n_convfilter[2], kernel_size=3, padding=1)
            self.conv3c = nn.Conv2d(in_channels=n_convfilter[1], out_channels=n_convfilter[2], kernel_size=1, padding=0)
            self.conv4b = nn.Conv2d(in_channels=n_convfilter[3], out_channels=n_convfilter[3], kernel_size=3, padding=1)
            self.conv5b = nn.Conv2d(in_channels=n_convfilter[4], out_channels=n_convfilter[4], kernel_size=3, padding=1)
            self.conv5c = nn.Conv2d(in_channels=n_convfilter[3], out_channels=n_convfilter[4], kernel_size=1, padding=0)
            self.conv6b = nn.Conv2d(in_channels=n_convfilter[5], out_channels=n_convfilter[5], kernel_size=3, padding=1)
        self.fc = nn.Linear(in_features=n_convfilter[5]*(self.get_spatial_dim(input_spatial_dim=127,num_pooling=6)**2), out_features=1024)
        self.pool = nn.MaxPool2d(kernel_size=2, padding=1)
        self.leakyReLU = nn.LeakyReLU()


    def forward(self, x_in):
        if self.type == 'simple':
            x = self.conv1a(x_in)
            x = self.pool(x)
            x = self.leakyReLU(x)
            x = self.conv2a(x)
            x = self.pool(x)
            x = self.leakyReLU(x)
            x = self.conv3a(x)
            x = self.pool(x)
            x = self.leakyReLU(x)
            x = self.conv4a(x)
            x = self.pool(x)
            x = self.leakyReLU(x)
            x = self.conv5a(x)
            x = self.pool(x)
            x = self.leakyReLU(x)
            x = self.conv6a(x)
            x = self.pool(x)
            x = self.leakyReLU(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            out = self.leakyReLU(x)
            return out
        else:
            conv1a = self.conv1a(x_in)
            leakyReLU1a = self.leakyReLU(conv1a)
            conv1b = self.conv1b(leakyReLU1a)
            leakyReLU1 = self.leakyReLU(conv1b)
            pool1 = self.pool(leakyReLU1)

            conv2a = self.conv2a(pool1)
            leakyReLU2a = self.leakyReLU(conv2a)
            conv2b = self.conv2b(leakyReLU2a)
            leakyReLU2 = self.leakyReLU(conv2b)
            conv2c = self.conv2c(pool1)
            res2 = conv2c + leakyReLU2
            pool2 = self.pool(res2)

            conv3a = self.conv3a(pool2)
            leakyReLU3a = self.leakyReLU(conv3a)
            conv3b = self.conv3b(leakyReLU3a)
            leakyReLU3 = self.leakyReLU(conv3b)
            conv3c = self.conv3c(pool2)
            res3 = conv3c + leakyReLU3
            pool3 = self.pool(res3)

            conv4a = self.conv4a(pool3)
            leakyReLU4a = self.leakyReLU(conv4a)
            conv4b = self.conv4b(leakyReLU4a)
            leakyReLU4 = self.leakyReLU(conv4b)
            pool4 = self.pool(leakyReLU4)

            conv5a = self.conv5a(pool4)
            leakyReLU5a = self.leakyReLU(conv5a)
            conv5b = self.conv5b(leakyReLU5a)
            leakyReLU5 = self.leakyReLU(conv5b)
            conv5c = self.conv5c(pool4)
            res5 = conv5c + leakyReLU5
            pool5 = self.pool(res5)

            conv6a = self.conv6a(pool5)
            leakyReLU6a = self.leakyReLU(conv6a)
            conv6b = self.conv6b(leakyReLU6a)
            leakyReLU6 = self.leakyReLU(conv6b)
            res6 = pool5 + leakyReLU6
            pool6 = self.pool(res6)

            pool6 = pool6.view(pool6.size(0), -1)

            fc7 = self.fc(pool6)
            out = self.leakyReLU(fc7)
            return out

    def get_spatial_dim(self, input_spatial_dim, num_pooling):
        kernel_size = 2
        padding = 1
        spatial_dim = input_spatial_dim
        for i in range(num_pooling):
            spatial_dim = np.floor((spatial_dim - kernel_size + 2*padding)/2 + 1)
        return int(spatial_dim)