import torch
import torch.nn as nn
from .layers import Unpool3DLayer

class Decoder(nn.Module):
    def __init__(self, type):
        super(Decoder, self).__init__()
        self.type = type
        n_deconvfilter = [128, 128, 128, 64, 32, 2]
        self.conv7a = nn.Conv3d(in_channels=n_deconvfilter[0], out_channels=n_deconvfilter[1],kernel_size=(3,3,3),padding=1)
        self.conv8a = nn.Conv3d(in_channels=n_deconvfilter[1], out_channels=n_deconvfilter[2], kernel_size=(3,3,3), padding=1)
        self.conv9a = nn.Conv3d(in_channels=n_deconvfilter[2], out_channels=n_deconvfilter[3], kernel_size=(3,3,3), padding=1)
        self.conv10a = nn.Conv3d(in_channels=n_deconvfilter[3], out_channels=n_deconvfilter[4], kernel_size=(3,3,3), padding=1)
        self.conv11 = nn.Conv3d(in_channels=n_deconvfilter[4], out_channels=n_deconvfilter[5], kernel_size=(3,3,3), padding=1)

        self.unpool3d = Unpool3DLayer()
        self.leakyReLU = nn.LeakyReLU()

        if self.type == "residual":
            self.conv7b = nn.Conv3d(in_channels=n_deconvfilter[1], out_channels=n_deconvfilter[1],kernel_size=(3, 3, 3), padding=1)
            self.conv8b = nn.Conv3d(in_channels=n_deconvfilter[2], out_channels=n_deconvfilter[2],kernel_size=(3, 3, 3), padding=1)
            self.conv9b = nn.Conv3d(in_channels=n_deconvfilter[3], out_channels=n_deconvfilter[3], kernel_size=(3, 3, 3), padding=1)
            self.conv9c = nn.Conv3d(in_channels=n_deconvfilter[2], out_channels=n_deconvfilter[3], kernel_size=(1, 1, 1), padding=0)
            self.conv10b = nn.Conv3d(in_channels=n_deconvfilter[4], out_channels=n_deconvfilter[4], kernel_size=(3,3,3), padding=1)
            self.conv10c = nn.Conv3d(in_channels=n_deconvfilter[4], out_channels=n_deconvfilter[4], kernel_size=(3,3,3), padding=1)

    def forward(self, x_in):
        if self.type == 'simple':
            x = self.unpool3d(x_in)
            x = self.conv7a(x)
            x = self.leakyReLU(x)
            x = self.unpool3d(x)
            x = self.conv8a(x)
            x = self.leakyReLU(x)
            x = self.unpool3d(x)
            x = self.conv9a(x)
            x = self.leakyReLU(x)
            x = self.conv10a(x)
            x = self.leakyReLU(x)
            out = self.conv11(x)
        else:
            unpool7 = self.unpool3d(x_in)
            conv7a = self.conv7a(unpool7)
            leakyReLU7a = self.leakyReLU(conv7a)
            conv7b = self.conv7b(leakyReLU7a)
            leakyReLU7 = self.leakyReLU(conv7b)
            res7 = unpool7 + leakyReLU7

            unpool8 = self.unpool3d(res7)
            conv8a = self.conv8a(unpool8)
            leakyReLU8a = self.leakyReLU(conv8a)
            conv8b = self.conv8b(leakyReLU8a)
            leakyReLU8 = self.leakyReLU(conv8b)
            res8 = unpool8 + leakyReLU8

            unpool9 = self.unpool3d(res8)
            conv9a = self.conv9a(unpool9)
            leakyReLU9a = self.leakyReLU(conv9a)
            conv9b = self.conv9b(leakyReLU9a)
            leakyReLU9 = self.leakyReLU(conv9b)

            conv9c = self.conv9c(unpool9)
            res9 = conv9c + leakyReLU9

            conv10a = self.conv10a(res9)
            leakyReLU10a = self.leakyReLU(conv10a)
            conv10b = self.conv10b(leakyReLU10a)
            leakyReLU10 = self.leakyReLU(conv10b)
            conv10c = self.conv10c(leakyReLU10)
            res10 = conv10c + leakyReLU10

            out = self.conv11(res10)
        return out
