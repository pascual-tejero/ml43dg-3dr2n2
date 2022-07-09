import torch
import torch.nn as nn
import numpy as np

class GRUGate3D(nn.Module):
    def __init__(self, fan_in, filter_params, output_shape):
        super(GRUGate3D, self).__init__()
        self.fan_in = fan_in # 1024
        self.filter_params = filter_params # (128, 128, 3, 3, 3)
        self.output_shape = output_shape # (batch_size, 128, 4, 4, 4)
        self.padding = int((filter_params[2] - 1) / 2)
        self.linear = nn.Linear(in_features=fan_in, out_features=int(np.prod(output_shape[1:])), bias=False) # Different from the final fully connected layer of the encoder. Please visit the 3DR2N2 article for the figure.
        self.conv3d = nn.Conv3d(in_channels=filter_params[0],
                                out_channels=filter_params[1],
                                kernel_size=filter_params[2],
                                padding=self.padding,
                                bias=False)
        self.bias = nn.Parameter(torch.FloatTensor(1, output_shape[1], 1, 1, 1).fill_(0.1))

    def forward(self, encoder_out, h):
        # TODO: This function might require some debugging because the output dimensions might not match!
        output_shape_tmp = list(self.output_shape)
        output_shape_tmp[0] = -1 # To deal with different batch sizes
        out = self.linear(encoder_out).view(*output_shape_tmp) + self.conv3d(h) + self.bias
        return out

class Unpool3DLayer(nn.Module):
    def __init__(self, unpool_size=2, padding=0):
        print("initializing \"Unpool3DLayer\"")
        super(Unpool3DLayer, self).__init__()
        self.unpool_size = unpool_size
        self.padding = padding

    def forward(self, x):
        n = self.unpool_size
        p = self.padding
        # x.size() is (batch_size, channels, depth, height, width)
        output_size = (x.size(0), x.size(1), n * x.size(2), n * x.size(3), n * x.size(4))

        out_tensor = torch.Tensor(*output_size).zero_()

        if torch.cuda.is_available():
            out_tensor = out_tensor.cuda()

        out = out_tensor

        out[:,:, p: p + output_size[2] + 1: n, p: p + output_size[3] + 1: n, p: p + output_size[4] + 1: n] = x
        return out