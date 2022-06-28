import torch
import torch.nn as nn

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