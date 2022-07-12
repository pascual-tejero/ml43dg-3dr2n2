import torch
import torch.nn as nn

class SoftmaxWithLoss3D(nn.Module):
    def __init__(self):
        super(SoftmaxWithLoss3D, self).__init__()

    def forward(self, inputs, y_true):

        """
        Before actually compute the loss, we need to address the possible numberical instability.
        If some elements of inputs are very large, and we compute their exponential value, then we
        might encounter some infinity. So we need to subtract them by the largest value along the
        "channels" dimension to avoid very large exponential.
        """
        # the size of inputs and y is (batch_size, channels, depth, height, width) (batch_size, 2, 32, 32, 32)
        # torch.max return a tuple of (max_value, index_of_max_value)
        max_channel = torch.max(inputs, dim=1, keepdim=True)[0]
        adj_inputs = inputs - max_channel
        exp_x = torch.exp(adj_inputs)
        sum_exp_x = torch.sum(exp_x, dim=1, keepdim=True)

        # if the ground truth is provided the loss will be computed
        loss = torch.mean(
            torch.sum(-y_true * adj_inputs, dim=1, keepdim=True) + \
            torch.log(sum_exp_x))
        return loss