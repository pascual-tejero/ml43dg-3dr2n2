import torch
import torch.nn as nn


class SoftmaxWithLoss3D(nn.Module):
    def __init__(self):
        super(SoftmaxWithLoss3D, self).__init__()

    def forward(self, y_pred, y_true):

        """
        Before actually compute the loss, we need to address the possible numberical instability.
        If some elements of inputs are very large, and we compute their exponential value, then we
        might encounter some infinity. So we need to subtract them by the largest value along the
        "channels" dimension to avoid very large exponential.

        # the size of inputs and y is (batch_size, channels, depth, height, width) (batch_size, 2, 32, 32, 32)
        # torch.max return a tuple of (max_value, index_of_max_value)

        y_true must be a tensor that has the same dimensions as the input. For each
        channel, only one element is one indicating the ground truth prediction
        label.
        """
        max_channel = torch.max(y_pred, dim=1, keepdim=True)[0]
        adj_inputs = y_pred - max_channel
        exp_x = torch.exp(adj_inputs)
        sum_exp_x = torch.sum(exp_x, dim=1, keepdim=True)

        # if the ground truth is provided the loss will be computed
        loss = torch.mean(
            torch.sum(-y_true * adj_inputs, dim=1, keepdim=True) + torch.log(sum_exp_x)
        )
        return loss
