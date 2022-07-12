import torch
import torch.nn as nn
from layers import GRUGate3D

class ConvGRU3D(nn.Module):
    def __init__(self, fan_in, hidden_size, grid_size, kernel_size):
        super(ConvGRU3D, self).__init__()
        self.fan_in = fan_in
        self.hidden_size = hidden_size
        self.grid_size = grid_size
        self.kernel_size = kernel_size
        filter_params = (hidden_size, hidden_size, kernel_size, kernel_size, kernel_size)
        output_shape = (-1, self.hidden_size, self.grid_size, self.grid_size, self.grid_size)

        self.update = GRUGate3D(self.fan_in, filter_params, output_shape)
        self.reset = GRUGate3D(self.fan_in, filter_params, output_shape)
        self.output = GRUGate3D(self.fan_in, filter_params, output_shape)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, encoder_out, h):
        """
        :param encoder_out: (batch_size, 1024,)
        :param h: (batch_size,128,4,4,4)
        :return:
            out: (B, C, D, H, W) = (batch_size, 128, 4, 4, 4) (same size as h)
        """
        t_x_s_reset = self.reset(encoder_out, h)
        t_x_s_update = self.update(encoder_out, h)

        reset_gate = self.sigmoid(t_x_s_reset)

        update_gate = self.sigmoid(t_x_s_update)
        complement_update_gate = 1 - update_gate

        rs = reset_gate * h
        t_x_rs = self.output(encoder_out, rs)
        tanh_t_x_rs = self.tanh(t_x_rs)

        gru_out = complement_update_gate * h + update_gate * tanh_t_x_rs

        return gru_out, update_gate
