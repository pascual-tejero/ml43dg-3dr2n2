import torch
import torch.nn as nn

from .layers import GRUGate3D


class ConvGRU3D(nn.Module):
    def __init__(self, fan_in, hidden_size, grid_size, kernel_size):
        super(ConvGRU3D, self).__init__()
        self.fan_in = fan_in
        self.hidden_size = hidden_size
        self.grid_size = grid_size
        self.kernel_size = kernel_size
        filter_params = (
            hidden_size,
            hidden_size,
            kernel_size,
            kernel_size,
            kernel_size,
        )
        output_shape = (
            -1,
            self.hidden_size,
            self.grid_size,
            self.grid_size,
            self.grid_size,
        )

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


class ConvLSTM3D(nn.Module):
    # Implementation of 3D Convolutional LSTM grid according to the 3DR2N2 paper.
    # Input - feature vector (B, C), previous hidden state (B, Nh, N, N, N)
    # Output - (B, Nh, N, N, N)
    def __init__(self,
                 feature_vector_length,
                 hidden_layer_length,
                 grid_size=4,
                 kernel_size=3,
                 stride=1,
                 padding=1
                 ):
        super(ConvLSTM3D, self).__init__()

        self.feature_vector_length = feature_vector_length
        self.hidden_layer_length = hidden_layer_length
        self.grid_size = grid_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        gate_channels = 3 * self.hidden_layer_length
        self.gate_channels = gate_channels

        self.linear = nn.Linear(feature_vector_length,
                                gate_channels * grid_size * grid_size * grid_size)
        self.conv3d = nn.Conv3d(in_channels=hidden_layer_length,
                                out_channels=gate_channels,
                                kernel_size=(kernel_size, kernel_size, kernel_size),
                                stride=1, padding=1, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()
        self.conv3d.reset_parameters()

    def forward(self, input, hidden):
        hx, cx = hidden
        gates = self.linear(input).view(-1, self.gate_channels, self.grid_size, self.grid_size, self.grid_size) \
                + self.conv3d(hx)
        ingate, forgetgate, cellgate = gates.chunk(3, 1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = torch.tanh(cy)

        return hy, cy
