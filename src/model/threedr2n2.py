import torch.nn as nn
from utils import initialize_tensor
from encoder import Encoder
from decoder import Decoder
from convRNN3D import ConvGRU3D

class ThreeDeeR2N2(nn.Module):
    def __init__(self, encoderDecoder_type, convRNN3D_type, convRNN3D_kernel_size, batch_size):
        super(ThreeDeeR2N2, self).__init__()
        self.batch_size = batch_size
        self.image_shape = (3, 127, 127)
        self.input_shape = (self.batch_size, *self.image_shape)

        self.grid_convRNN3D = 4
        self.hidden_size = 128
        self.h_shape = (self.batch_size, self.hidden_size, self.grid_convRNN3D, self.grid_convRNN3D, self.grid_convRNN3D)

        self.encoder, self.decoder, self.convRNN3D = None, None, None

        self.initialize_encoder(encoderDecoder_type)
        self.initialize_decoder(encoderDecoder_type)
        self.initialize_convRNN3d(convRNN3D_type, convRNN3D_kernel_size)

    def initialize_encoder(self, type):
        if type.lower() not in ['simple','residual']:
            raise Exception("Type Error: Encoder")
        self.encoder = Encoder(type.lower())

    def initialize_decoder(self, type):
        if type.lower() not in ['simple','residual']:
            raise Exception("Type Error: Decoder")
        self.decoder = Decoder(type.lower())

    def initialize_convRNN3d(self, type, kernel_size):
        if type.lower() not in ['lstm', 'gru']:
            raise Exception("Type Error: 3D Convolutional RNN")
        if kernel_size not in [1, 3]:
            raise Exception("Value Error: Kernel size of 3D Convolutional RNN")
        if type == 'gru':
            self.convRNN3D = ConvGRU3D(
                fan_in=1024,
                hidden_size=self.hidden_size,
                grid_size=self.grid_convRNN3D,
                kernel_size=kernel_size
            )
        else:
            self.convRNN3D = None

    def forward(self, X):
        if self.encoder is None:
            raise Exception("The encoder is not initialized!")
        if self.convRNN3D is None:
            raise Exception("The convolutional recurrent network is not initialized!")
        if self.decoder is None:
            raise Exception("The decoder is not initialized!")

        h, u = initialize_tensor(self.h_shape), initialize_tensor(self.h_shape)
        u_list = []

        """
        x is the input and the size of x is (num_views, batch_size, channels, heights, widths).
        h and u is the hidden state and activation of last time step respectively.
        The following loop computes the forward pass of the whole network. 
        """
        for time_step in range(X.shape[0]):
            encoder_out = self.encoder(X[time_step])
            convRNN3D_out, update_gate = self.convRNN3D(encoder_out, h)
            h = convRNN3D_out
            u = update_gate
            u_list.append(u)
        decoder_out = self.decoder(h)
        return decoder_out