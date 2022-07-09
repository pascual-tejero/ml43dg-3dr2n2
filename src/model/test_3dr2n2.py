import torch

from encoder import Encoder
from decoder import Decoder
from convRNN3D import ConvGRU3D

# The input to the decoder is the hidden state of the 3DConvLSTM,
# a tensor of size (B, C, D, H, W), where (D, H, W) are the 3D LSTM grid,
# (4, 4, 4) in the paper, C is the hidden state size at each point in the grid
# (128 according to the original code), and B is the batch size.

enDecType = "residual" # Options: ['simple', 'residual']
conv_3d_rnn_kernel_size = 3 # Options: [1, 3]

encoder = Encoder(type=enDecType)

num_views, B, C, H, W = 5, 32, 3, 127, 127
in_shape = (num_views, B, C, H, W)

X = torch.rand(in_shape)
print(f"Shape of X: {X.shape}")

encoder_out = encoder(X[0])
print(f"Shape of Encoder Output: {encoder_out.shape}")

convGRU3D_model = ConvGRU3D(fan_in=encoder_out.shape[1], hidden_size=128, grid_size=4, kernel_size=conv_3d_rnn_kernel_size)
h_shape = (encoder_out.shape[0], convGRU3D_model.hidden_size, convGRU3D_model.grid_size, convGRU3D_model.grid_size, convGRU3D_model.grid_size)
print(h_shape)
h = torch.zeros(h_shape) # Hidden state is initialized as 0
print(f"Shape of Hidden State (GRU): {h.shape}")
h_out, u_out = convGRU3D_model(encoder_out, h)
print(f"Shape of Convolutional 3D RNN (GRU) Output: {h_out.shape}")

decoder = Decoder(type=enDecType)

decoder_out = decoder(h_out)
print(f"Shape of Decoder Output: {decoder_out.shape}")





