import torch

from decoder import Decoder

# The input to the decoder is the hidden state of the 3DConvLSTM,
# a tensor of size (B, C, D, H, W), where (D, H, W) are the 3D LSTM grid,
# (4, 4, 4) in the paper, C is the hidden state size at each point in the grid
# (128 according to the original code), and B is the batch size.

model = Decoder(type="residual")

B, C, D, H, W = (1, 128, 4, 4, 4)

x = torch.rand(B,C,D,H,W)

out = model(x)
print(out.shape)
