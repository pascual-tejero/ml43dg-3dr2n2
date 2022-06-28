import torch

from encoder import Encoder

# The input to the decoder is the hidden state of the 3DConvLSTM,
# a tensor of size (B, C, D, H, W), where (D, H, W) are the 3D LSTM grid,
# (4, 4, 4) in the paper, C is the hidden state size at each point in the grid
# (128 according to the original code), and B is the batch size.

model = Encoder(type="simple")

B, C, H, W = (1, 3, 127, 127)

x = torch.rand(B,C,H,W)

out = model(x)
print(out.shape)