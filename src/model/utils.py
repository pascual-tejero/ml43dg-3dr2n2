import torch
from torch.autograd import Variable


def initialize_tensor(tensor_shape):
    tensor = torch.zeros(tensor_shape)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor)
