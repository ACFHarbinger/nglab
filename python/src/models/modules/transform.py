import torch


class Transpose(torch.nn.Module):
    def __init__(self, dims=(-1, 1)):
        super(Transpose, self).__init__()
        self.dims = dims

    def forward(self, x):
        return torch.transpose(x, *self.dims)