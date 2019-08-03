import torch
import torch.nn as nn


class NoneRegularizer(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, _):
        return torch.tensor(0.0).cuda()
