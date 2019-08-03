from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn

import os

w_rate = 1e-4


class SORegularizer(nn.Module):

    def __init__(self, controller):
        super().__init__()

        os_beta = None

        # try:
        #     os_beta = float(os.environ.get('beta'))
        # except (ValueError, TypeError):
        #     raise RuntimeError('No beta specified. ABORTED.')
        # self.beta = os_beta
        self.param_controller = controller

    def dominant_eigenvalue(self, A: 'N x N'):

        N, _ = A.size()
        x = torch.rand(N, 1, device='cuda')

        Ax = (A @ x).squeeze()
        AAx = (A @ Ax).squeeze()

        return torch.norm(AAx, p=2) / torch.norm(Ax, p=2)

    def forward(self, W: 'C x S x H x W'):

        # old_W = W
        old_size = W.size()

        W = W.view(old_size[0], -1).permute(1, 0)
        # W = W.permute(2, 3, 0, 1).view(old_size[0] * old_size[2] * old_size[3], old_size[1])

        d_ev = self.dominant_eigenvalue(
            W.permute(1, 0) @ W - torch.eye(old_size[0], device='cuda')
        )
        return (
            self.param_controller.get_value() * d_ev
        ).squeeze()
