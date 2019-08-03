from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn

import os

w_rate = 1e-4


class SVDORegularizer(nn.Module):

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

        # Ax = (A @ x).squeeze()
        # AAx = (A @ Ax).squeeze()

        # return torch.norm(AAx, p=2) / torch.norm(Ax, p=2)

        Ax = (A @ x)
        AAx = (A @ Ax)

        return AAx.permute(1, 0) @ Ax / (Ax.permute(1, 0) @ Ax)

        # for _ in range(1):
        #     x = A @ x
        # numerator = torch.bmm(
        #     torch.bmm(A, x).permute(0, 2, 1),
        #     x
        # ).squeeze()
        # denominator = torch.bmm(
        #     x.permute(0, 2, 1),
        #     x
        # ).squeeze()

        # return numerator / denominator

    def get_singular_values(self, A: 'M x N, M >= N'):

        ATA = A.permute(1, 0) @ A
        N, _ = ATA.size()
        largest = self.dominant_eigenvalue(ATA)
        I = torch.eye(N, device='cuda')  # noqa
        I = I * largest  # noqa
        tmp = self.dominant_eigenvalue(ATA - I)
        return tmp + largest, largest

    def forward(self, W: 'C x S x H x W'):

        # old_W = W
        old_size = W.size()

        W = W.view(old_size[0], -1).permute(1, 0)

        smallest, largest = self.get_singular_values(W)
        return (
            self.param_controller.get_value() * (largest / smallest - 1) ** 2
        ).squeeze()
