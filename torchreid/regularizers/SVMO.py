from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn

import logging
logger = logging.getLogger(__name__)

w_rate = 1e-4


class SVMORegularizer(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.beta = args['ow_beta']

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

    def forward(self, W: 'S x C x H x W'):

        logger.debug('svmo')

        # old_W = W
        old_size = W.size()

        if old_size[0] == 1:
            return 0

        W = W.view(old_size[0], -1).permute(1, 0)  # (C x H x W) x S

        smallest, largest = self.get_singular_values(W)
        return (
            self.beta * 10 * (largest - smallest)**2
        ).squeeze()
