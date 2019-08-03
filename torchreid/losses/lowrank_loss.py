from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn

import os

from .cross_entropy_loss import CrossEntropyLoss

CONSTRAINT_WEIGHTS = os.environ.get('constraint_weights') is not None


class LowRankLoss(nn.Module):

    def __init__(self, num_classes, *, use_gpu=True, label_smooth=True, beta=None):
        super().__init__()

        os_beta = None

        try:
            os_beta = float(os.environ.get('beta'))
        except ValueError:
            raise RuntimeError('No beta specified. ABORTED.')

        self.beta = beta if not os_beta else os_beta
        self.xent_loss = CrossEntropyLoss(num_classes, use_gpu, label_smooth)

    def forward(self, inputs, pids):

        x, y, _, weights = inputs

        if CONSTRAINT_WEIGHTS:
            height, width = weights.size()
            batches = 1
            channels = height
            W = weights.view(1, height, width)
        else:
            batches, channels, height, width = x.size()
            W = x.view(batches, channels, -1)
        WT = W.permute(0, 2, 1)
        WWT = torch.bmm(W, WT)
        I = torch.eye(channels).expand(batches, channels, channels).cuda()  # noqa
        delta = WWT - I
        norm = torch.norm(delta.view(batches, -1), 2, 1) ** 2
        return norm.sum() * self.beta + self.xent_loss(y, pids)
