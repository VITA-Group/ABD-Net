from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, num_classes, args, use_gpu=True):
        super(TripletLoss, self).__init__()
        margin = args['margin']
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

        from .cross_entropy_loss import CrossEntropyLoss
        self.xent = CrossEntropyLoss(num_classes=num_classes, use_gpu=use_gpu, label_smooth=args['label_smooth'])
        self.lambda_xent = args['lambda_xent']
        self.lambda_htri = args['lambda_htri']

    def _forward(self, inputs, targets):

        if not isinstance(inputs, tuple):
            inputs_tuple = (inputs,)
        else:
            inputs_tuple = inputs

        results = sum([self.apply_loss(x, targets) for x in inputs_tuple])
        return results / len(inputs_tuple)

    def forward(self, inputs, targets):

        xent_loss = self.xent(inputs, targets)
        htri_loss = self._forward(inputs[2], targets)

        return self.lambda_xent * xent_loss + self.lambda_htri * htri_loss

    def apply_loss(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss
