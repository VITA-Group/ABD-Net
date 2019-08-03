import torch
import torch.nn as nn
import torch.nn.functional as F


from torchreid.utils.environ import get_env_or_raise

THRESHOLD = get_env_or_raise(float, 'IL_THRESHOLD')
NORM = get_env_or_raise(str, 'IL_NORM')
print(THRESHOLD, NORM)


class IncidenceLoss(nn.Module):

    def forward(self, x, pids):

        _, _, features, _ = x
        features: 'N x C x W x H'

        W = features.view(features.size(0), -1)
        W = F.normalize(W, p=2, dim=1)
        WWT = W @ W.permute(1, 0)

        targets = pids.data.cpu().numpy()
        # print(list(targets))
        # print('Kinds:', len(set(list(targets))))
        A = [
            [1. if i == j else 0. for j in targets]
            for i in targets
        ]
        A = torch.tensor(A, requires_grad=True).cuda()
        # print('WWT', WWT.size())
        # print('A', A.size())
        #

        W = WWT - A

        norm_p = {'inf': float('inf'), 'l2': 2}[NORM]

        W = torch.clamp(torch.abs(W), min=THRESHOLD)

        p = torch.norm(W, p=norm_p, dim=1)
        return p.sum()

        # return ((WWT - A)**2).sum() ** (1 / 2)
