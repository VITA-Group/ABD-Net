import os
from torch.nn import functional as F
from torch import nn


class SimpleDropoutOptimizer(nn.Module):

    def __init__(self, p):

        super().__init__()
        if p is not None:
            self.dropout = nn.Dropout(p=p)
        else:
            self.dropout = None

    def forward(self, x):

        if self.dropout is not None:
            x = self.dropout(x)

        return x


class DropoutOptimizer(nn.Module):

    def __init__(self, args):

        super().__init__()

        self.args = args
        self.epoch = self.__p = 0
        self.__training = not args.evaluate

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_training(self, training):
        self.__training = training

    @property
    def p(self):

        if not self.__training:
            return self.__p

        dropout_settings = self.args.dropout

        if dropout_settings == 'none':
            return 0

        try:
            max_dropout = float(os.environ.get('max_dropout'))
        except (ValueError, TypeError):
            max_dropout = 0.5

        if dropout_settings == 'fix':
            return max_dropout

        # print(self.epoch)
        p = .2 + .1 * (self.epoch // 10)
        p = min(p, max_dropout)

        return p

    def set_p(self, p):

        if self.__training:
            raise RuntimeError('Cannot explicitly set dropout during training')

        assert isinstance(p, (int, float))

        self.__p = p

    def forward(self, x):

        p = self.p
        # print('Dropout p', p)
        # print(p, self.__training)
        if p > 0:
            return F.dropout(x, p=p, training=self.__training)
        else:
            return x
