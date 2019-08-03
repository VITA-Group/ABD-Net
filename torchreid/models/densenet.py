from __future__ import absolute_import
from __future__ import division

from collections import OrderedDict
import re

import torch
import torch.nn as nn
from torch.utils import model_zoo
from torch.nn import functional as F

from torchreid.components.shallow_cam import ShallowCAM
from torchreid.components import branches

from copy import deepcopy

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


class DummyFD(nn.Module):

    def __init__(self, fd_getter):

        super().__init__()
        self.fd_getter = fd_getter

    def forward(self, x):

        B, C, H, W = x.shape

        for cs, cam in self.fd_getter().cam_modules:
            # try:
            #     c_tensor = torch.tensor(cs).cuda()
            # except RuntimeError:
            c_tensor = torch.tensor(cs).cuda()

            new_x = x[:, c_tensor]
            new_x = cam(new_x)
            x[:, c_tensor] = new_x

        return x


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(
            num_input_features, bn_size *
            growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(
            bn_size * growth_rate, growth_rate,
            kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, stride=2):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=stride, stride=stride))


class DenseNet(nn.Module):
    """Densely connected network.

    Reference:
        Huang et al. Densely Connected Convolutional Networks. CVPR 2017.

    Public keys:
        - ``densenet121``: DenseNet121.
        - ``densenet169``: DenseNet169.
        - ``densenet201``: DenseNet201.
        - ``densenet161``: DenseNet161.
        - ``densenet121_fc512``: DenseNet121 + FC.
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, fc_dims=None, dropout_p=None, last_stride=2, **kwargs):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features

        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:

                if i == len(block_config) - 2:
                    stride = last_stride
                else:
                    stride = 2

                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2, stride=stride)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        self.feature_dim = num_features
    def _init_params(self, x):
        for m in x.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def init_pretrained_weights(model, model_url):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)

    # '.'s are no longer allowed in module names, but pervious _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
    for key in list(pretrain_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            pretrain_dict[new_key] = pretrain_dict[key]
            del pretrain_dict[key]

    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
    print('Initialized model with pretrained weights from {}'.format(model_url))


"""
Dense network configurations:
--
densenet121: num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16)
densenet169: num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32)
densenet201: num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32)
densenet161: num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24)
"""

def densenet121_backbone():

    model = DenseNet(
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        last_stride=1,
        fc_dims=None,
        dropout_p=None,
    )
    init_pretrained_weights(model, model_urls['densenet121'])
    return model


def densenet161_backbone():

    model = DenseNet(
        num_init_features=96,
        growth_rate=48,
        block_config=(6, 12, 36, 24),
        last_stride=1,
        fc_dims=None,
        dropout_p=None,
    )
    init_pretrained_weights(model, model_urls['densenet161'])
    return model

def _copy_dense_layer(denseblock, start, end):

    return [
        deepcopy(getattr(denseblock, 'denselayer%s' % i))
        for i in range(start, end + 1)
    ]


class DenseNetCommonBranch(nn.Module):

    def __init__(self, owner, backbone, args):

        super().__init__()
        type_ = owner.type_

        self.backbone1 = nn.Sequential(*backbone.features[:6])
        self.shallow_cam = ShallowCAM(args, 128)

        if type_ == 'd4':
            self.backbone2 = nn.Sequential(*backbone.features[6:-2])
        elif type_ == 't3_d4':
            self.backbone2 = nn.Sequential(*backbone.features[6:-3])
        elif type_ == 'd3_t3_d4':
            denseblock_3 = backbone.features[-4]
            total_layers = len(denseblock_3._modules)
            self.backbone2 = nn.Sequential(
                *backbone.features[6:-4],
                *_copy_dense_layer(denseblock_3, 1, total_layers // 6)
            )

    def backbone_modules(self):

        return [self.backbone1, self.backbone2]

    def forward(self, x):

        x = self.backbone1(x)
        intermediate = x = self.shallow_cam(x)
        x = self.backbone2(x)

        return x, intermediate

class DenseNetDeepBranch(nn.Module):

    def __init__(self, owner, backbone, args):

        super().__init__()
        type_ = owner.type_

        if type_ == 'd4':
            backbone_ = nn.Sequential(*backbone.features[-2:])
        elif type_ == 't3_d4':
            backbone_ = nn.Sequential(*backbone.features[-3:])
        elif type_ == 'd3_t3_d4':
            denseblock_3 = backbone.features[-4]
            total_layers = len(denseblock_3._modules)
            backbone_ = nn.Sequential(
                *_copy_dense_layer(denseblock_3, total_layers // 6 + 1, total_layers),
                *backbone.features[-3:]
            )

        self.backbone = deepcopy(backbone_)
        self.out_dim = backbone.feature_dim

    def backbone_modules(self):

        return [self.backbone]

    def forward(self, x):
        return self.backbone(x)


class MultiBranchDenseNet(branches.MultiBranchNetwork):

    def __init__(self, backbone, args, num_classes, type_, **kwargs):
        self.type_ = type_
        super().__init__(backbone, args, num_classes, **kwargs)

    def _get_common_branch(self, backbone, args):

        return DenseNetCommonBranch(self, backbone, args)

    def _get_middle_subbranch_for(self, backbone, args, last_branch_class):

        return DenseNetDeepBranch(self, backbone, args)


def _make_densenet(backbone_name, type_):

    def _initializer(num_classes, args, **kw):
        backbone = globals()[backbone_name + '_backbone']()
        return MultiBranchDenseNet(backbone, args, num_classes, type_)

    return _initializer


model_mapping = {}

for backbone_name in ['densenet121', 'densenet161']:

    for type_ in ['d4', 't3_d4', 'd3_t3_d4']:

        name = backbone_name + '_' + type_

        model_mapping[name] = globals()[name] = _make_densenet(backbone_name, type_)

model_mapping['densenet121'] = densenet121 = globals()['densenet121_d4']
model_mapping['densenet161'] = densenet161 = globals()['densenet161_d4']
