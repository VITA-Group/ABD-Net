from __future__ import absolute_import
from __future__ import division

import os
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torch.utils.model_zoo as model_zoo
from copy import deepcopy


channels = {
    'a': [4, 40, 64, 68, 70, 71, 101, 102, 127, 141, 152, 158, 162, 164, 171, 172, 175, 186, 201, 209, 225, 227, 246],
    'b': [2, 11, 12, 17, 23, 24, 28, 30, 36, 39, 47, 48, 49, 57, 59, 60, 61, 66, 77, 78, 83, 87, 88, 89, 91, 93, 99, 107, 110, 117, 120, 121, 123, 124, 126, 133, 139, 140, 145, 151, 155, 165, 168, 174, 180, 185, 192, 199, 202, 211, 216, 217, 219, 222, 230, 239, 245, 247, 249, 251],
    'c': [0, 1, 3, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 18, 19, 20, 21, 22, 25, 26, 27, 29, 31, 32, 33, 34, 35, 37, 38, 41, 42, 43, 44, 45, 46, 50, 51, 52, 53, 54, 55, 56, 58, 62, 63, 65, 67, 69, 72, 73, 74, 75, 76, 79, 80, 81, 82, 84, 85, 86, 90, 92, 94, 95, 96, 97, 98, 100, 103, 104, 105, 106, 108, 109, 111, 112, 113, 114, 115, 116, 118, 119, 122, 125, 128, 129, 130, 131, 132, 134, 135, 136, 137, 138, 142, 143, 144, 146, 147, 148, 149, 150, 153, 154, 156, 157, 159, 160, 161, 163, 166, 167, 169, 170, 173, 176, 177, 178, 179, 181, 182, 183, 184, 187, 188, 189, 190, 191, 193, 194, 195, 196, 197, 198, 200, 203, 204, 205, 206, 207, 208, 210, 212, 213, 214, 215, 218, 220, 221, 223, 224, 226, 228, 229, 231, 232, 233, 234, 235, 236, 237, 238, 240, 241, 242, 243, 244, 248, 250, 252, 253, 254, 255],

}

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetBackbone(nn.Module):
    """
    Residual network

    Reference:
    He et al. Deep Residual Learning for Image Recognition. CVPR 2016.
    """

    def __init__(self, block, layers, last_stride=2):
        self.inplanes = 64
        super().__init__()

        # backbone network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


class ResNet(nn.Module):
    """
    Residual network

    Reference:
    He et al. Deep Residual Learning for Image Recognition. CVPR 2016.
    """

    def __init__(self, num_classes,
                 last_stride=2,
                 fc_dims=None,
                 dropout_p=None,
                 *,
                 fd_config=None,
                 attention_config=None,
                 dropout_optimizer=None,
                 sum_fusion: bool=False,
                 tricky: int=0,
                 **kwargs):

        self.sum_fusion = True
        super(ResNet, self).__init__()
        self.feature_dim = 2048

        # backbone network
        backbone = ResNetBackbone(Bottleneck, [3, 4, 6, 3], last_stride=1)
        init_pretrained_weights(backbone, model_urls['resnet50'])
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        normal_branch_stride = 2

        self.layer4_normal_branch = nn.Sequential(
            Bottleneck(
                1024,
                512,
                stride=normal_branch_stride,
                downsample=nn.Sequential(
                    nn.Conv2d(
                        1024, 2048, kernel_size=1, stride=normal_branch_stride, bias=False
                    ),
                    nn.BatchNorm2d(2048)
                )
            ),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512)
        )
        self.layer4_normal_branch.load_state_dict(backbone.layer4.state_dict())

        # Begin Feature Distilation
        if fd_config is None:
            fd_config = {'parts': (), 'use_conv_head': False}
        from .tricks.feature_distilation import FeatureDistilationTrick
        self.feature_distilation = FeatureDistilationTrick(
            fd_config['parts'],
            channels=channels,
            use_conv_head=fd_config['use_conv_head']
        )
        # End Feature Distilation

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        num_features = 2048
        # Begin Attention Module
        if attention_config is None:
            attention_config = {'parts': (), 'use_conv_head': False}
        from .tricks.attention import AttentionModule

        self.attention_module = AttentionModule(
            attention_config['parts'],
            fc_dims[0],
            use_conv_head=attention_config['use_conv_head'],
            sum_fusion=self.sum_fusion
        )
        if not sum_fusion:
            self.feature_dim = num_features = num_features + self.attention_module.output_dim
        # End Attention Module

        # Begin Dropout Module
        if dropout_optimizer is None:
            from .tricks.dropout import SimpleDropoutOptimizer
            dropout_optimizer = SimpleDropoutOptimizer(dropout_p)
        # End Dropout Module

        self.fc = self._construct_fc_layer(fc_dims, num_features, dropout_optimizer)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        self.reduction_tr = nn.Sequential(
            nn.Conv2d(2048, fc_dims[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(fc_dims[0]),
            nn.ReLU(inplace=True)
        )
        self.classifier_tr = nn.Linear(fc_dims[0], num_classes)
        self._init_params(self.reduction_tr)
        self._init_params(self.classifier_tr)

        self._init_params(self.feature_distilation)
        self._init_params(self.attention_module)
        self._init_params(self.fc)
        self._init_params(self.classifier)

    def backbone_convs(self):

        convs = [
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer4_normal_branch,
        ]

        return convs

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_optimizer):
        """
        Construct fully connected layer

        - fc_dims (list or tuple): dimensions of fc layers, if None,
                                   no fc layers are constructed
        - input_dim (int): input dimension
        - dropout_p (float): dropout probability, if None, dropout is unused
        """

        if fc_dims is None:
            self.feature_dim = input_dim
            return None

        assert isinstance(fc_dims, (list, tuple)), "fc_dims must be either list or tuple, but got {}".format(type(fc_dims))

        layers = []

        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(dropout_optimizer)
            # if dropout_p is not None:
            #     layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    def _init_params(self, x):
        for m in x.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1, 0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.normal_(m.weight, 1, 0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)

        layer5 = x

        B, C, H, W = x.shape

        for cs, cam in self.feature_distilation.cam_modules:
            c_tensor = torch.tensor(cs).cuda()

            new_x = x[:, c_tensor]
            new_x = cam(new_x)
            x[:, c_tensor] = new_x

        x = self.layer2(x)
        x = self.layer3(x)

        triplet_features = []
        xent_features = []
        predict_features = []

        # normal branch
        x1 = x
        x1 = self.layer4_normal_branch(x1)
        x1 = self.global_avgpool(x1).squeeze()
        x1 = self.fc(x1)
        triplet_features.append(x1)
        predict_features.append(x1)
        x1 = self.classifier(x1)
        xent_features.append(x1)

        # our branch
        x2 = x
        f = self.layer4(x2)
        f = self.reduction_tr(f)
        feature_dict, _ = self.attention_module(f)
        feature_dict['before'] = f
        f = sum(feature_dict.values())
        feature_dict['after'] = f
        v = self.global_avgpool(f).squeeze()
        feature_dict['layer5'] = layer5

        triplet_features.append(v)
        predict_features.append(v)
        v = self.classifier_tr(v)
        xent_features.append(v)

        if not self.training:
            return torch.cat(predict_features, 1)

        return None, tuple(xent_features), tuple(triplet_features), feature_dict


def init_pretrained_weights(model, model_url):
    """
    Initialize model with pretrained weights.
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
    print("Initialized model with pretrained weights from {}".format(model_url))


"""
Residual network configurations:
--
resnet18: block=BasicBlock, layers=[2, 2, 2, 2]
resnet34: block=BasicBlock, layers=[3, 4, 6, 3]
resnet50: block=Bottleneck, layers=[3, 4, 6, 3]
resnet101: block=Bottleneck, layers=[3, 4, 23, 3]
resnet152: block=Bottleneck, layers=[3, 8, 36, 3]
"""


def resnet50_best(num_classes, loss, pretrained='imagenet', **kwargs):
    return ResNet(
        num_classes=num_classes,
        last_stride=1,
        dropout_p=None,
        sum_fusion=True,
        fc_dims=[1024],
        fd_config={
            'parts': 'ab',
            'use_conv_head': False
        },
        attention_config={
            'parts': ('cam', 'pam'),
            'use_conv_head': False
        },
        **kwargs
    )
