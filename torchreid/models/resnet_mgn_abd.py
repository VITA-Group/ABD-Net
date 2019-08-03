from __future__ import absolute_import
from __future__ import division

import os
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torch.utils.model_zoo as model_zoo
from copy import deepcopy

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


class ResNetABDMGN(nn.Module):
    """
    Residual network

    Reference:
    He et al. Deep Residual Learning for Image Recognition. CVPR 2016.
    """

    def __init__(self, num_classes,
                 dropout_p=None,
                 *,
                 fd_config=None,
                 attention_config=None,
                 dropout_optimizer=None,
                 **kwargs):

        self.sum_fusion = True
        last_stride = 1

        super(ResNetABDMGN, self).__init__()
        self.feature_dim = 2048  # 512 * block.expansion

        # backbone network
        backbone = ResNetBackbone(Bottleneck, [3, 4, 6, 3], 2)
        init_pretrained_weights(backbone, model_urls['resnet50'])
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3[0]

        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(backbone.layer4.state_dict())

        self.layer4_1 = nn.Sequential(
            *deepcopy(backbone.layer3[1:]),
            deepcopy(backbone.layer4)
        )
        self.layer4_2 = nn.Sequential(
            *deepcopy(backbone.layer3[1:]),
            deepcopy(res_p_conv5)
        )
        self.layer4_3 = deepcopy(self.layer4_2)
        # self.layer4_normal_branch = nn.Sequential(
        #     Bottleneck(
        #         1024,
        #         512,
        #         stride=normal_branch_stride,
        #         downsample=nn.Sequential(
        #             nn.Conv2d(
        #                 1024, 2048, kernel_size=1, stride=normal_branch_stride, bias=False
        #             ),
        #             nn.BatchNorm2d(2048)
        #         )
        #     ),
        #     Bottleneck(2048, 512),
        #     Bottleneck(2048, 512)
        # )
        # self.layer4_normal_branch.load_state_dict(backbone.layer4.state_dict())
        #
        self.layer4_abd = deepcopy(self.layer4_3)

        feats = 256

        self.maxpool_zg_p1 = nn.MaxPool2d(kernel_size=(12, 4))
        self.maxpool_zg_p2 = nn.MaxPool2d(kernel_size=(24, 8))
        self.maxpool_zg_p3 = nn.MaxPool2d(kernel_size=(24, 8))
        self.maxpool_zp2 = nn.MaxPool2d(kernel_size=(12, 8))
        self.maxpool_zp3 = nn.MaxPool2d(kernel_size=(8, 8))

        self.reduction = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())

        self._init_reduction(self.reduction)

        self.fc_id_2048_0 = nn.Linear(feats, num_classes)
        self.fc_id_2048_1 = nn.Linear(feats, num_classes)
        self.fc_id_2048_2 = nn.Linear(feats, num_classes)

        self.fc_id_256_1_0 = nn.Linear(feats, num_classes)
        self.fc_id_256_1_1 = nn.Linear(feats, num_classes)
        self.fc_id_256_2_0 = nn.Linear(feats, num_classes)
        self.fc_id_256_2_1 = nn.Linear(feats, num_classes)
        self.fc_id_256_2_2 = nn.Linear(feats, num_classes)

        self._init_fc(self.fc_id_2048_0)
        self._init_fc(self.fc_id_2048_1)
        self._init_fc(self.fc_id_2048_2)

        self._init_fc(self.fc_id_256_1_0)
        self._init_fc(self.fc_id_256_1_1)
        self._init_fc(self.fc_id_256_2_0)
        self._init_fc(self.fc_id_256_2_1)
        self._init_fc(self.fc_id_256_2_2)

        # Begin Feature Distilation
        if fd_config is None:
            fd_config = {'parts': (), 'use_conv_head': False}
        from .tricks.feature_distilation import FeatureDistilationTrick
        self.feature_distilation = FeatureDistilationTrick(
            fd_config['parts'],
            channels=channels,
            use_conv_head=fd_config['use_conv_head']
        )
        self._init_params(self.feature_distilation)
        self.dummy_fd = DummyFD(lambda: self.feature_distilation)
        # End Feature Distilation

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        num_features = 2048
        # Begin Attention Module
        self.get_attention_module()
        self.attention_config = attention_config
        # End Attention Module

        # Begin Dropout Module
        if dropout_optimizer is None:
            from .tricks.dropout import SimpleDropoutOptimizer
            dropout_optimizer = SimpleDropoutOptimizer(dropout_p)
        # End Dropout Module

        if os.environ.get('dropout_reduction'):
            dropout = [dropout_optimizer]
        else:
            dropout = []

        fc_dims = [1024]
        # self.fc = self._construct_fc_layer(fc_dims, num_features, dropout_optimizer)
        # self.classifier = nn.Linear(self.feature_dim, num_classes)

        self.reduction_tr = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            *dropout
        )
        self._init_params(self.reduction_tr)

        try:
            part_num = int(os.environ.get('part_num'))
        except (TypeError, ValueError):
            part_num = 2

        self.part_num = part_num

        for i in range(1, part_num + 1):
            c = nn.Linear(1024, num_classes)
            setattr(self, 'classifier_p{}'.format(i), c)
            self._init_params(c)

        # self._init_params(self.fc)
        # self._init_params(self.classifier)

    def get_attention_module(self):

        from .tricks.attention import DANetHead, CAM_Module, PAM_Module

        in_channels = 1024
        out_channels = 1024

        self.before_module1 = DANetHead(in_channels, out_channels, nn.BatchNorm2d, lambda _: lambda x: x)
        self.pam_module1 = DANetHead(in_channels, out_channels, nn.BatchNorm2d, PAM_Module)
        self.cam_module1 = DANetHead(in_channels, out_channels, nn.BatchNorm2d, CAM_Module)
        self.sum_conv1 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(out_channels, out_channels, 1))

        self._init_params(self.before_module1)
        self._init_params(self.cam_module1)
        self._init_params(self.pam_module1)
        self._init_params(self.sum_conv1)

    def backbone_convs(self):

        convs = [
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4_1,
            self.layer4_2,
            self.layer4_3,
            self.layer4_abd,
        ]

        return convs

    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        # nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        # nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)

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
        if x is None:
            return

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

        # B, C, H, W = x.shape

        # for cs, cam in self.feature_distilation.cam_modules:
        #     c_tensor = torch.tensor(cs).cuda()

        #     new_x = x[:, c_tensor]
        #     new_x = cam(new_x)
        #     x[:, c_tensor] = new_x

        # layer5 = x

        x = self.dummy_fd(x)
        layer5 = x
        x = self.layer2(x)
        x = self.layer3(x)

        triplet_features = []
        xent_features = []
        predict_features = []

        # normal branch
        p1 = self.layer4_1(x)
        p2 = self.layer4_2(x)
        p3 = self.layer4_3(x)

        zg_p1 = self.maxpool_zg_p1(p1)
        zg_p2 = self.maxpool_zg_p2(p2)
        zg_p3 = self.maxpool_zg_p3(p3)

        zp2 = self.maxpool_zp2(p2)
        z0_p2 = zp2[:, :, 0:1, :]
        z1_p2 = zp2[:, :, 1:2, :]

        zp3 = self.maxpool_zp3(p3)
        z0_p3 = zp3[:, :, 0:1, :]
        z1_p3 = zp3[:, :, 1:2, :]
        z2_p3 = zp3[:, :, 2:3, :]

        fg_p1 = self.reduction(zg_p1).squeeze(dim=3).squeeze(dim=2)
        fg_p2 = self.reduction(zg_p2).squeeze(dim=3).squeeze(dim=2)
        fg_p3 = self.reduction(zg_p3).squeeze(dim=3).squeeze(dim=2)
        f0_p2 = self.reduction(z0_p2).squeeze(dim=3).squeeze(dim=2)
        f1_p2 = self.reduction(z1_p2).squeeze(dim=3).squeeze(dim=2)
        f0_p3 = self.reduction(z0_p3).squeeze(dim=3).squeeze(dim=2)
        f1_p3 = self.reduction(z1_p3).squeeze(dim=3).squeeze(dim=2)
        f2_p3 = self.reduction(z2_p3).squeeze(dim=3).squeeze(dim=2)

        l_p1 = self.fc_id_2048_0(fg_p1)
        l_p2 = self.fc_id_2048_1(fg_p2)
        l_p3 = self.fc_id_2048_2(fg_p3)

        l0_p2 = self.fc_id_256_1_0(f0_p2)
        l1_p2 = self.fc_id_256_1_1(f1_p2)
        l0_p3 = self.fc_id_256_2_0(f0_p3)
        l1_p3 = self.fc_id_256_2_1(f1_p3)
        l2_p3 = self.fc_id_256_2_2(f2_p3)

        predict = torch.cat([fg_p1, fg_p2, fg_p3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3], dim=1)
        predict_features += ([fg_p1, fg_p2, fg_p3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3])
        xent_features += [l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3]
        triplet_features += [fg_p1, fg_p2, fg_p3]

        # our branch
        x2 = x
        x2 = self.layer4_abd(x2)
        x2 = self.reduction_tr(x2)

        feature_dict = {
            'cam': [],
            'pam': [],
            'before': [],
            'after': [],
            'layer5': layer5
        }

        margin = 24 // self.part_num

        for p in range(1, self.part_num + 1):

            f = x2[:, :, margin * (p - 1):margin * p, :]

            if not self.attention_config['parts']:
                f_after1 = f_before1 = self.before_module1(f)
                feature_dict['before'] = (*feature_dict['before'], f_before1)
                feature_dict['after'] = (*feature_dict['after'], f_after1)
            else:
                f_before1 = self.before_module1(f)
                f_cam1 = self.cam_module1(f)
                f_pam1 = self.pam_module1(f)

                f_sum1 = f_cam1 + f_pam1 + f_before1
                f_after1 = self.sum_conv1(f_sum1)

                feature_dict['cam'] = (*feature_dict['cam'], f_cam1)
                feature_dict['pam'] = (*feature_dict['pam'], f_pam1)
                feature_dict['before'] = (*feature_dict['before'], f_before1)
                feature_dict['after'] = (*feature_dict['after'], f_after1)

            v = self.global_avgpool(f_after1)
            v = v.view(v.size(0), -1)
            triplet_features.append(v)
            predict_features.append(v)
            v = getattr(self, 'classifier_p{}'.format(p))(v)
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

#
def make_function_sf_tricky_50(name, config, tricky, base_class):

    def _func(num_classes, loss, pretrained='imagenet', **kwargs):
        print(config)
        return base_class(
            num_classes=num_classes,
            last_stride=2,
            dropout_p=None,
            sum_fusion=True,
            tricky=tricky,
            **config,
            **kwargs
        )

    _func.config = config

    name_function_mapping[name] = _func
    globals()[name] = _func


from collections import OrderedDict

configurations = OrderedDict([
    (
        'fc_dims',
        [
            (None, '_nofc'),
            ([256], '_fc256'),
            ([512], '_fc512'),
            ([1024], '_fc1024'),
        ],
    ),
    (
        'fd_config',
        [
            (
                {
                    'parts': parts,
                    'use_conv_head': use_conv_head
                },
                '_fd_{}_{}'.format(parts_name, "head" if use_conv_head else "nohead")
            )
            for parts, parts_name in [
                [('ab', 'c'), 'ab_c'],
                [('ab',), 'ab'],
                [('a',), 'a'],
                [(), 'none'],
                [('abc',), 'abc']
            ]
            for use_conv_head in (True, False)
        ],
    ),
    (
        'attention_config',
        [
            (
                {
                    'parts': parts,
                    'use_conv_head': use_conv_head
                },
                '_dan_{}_{}'.format(parts_name, "head" if use_conv_head else "nohead")
            )
            for parts, parts_name in [
                [('cam', 'pam'), 'cam_pam'],
                [('cam',), 'cam'],
                [('pam',), 'pam'],
                [(), 'none'],
            ]
            for use_conv_head in (True, False)
        ]
    )
])

# configurations: 'Dict[str, List[Tuple[config, name_frag]]]'

import itertools

fragments = list(itertools.product(*configurations.values()))
keys = list(configurations.keys())

name_function_mapping = {}

for fragment in fragments:

    name = 'resnet50_mgn'
    config = {}
    for key, (sub_config, name_frag) in zip(keys, fragment):
        name += name_frag
        config.update({key: sub_config})

    make_function_sf_tricky_50(name, config, 12, base_class=ResNetABDMGN)
