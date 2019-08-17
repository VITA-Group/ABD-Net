from __future__ import absolute_import
from __future__ import division

from torchvision.transforms import *
import torchvision.transforms.functional as TF
import torch

from PIL import Image
import random
import numpy as np
import math


class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


class Random2DTranslation(object):
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.

    Args:
    - height (int): target image height.
    - width (int): target image width.
    - p (float): probability of performing this transformation. Default: 0.5.
    """

    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
        - img (PIL Image): Image to be cropped.
        """
        if random.uniform(0, 1) > self.p:
            return img.resize((self.width, self.height), self.interpolation)

        new_width, new_height = int(round(self.width * 1.125)), int(round(self.height * 1.125))
        resized_img = img.resize((new_width, new_height), self.interpolation)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img.crop((x1, y1, x1 + self.width, y1 + self.height))
        return croped_img


# def build_transforms(height, width, is_train, **kwargs):
#     """Build transforms

#     Args:
#     - height (int): target image height.
#     - width (int): target image width.
#     - is_train (bool): train or test phase.
#     """

#     # use imagenet mean and std as default
#     imagenet_mean = [0.485, 0.456, 0.406]
#     imagenet_std = [0.229, 0.224, 0.225]
#     normalize = Normalize(mean=imagenet_mean, std=imagenet_std)

#     transforms = []

#     if is_train:
#         transforms += [Random2DTranslation(height, width)]
#         transforms += [RandomHorizontalFlip()]
#     else:
#         transforms += [Resize((height, width))]

#     transforms += [ToTensor()]
#     transforms += [normalize]

#     transforms = Compose(transforms)

#     return transforms

def build_training_transforms(height, width, data_augment):

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    normalize = Normalize(mean=imagenet_mean, std=imagenet_std)

    data_augment = set(data_augment)
    print('Using augmentation:', data_augment)

    transforms = []
    if 'crop' in data_augment:
        transforms.append(Random2DTranslation(height, width))
    else:
        transforms.append(Resize((height, width)))

    transforms.append(RandomHorizontalFlip())

    if 'color-jitter' in data_augment:
        transforms.append(ColorJitter())

    transforms.append(ToTensor())
    transforms.append(normalize)

    if 'random-erase' in data_augment:
        transforms.append(RandomErasing())

    return transforms


def build_transforms(height, width, is_train, data_augment, **kwargs):
    """Build transforms

    Args:
    - height (int): target image height.
    - width (int): target image width.
    - is_train (bool): train or test phase.
    - data_augment (str)
    """

    # use imagenet mean and std as default
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    normalize = Normalize(mean=imagenet_mean, std=imagenet_std)

    transforms = []

    if is_train:
        transforms = build_training_transforms(height, width, data_augment)
    else:
        transforms += [Resize((height, width))]

        if kwargs.get('flip', False):
            transforms += [Lambda(lambda img: TF.hflip(img))]

        transforms += [ToTensor()]
        transforms += [normalize]

    transforms = Compose(transforms)
    if is_train:
        print('Using transform:', transforms)

    return transforms
