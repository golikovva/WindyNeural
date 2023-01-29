import numpy as np
import torch
from imgaug import augmenters as iaa

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random


# from standartminmaxscaler import *


class StationNormalNoize(object):
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return x + torch.normal(self.mean, self.std, size=x.shape)


class GFSNormalNoize(torch.nn.Module):
    def __init__(self, mean=0, std=1, size=(10, 10)):
        super(GFSNormalNoize, self).__init__()
        self.mean = mean
        self.std = std
        self.size = size

    def __call__(self, x):
        if self.size is None:
            noize = torch.normal(self.mean, self.std, size=x.shape)
        else:
            noize = torch.normal(self.mean, self.std, size=x.shape[:-2] + self.size)
            noize = transforms.functional.resize(noize, x.shape[-2:],
                                                 interpolation=transforms.InterpolationMode.BICUBIC)
        return x + noize


def additive_gauss_noise(images, mean=0, var=0.01, mask_size=(10, 10, 3)):
    bs, ch, row, col = images.shape
    seq = iaa.Sequential([iaa.AdditiveGaussianNoise(mean, var, per_channel=True)])
    base = np.zeros((bs, 10, 10, 3)).astype('float32')
    gauss = seq(images=base)
    gauss = iaa.Resize(row, interpolation='cubic').augment(images=gauss)
    gauss = iaa.AverageBlur(k=7).augment(images=gauss)
    noisy = images + torch.from_numpy(np.moveaxis(gauss, -1, 1))
    return noisy


class MultiImageTransform(torch.nn.Module):
    """Transform stacked 1D images"""

    def __init__(self, transform, **kwargs):
        super(MultiImageTransform, self).__init__()
        self.transform = transform
        self.kwargs = kwargs

    def __call__(self, x):
        multi_x = torch.split(x, 1, dim=-3)
        new_x = []
        for channel in multi_x:
            new_x.append(self.transform(channel, **self.kwargs))
        new_x = torch.cat(new_x, dim=-3)
        return new_x


class GFSRandomPosterize(MultiImageTransform):
    def __init__(self, bits, p, all_tensor_apply=True):
        super(GFSRandomPosterize, self).__init__(TF.posterize, bits=bits)
        self.bits = bits
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            orig_type = x.type()
            x = (x * 255).type(torch.uint8)
            x = super(GFSRandomPosterize, self).__call__(x)
            x = x.type(orig_type) / 255
            x = TF.gaussian_blur(x, 3)
        return x


class ChannelRandomErasing(MultiImageTransform):
    def __init__(self, p=1, scale=(0.01, 0.07), ratio=(0.3, 3.3), value=0.37):
        super(ChannelRandomErasing, self).__init__(transforms.RandomErasing(p, scale, ratio, value))
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.randomerase = transforms.RandomErasing(p, scale, ratio, value)

    def __call__(self, x):
        x = super(ChannelRandomErasing, self).__call__(x)
        return x


class GFSAffine(torch.nn.Module):
    def __init__(self, p=0.2, degrees=3, scale=1.05, fill=0.3714,
                 interpolation=transforms.InterpolationMode.BILINEAR, translate=(0, 0), shear=0.):
        super(GFSAffine, self).__init__()
        self.p = p
        self.degrees = degrees
        self.scale = scale
        self.fill = fill
        self.interpolation = interpolation
        self.translate = translate
        self.shear = shear

    def __call__(self, x):
        if random.random() < self.p:
            degrees = random.random() * 2 * self.degrees - self.degrees
            x = TF.affine(x, degrees, scale=self.scale, fill=[self.fill],
                          interpolation=self.interpolation, translate=self.translate,
                          shear=[self.shear])
        return x
