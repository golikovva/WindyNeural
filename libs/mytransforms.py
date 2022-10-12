import numpy as np
import torch
from imgaug import augmenters as iaa
from standartminmaxscaler import *


class StationNormalNoize(object):
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return x+torch.normal(self.mean, self.std, size=x.shape)


def additive_gauss_noise(images, mean=0, var=0.01, mask_size=(10, 10, 3)):
    bs, ch, row, col = images.shape
    seq = iaa.Sequential([iaa.AdditiveGaussianNoise(mean, var, per_channel=True)])
    base = np.zeros((bs, 10, 10, 3)).astype('float32')
    gauss = seq(images=base)
    gauss = iaa.Resize(row, interpolation='cubic').augment(images=gauss)
    gauss = iaa.AverageBlur(k=7).augment(images=gauss)
    noisy = images + torch.from_numpy(np.moveaxis(gauss, -1, 1))
    return noisy
