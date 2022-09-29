import numpy as np
import torch
from imgaug import augmenters as iaa


def additive_gauss_noise(images, mean=0, var=0.01, mask_size=(10, 10, 3)):
    bs, ch, row, col = images.shape
    seq = iaa.Sequential([iaa.AdditiveGaussianNoise(mean, var, per_channel=True)])
    base = np.zeros((bs, 10, 10, 3)).astype('float32')
    gauss = seq(images=base)
    gauss = iaa.Resize(row, interpolation='cubic').augment(images=gauss)
    gauss = iaa.AverageBlur(k=7).augment(images=gauss)
    noisy = images + torch.from_numpy(np.moveaxis(gauss, -1, 1))
    return noisy
