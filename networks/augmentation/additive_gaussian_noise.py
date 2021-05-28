import torch
import torch.nn.functional as F
import random

class AdditiveGaussianNoiseTransform(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        normal_noise = torch.randn(x.shape).to(x.device)
        return x + normal_noise * self.std + self.mean

class AdditiveGaussianNoise:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        normal_noise = torch.randn(x.shape).to(x.device)
        return x + normal_noise * self.std + self.mean

class Sobel:
    def __init__(self, device, n_channels=1, p=0.2):
        sobel = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]])
        kernel = (sobel.T).view(1,1,3,3).repeat(1,n_channels, 1,1)*1.0
        self.kernel = kernel.to(device)
        self.p = p
    def __call__(self, x):
        if random.random() < self.p:
            return F.conv2d(x, self.kernel, padding=1)
        else:
            return x
