import torch.nn as nn
import torch.nn.functional as F

"""
Projection architecture from: https://arxiv.org/pdf/2012.06985.pdf
"""
class ProjectionHead(nn.Module):

    def __init__(self, n_classes, n_input_channels, n_hidden_dims=256):
        super(ProjectionHead, self).__init__()
        conv1 = nn.Conv2d(n_input_channels, n_hidden_dims, kernel_size=1)
        relu = nn.ReLU()
        conv2 = nn.Conv2d(n_hidden_dims, n_hidden_dims, kernel_size=1)
        conv3 = nn.Conv2d(n_hidden_dims, n_classes, kernel_size=1)

        sequence = [
            conv1,
            relu,
            conv2,
            relu,
            conv3
        ]

        self.net = nn.Sequential(*sequence)

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1)
