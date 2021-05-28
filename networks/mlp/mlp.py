import torch.nn as nn
import torch
import torch.nn.functional as F

"""
Multi-layer perceptron, following BYOL.
"""
class MLP(nn.Module):

    def __init__(self, input_size=512, hidden_size=1024, output_size=256):
        super(MLP, self).__init__()

        mlp = [
            nn.Linear(in_features=input_size, out_features=hidden_size, bias=True),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=output_size, bias=True)
        ]

        self.net = nn.Sequential(*mlp)

    def forward(self, x):
        return self.net(x)
