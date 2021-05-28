import torch.nn as nn
import torch
import torch.nn.functional as F
from .wideresnet import build_wideresnet

class FixMatch(nn.Module):

    # TODO: No dropout implemented
    def __init__(self, n_classes=10, widen_factor=2, depth=28, dropout=0):
        super(FixMatch, self).__init__()
        self.model = build_wideresnet(num_classes=n_classes, widen_factor=widen_factor, depth=depth, dropout=dropout)
        self.first = True

        self.softmax = nn.Softmax()
        self.log_softmax = nn.LogSoftmax()

    """

    Args:
        - x     Input (batch_size, channels, width, height)
        - tau     Pseudo-label threshold value [0,1]
    """
    def forward(self, x):
        return self.model(x)

    def calc_ema(self, previous_mean, incoming_value, alpha=0.99):
        return alpha * incoming_value + (1 - alpha) * previous_mean

    def ema(self, model, alpha):
        for ema_p, online_p in zip(self.parameters(), model.parameters()):
            ema_p.data = self.calc_ema(ema_p.data, online_p.data, alpha=alpha)

    def predict(self, x):
        return self.softmax(self.model(x))

    def optimize(self, forward_output, tau):
        weak_path_output, strong_path_output = forward_output
        weak_path_output = F.softmax(weak_path_output, dim=-1)

        max_values, max_indices = torch.max(weak_path_output, dim=-1)
        pseudo_nonzero_mask = max_values.ge(tau).float()
        loss = F.cross_entropy(strong_path_output, max_indices, reduction='none')
        loss = loss * pseudo_nonzero_mask
        loss = loss.mean()
        return loss
