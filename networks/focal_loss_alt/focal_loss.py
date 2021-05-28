"""
Modified.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, alpha=None, gamma=2, reduction='mean', size_average=True, device=torch.device('cuda:0')):
        alpha = torch.tensor(alpha).to(device)
        super(FocalLoss, self).__init__(alpha, reduction=reduction)
        self.gamma = gamma
        self.alpha = alpha #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target, reduction=self.reduction, weight=self.alpha)
        pt = torch.exp(-ce_loss)

        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

def focal_loss_function(input, target, gamma, alpha, reduction='mean', xe_reduction='mean'):
    device = input.device
    if isinstance(alpha, (list)):
        alpha = torch.tensor(alpha).to(device)

    ce_loss = F.cross_entropy(input, target, reduction=xe_reduction, weight=alpha)
    pt = torch.exp(-ce_loss)

    focal_loss = ((1 - pt) ** gamma * ce_loss)
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    elif reduction == 'none':
        return focal_loss
