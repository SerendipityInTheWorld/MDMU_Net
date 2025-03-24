import torch
import torch.nn as nn
from monai.losses import DiceLoss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
class CombinedLoss(nn.Module):
    def __init__(self, alpha_focal=0.25, gamma_focal=2, weight_focal=0.5):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha_focal, gamma=gamma_focal)
        self.ce_loss = nn.CrossEntropyLoss()
        self.weight_focal = weight_focal

    def forward(self, inputs, targets):
        loss_focal = self.focal_loss(inputs, targets)
        loss_ce = self.ce_loss(inputs, targets)
        return self.weight_focal * loss_focal + (1 - self.weight_focal) * loss_ce