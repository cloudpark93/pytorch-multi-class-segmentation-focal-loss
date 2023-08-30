import torch
from torch.nn import CrossEntropyLoss, Module


class FocalLoss(Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        #         focal_loss = self.alpha[targets] * (1 - pt)**self.gamma * ce_loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        focal_loss = focal_loss.mean()

        return focal_loss
