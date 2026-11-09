"""DiceCE loss = w_d * Dice + w_c * CrossEntropy (Dice ignores background by default)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, ce_weight=0.5, eps=1e-5, include_background=False):
        super().__init__()
        self.wd, self.wc, self.eps, self.bg = dice_weight, ce_weight, eps, include_background

    def forward(self, logits, target):
        """logits: [B,C,D,H,W]; target: [B,D,H,W] integer labels."""
        ce = F.cross_entropy(logits, target.long())
        prob = torch.softmax(logits, dim=1)
        onehot = F.one_hot(target.long(), logits.shape[1]).permute(0, 4, 1, 2, 3).float()
        dims = (0, 2, 3, 4)
        inter = (prob * onehot).sum(dims)
        dice = (2 * inter + self.eps) / (prob.sum(dims) + onehot.sum(dims) + self.eps)
        dice = dice if self.bg else dice[1:]
        return self.wd * (1 - dice.mean()) + self.wc * ce
