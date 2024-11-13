#!/usr/env/bin python3

import torch
import torch.functional as F
import torch.nn as nn

# TODO


def focal_loss(outputs, targets, gamma):
    """
    outputs: (n, class_num, h, w), labels (n, h, w)
    """
    torch.set_printoptions(profile="full")
    probs = torch.nn.functional.softmax(outputs, dim=1)

    # (n, h, w)
    p_true_class = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    log_p_true_class = torch.log(p_true_class + 1e-8)
    fl = -((1 - p_true_class) ** gamma) * (log_p_true_class)
    return fl.mean()


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma=8,
    ):
        super().__init__()
        self._gamma = gamma

    def forward(self, outputs, labels):
        return focal_loss(outputs, labels, self._gamma)


def dice_loss(outputs, labels, epsilon=1e-6):
    """
    outputs: (n, class_num, h, w), labels (n, h, w)
    """
    # Ensure labels are in long (int64) type
    if labels.dtype != torch.int64:
        labels = labels.to(torch.int64)
    class_num = outputs.shape[1]
    outputs = torch.nn.functional.softmax(outputs, dim=1)
    labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=class_num)
    # Permute one-hot labels to match the dimensions of outputs [batch, classes, height, width]
    labels_one_hot = labels_one_hot.permute(0, 3, 1, 2).float()

    intersect = 2 * (outputs * labels_one_hot).sum(dim=(2, 3))
    total = outputs.sum(dim=(2, 3)) + labels_one_hot.sum(dim=(2, 3))
    dice = (intersect + epsilon) / (total + epsilon)
    dice_loss = 1 - dice.mean()
    return dice_loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, ignore_index=None):
        super().__init__()
        self._smooth = smooth

    def forward(self, outputs, labels):
        return dice_loss(outputs, labels, self._smooth)
