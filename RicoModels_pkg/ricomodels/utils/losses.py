#!/usr/env/bin python3

import torch
import torch.nn as nn
import torch.functional as F

def dice_loss(outputs, labels,  epsilon=1e-6):
    """
    outputs: (n, class_num, h, w)
    """
    # Ensure labels are in long (int64) type
    if labels.dtype != torch.int64:
        labels = labels.to(torch.int64)
    class_num = outputs.shape[1]

    # Apply softmax to outputs (assumed to be logits)
    outputs = torch.nn.functional.softmax(outputs, dim=1)

    # Convert labels to one-hot encoding
    labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=class_num)

    # Permute one-hot labels to match the dimensions of outputs [batch, classes, height, width]
    labels_one_hot = labels_one_hot.permute(0, 3, 1, 2).float()

    # print("labels_one_hot: ", 2 * (outputs * labels_one_hot))

    # Compute intersection and union
    intersect = 2 * (outputs * labels_one_hot).sum(dim=(2, 3))
    total = outputs.sum(dim=(2, 3)) + labels_one_hot.sum(dim=(2, 3))

    # Compute Dice coefficient
    dice = (intersect + epsilon) / (total + epsilon)
    # print("dice: ", dice)

    # Compute Dice loss as a mean of all one hot vectors across batches
    dice_loss = 1 - dice.mean()
    return dice_loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, ignore_index=None):
        super().__init__()
        self._smooth = smooth
    def forward(self, outputs, labels):
        return dice_loss(outputs, labels, self._smooth)
