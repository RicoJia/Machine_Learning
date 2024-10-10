#!/usr/env/bin python3

import torch
import torch.nn as nn
import torch.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, ignore_index=None):
        super().__init__()
        self._smooth = smooth
    def forward(self, predictions, targets):
        num_classes = predictions.size(1)
        predictions = F.softmax(predictions, dim=1)

        # M, H, W, N
        targets_1_hot = F.one_hot(targets, num_classes=num_classes)
        # M, N, H, W
        targets_1_hot = targets_1_hot.permute(0,3,1,2).float()
        # TODO?
        predictions_flat = predictions.view(predictions.size(0), predictions.size(1), -1)  #M, N, H*W
        targets_flat = targets_1_hot.view(targets_1_hot.size(0), targets_1_hot.size(1), -1)  # (M, N, H*W)

        intersection = (predictions_flat * targets_flat).sum(dim=2)  # (M, N)
        denominator = predictions_flat.sum(dim=2) + targets_flat.sum(dim=2)  # (M, N)

        dice_score = (2. * intersection + self._smooth) / (denominator + self._smooth)  # (M, N)

        dice_loss_per_class = 1 - dice_score.mean(dim=0)  # (N,)
        dice_loss = dice_loss_per_class.mean()

        return dice_loss


def dice_loss(outputs, labels,  epsilon=1e-6):
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

    print("labels_one_hot: ", 2 * (outputs * labels_one_hot))

    # Compute intersection and union
    intersect = 2 * (outputs * labels_one_hot).sum(dim=(2, 3))
    total = outputs.sum(dim=(2, 3)) + labels_one_hot.sum(dim=(2, 3))

    # Compute Dice coefficient
    dice = (intersect + epsilon) / (total + epsilon)
    print("dice: ", dice)

    # Compute Dice loss as a mean of all one hot vectors across batches
    dice_loss = 1 - dice.mean()
    return dice_loss

