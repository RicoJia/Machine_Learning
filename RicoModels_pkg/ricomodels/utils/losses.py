#!/usr/env/bin python3

import torch
import torch.functional as F
import torch.nn as nn
import logging

MULTICLASS_CLASSIFICATION_THRE = 0.5


def focal_binary_multi_class(logits, targets, gamma=2):
    """In a multiclass classification problem, each element in the output vector
    is a binary-classification problem

    Args:
        logits (tensor.Torch): raw logits before sigmoid (note, NOT softmax)
        targets (tensor.Torch): targets
        gamma (int, optional): decay parameter

    Returns:
        focal loss
    """
    l = logits.reshape(-1)
    t = targets.reshape(-1)
    p = torch.sigmoid(l)
    p = torch.where(t >= MULTICLASS_CLASSIFICATION_THRE, p, 1 - p)
    logp = -torch.log(torch.clamp(p, 1e-4, 1 - 1e-4))
    loss = logp * ((1 - p) ** gamma)
    loss = loss.sum()
    return loss


def focal_loss(outputs, targets, gamma):
    """
    outputs: (n, class_num, h, w), targets (n, h, w)
    """
    torch.set_printoptions(profile="full")
    probs = torch.nn.functional.softmax(outputs, dim=1)

    # (n, h, w)
    # `tensor.gather(dim, indices)` here selects the values at the locations
    # indicated in targets. `targets` cleverly stores indices of one-hot vector
    # as class labels.
    p_true_class = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    log_p_true_class = torch.log(p_true_class + 1e-8)
    fl = -((1 - p_true_class) ** gamma) * (log_p_true_class)
    return fl.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, use_focal_binary_multi_class=False):
        super().__init__()
        self._gamma = gamma
        self.loss_func = (
            focal_binary_multi_class if use_focal_binary_multi_class else focal_loss
        )

    def forward(self, outputs, labels):
        return self.loss_func(outputs, labels, self._gamma)


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


class AccuracyCounter:
    """General purpose accuracy counter that could do correct/total"""

    def __init__(self, device):
        self.correct = torch.tensor(0.0, device=device)
        self.total = torch.tensor(0.0, device=device)

    def update(self, epoch_correct, epoch_total):
        # Not doing item() here because that's an implicit synchronization call
        # .cpu(), .numpy() have synchronization calls, too
        self.total += epoch_total
        self.correct += epoch_correct

    def get_result(self):
        accuracy = 100.0 * self.correct.item() / self.total.item()
        return accuracy

    def print_result(self):
        accuracy = self.get_result()
        logging.info(
            f"""
            Accuracy: {accuracy}
            """
        )


class F1ScoreCounter:
    def __init__(self, device):
        self.precision_counter = AccuracyCounter(device=device)
        self.recall_counter = AccuracyCounter(device=device)

    def update(self, true_positives, actual_positives, pred_positives):
        self.recall_counter.update(
            epoch_correct=true_positives, epoch_total=actual_positives
        )
        self.precision_counter.update(
            epoch_correct=true_positives, epoch_total=pred_positives
        )

    def get_result(self):
        recall = self.recall_counter.get_result()
        precision = self.precision_counter.get_result()
        f1 = 2 * recall * precision / (recall + precision)
        return f1, precision, recall

    def print_result(self):
        f1, precision, recall = self.get_result()
        logging.info(
            f"""
            F1 score: {f1}, precision: {precision}, recall: {recall}
            """
        )
