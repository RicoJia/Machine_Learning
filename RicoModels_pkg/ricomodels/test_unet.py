#! /usr/bin/env python3

from numpy import ndim
import os
from utils.losses import dice_loss, DiceLoss
import torch
import torchvision
import torch.nn.functional as F
import time

###############################################################
# Model Training
###############################################################





###############################################################
# Model Evaluation
###############################################################

def calculate_average_weights(model):
    total_sum = 0
    total_elements = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            weight_mean = param.mean().item()
            total_sum += param.sum().item()
            total_elements += param.numel()
            print(f"Layer: {name} | Average Weight: {weight_mean:.6f}")

    overall_average = total_sum / total_elements if total_elements > 0 else 0
    print(f"Overall Average Weight in the Network: {overall_average:.6f}")

calculate_average_weights(model)

eval_model(model, test_dataloader, device=device, visualize=True)
