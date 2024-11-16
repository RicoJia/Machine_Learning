#!/usr/bin/env python3
import argparse
import logging
import os

import torch
import wandb
from ricomodels.unet.unet import UNet
from ricomodels.utils.data_loading import (
    get_data_loader,
    get_package_dir,
    get_coco_classification_datasets,
)
from ricomodels.utils.losses import DiceLoss, FocalLoss, dice_loss
from ricomodels.utils.training_tools import (
    EarlyStopping,
    check_model_image_channel_num,
    eval_model,
)
from ricomodels.utils.visualization import (
    TrainingTimer,
    get_total_weight_norm,
    wandb_weight_histogram_logging,
)
from torch import optim
from tqdm import tqdm
# Input args
USE_AMP = False
SAVE_CHECKPOINTS = False

# Configurable contants
BATCH_SIZE = 2
MODEL_PATH = os.path.join(get_package_dir(), "unet/unet_pascal.pth")
CHECKPOINT_DIR = os.path.join(get_package_dir(), "unet/checkpoints")
ACCUMULATION_STEPS = int(32 / BATCH_SIZE)
NUM_EPOCHS = 70
LEARNING_RATE = 1e-5
SAVE_EVERY_N_EPOCH = 5
INTERMEDIATE_BEFORE_MAX_POOL = False
WEIGHT_DECAY = 1e-8
MOMENTUM = 0.999

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    train_dataset, val_dataset, _, class_num = (
        get_coco_classification_datasets()
    )
    train_dataloader, val_dataloader, _ = get_data_loader(
        train_dataset, val_dataset, None, batch_size=BATCH_SIZE
    )
    print(
        f"Lengths of train_dataset, val_dataset: {len(train_dataset), len(val_dataset)}"
    )
