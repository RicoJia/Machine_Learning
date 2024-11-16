#!/usr/bin/env python3
import argparse
import logging
import os

import torch
import wandb
from ricomodels.unet.unet import UNet
from ricomodels.utils.data_loading import (
    get_carvana_datasets,
    get_data_loader,
    get_gta5_datasets,
    get_package_dir,
    get_VOC_segmentation_datasets,
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
    train_dataset, val_dataset, test_dataset, class_num = (
        get_VOC_segmentation_datasets()
    )
    # # train_dataset, val_dataset, test_dataset, class_num = get_carvana_datasets()
    # # train_dataset, val_dataset, test_dataset, class_num = get_gta5_datasets()
    # train_dataloader, val_dataloader, test_dataloader = get_data_loader(
    #     train_dataset, val_dataset, test_dataset, batch_size=BATCH_SIZE
    # )
    # print(
    #     f"Lengths of train_dataset, val_dataset, test_dataset: {len(train_dataset), len(val_dataset), len(test_dataset)}"
    # )
