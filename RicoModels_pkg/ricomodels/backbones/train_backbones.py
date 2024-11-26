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
from ricomodels.utils.losses import FocalLoss, MULTICLASS_CLASSIFICATION_THRE
from ricomodels.utils.training_tools import (
    EarlyStopping,
    check_model_image_channel_num,
    eval_model,
    load_model,
)
from ricomodels.utils.visualization import (
    TrainingTimer,
    get_total_weight_norm,
    wandb_weight_histogram_logging,
)
from torch import optim
from tqdm import tqdm

from ricomodels.backbones.mobilenetv2.mobilenet_v2 import MobileNetV2
import torchsummary

# Input args
USE_AMP = True

# Configurable contants
BATCH_SIZE = 4
MODEL_PATH = os.path.join(get_package_dir(), "backbones/mobilenetv2/mobilenetv2.pth")
ACCUMULATION_STEPS = int(32 / BATCH_SIZE)
NUM_EPOCHS = 70
LEARNING_RATE = 1e-3
SAVE_EVERY_N_EPOCH = 2
WEIGHT_DECAY = 1e-8
MOMENTUM = 0.999

def init_logging():
    wandb_logger = wandb.init(project="Rico-mobilenetv2", resume="allow", anonymous="must")
    wandb_logger.config.update(
        dict(
            epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE * ACCUMULATION_STEPS,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            training_size=len(train_dataset),
            amp=USE_AMP,
            optimizer=str(optimizer),
        )
    )
    logging.info(
        f"""Starting training:
        Epochs:          {NUM_EPOCHS}
        Batch size:      {BATCH_SIZE}
        Learning rate:   {LEARNING_RATE}
        Weight decay:    {WEIGHT_DECAY}
        Training size:   {len(train_dataset)}
        Device:          {device.type}
        Mixed Precision: {USE_AMP},
        Optimizer:       {str(optimizer)}
    """
    )
    return wandb_logger

def train_model(
    model,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    num_training,
    wandb_logger,
    NUM_EPOCHS,
    device,
):
    early_stopping = EarlyStopping(delta=1e-3, patience=1)
    timer = TrainingTimer()
    device_type = str(device)
    scaler = torch.amp.GradScaler(device = device_type, enabled=USE_AMP)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        with tqdm(
            total=num_training, desc=f"Epoch [{epoch }/{NUM_EPOCHS}]", unit="img"
        ) as pbar:
            for i, (inputs, labels) in enumerate(train_loader):
                check_model_image_channel_num(
                    img_channels=inputs.shape[1], model_channels=model.n_channels
                )
                # outside autocast because labels (long) could have issues with fp16 casting
                inputs = inputs.to(device)
                labels = labels.to(device)
                with torch.autocast(
                    device_type=device_type, dtype=torch.float16, enabled=USE_AMP
                ):
                    # this should be torch.float16 if USE_AMP
                    outputs = model(inputs)
                    # loss is autocast to torch.float32
                    loss = criterion(outputs, labels) / ACCUMULATION_STEPS
                pbar.update(inputs.shape[0])
                pbar.set_postfix(**{"loss (batch)": loss.item()})
                # exits autocast before backward()
                # create scaled gradients
                scaler.scale(loss).backward()
                # loss.backward()
                if (i + 1) % ACCUMULATION_STEPS == 0:
                    # optimizer.step()
                    # First, gradients of optimizer params are unscaled here. Unless nan or inf shows up, optimizer.step() is called
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                epoch_loss += loss.item()
            epoch_loss /= num_training
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(metrics=epoch_loss)
            total_weight_norm = get_total_weight_norm(model)
            wandb_weight_histogram_logging(model, epoch)
            wandb_logger.log(
                {
                    "epoch loss": epoch_loss,
                    "epoch": epoch,
                    "learning rate": current_lr,
                    "total_weight_norm": total_weight_norm,
                    "elapsed_time": timer.lapse_time(),
                }
            )

        if epoch % SAVE_EVERY_N_EPOCH == 0:
            torch.save(model.state_dict(), MODEL_PATH)
            logging.info(f"Saved model {MODEL_PATH}")
        if early_stopping(epoch_loss):
            logging.info("Early stopping triggered")
            break
    logging.info("Training complete")
    return epoch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", "-e", action="store_true", default=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    train_dataset, val_dataset, _, class_num = (
        get_coco_classification_datasets()
    )
    train_dataloader, val_dataloader, _ = get_data_loader(
        train_dataset, val_dataset, None, batch_size=BATCH_SIZE
    )
    print(
        f"Lengths of train_dataset, {len(train_dataset)}, val_dataset:  {len(val_dataset)}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = FocalLoss(use_focal_binary_multi_class=True) # torch.nn.BCEWithLogitsLoss()

    model = MobileNetV2(num_classes= class_num, output_stride=4)
    load_model(model_path=MODEL_PATH, model=model, device=device)
    model.to(device)
    # input size is from the dataloader. It could change
    torchsummary.summary(model, input_size=(3, 256, 256), device=str(device))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=1
    )  # goal: minimize focal loss
    wandb_logger = init_logging()
    if not args.eval:
        epoch = train_model(
            model=model,
            train_loader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            NUM_EPOCHS=NUM_EPOCHS,
            device=device,
            num_training=len(train_dataset),
            wandb_logger=wandb_logger,
        )
    train_dataset.set_effective_length_if_necessary(stop_at=10)
    multiclass_thre = MULTICLASS_CLASSIFICATION_THRE

    eval_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=None,
        device=device,
        class_num=class_num,
        task_mode=train_dataset.task_mode,
        class_names=train_dataset.class_names,
        multiclass_thre=MULTICLASS_CLASSIFICATION_THRE,
        visualize=True,
    )

    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()
