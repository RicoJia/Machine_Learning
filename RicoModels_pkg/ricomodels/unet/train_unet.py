#! /usr/bin/env python3
import argparse
import logging
import os

import torch
import wandb
from ricomodels.unet.unet import UNet
from ricomodels.utils.data_loading import (get_carvana_datasets,
                                           get_data_loader, get_gta5_datasets,
                                           get_package_dir,
                                           get_VOC_segmentation_datasets)
from ricomodels.utils.losses import DiceLoss, FocalLoss, dice_loss
from ricomodels.utils.training_tools import (EarlyStopping,
                                             check_model_image_channel_num,
                                             eval_model)
from ricomodels.utils.visualization import (TrainingTimer,
                                            get_total_weight_norm,
                                            wandb_weight_histogram_logging)
from torch import optim
from tqdm import tqdm

# Input args
USE_AMP = False
SAVE_CHECKPOINTS = False

# Configurable contants
BATCH_SIZE = 8
MODEL_PATH = os.path.join(get_package_dir(), "unet/unet_pascal.pth")
CHECKPOINT_DIR = os.path.join(get_package_dir(), "unet/checkpoints")
ACCUMULATION_STEPS = int(32 / BATCH_SIZE)
NUM_EPOCHS = 70
LEARNING_RATE = 1e-5
SAVE_EVERY_N_EPOCH = 5
INTERMEDIATE_BEFORE_MAX_POOL = False
WEIGHT_DECAY = 1e-8
MOMENTUM = 0.999


# Check against example
def train_model(
    model,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    num_training,
    wandb_logger,
    NUM_EPOCHS,
    device="cpu",
):
    early_stopping = EarlyStopping(delta=1e-5, patience=3)
    timer = TrainingTimer()
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
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
            if SAVE_CHECKPOINTS:
                if not os.path.exists(CHECKPOINT_DIR):
                    os.mkdir(CHECKPOINT_DIR)
                file_name = f"unet_epoch_{epoch}.pth"
                torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, file_name))
                logging.info(f"Saved model {file_name}")
            else:
                torch.save(model.state_dict(), MODEL_PATH)
                logging.info(f"Saved model {MODEL_PATH}")

        if early_stopping(epoch_loss):
            logging.info("Early stopping triggered")
            break
    logging.info("Training complete")
    return epoch


def parse_args():
    """
    Parse args and set global input args
    """
    global USE_AMP, SAVE_CHECKPOINTS
    parser = argparse.ArgumentParser(description="Set training options")
    parser.add_argument(
        "--use_amp",
        "-u",
        action="store_true",
        help="Enable automatic mixed precision (AMP)",
    )
    parser.add_argument(
        "--save_checkpoints",
        "-s",
        action="store_true",
        help="Enable saving model checkpoints",
    )
    args = parser.parse_args()
    USE_AMP = args.use_amp
    SAVE_CHECKPOINTS = args.save_checkpoints


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parse_args()
    train_dataset, val_dataset, test_dataset, class_num = (
        get_VOC_segmentation_datasets()
    )
    # train_dataset, val_dataset, test_dataset, class_num = get_carvana_datasets()
    # train_dataset, val_dataset, test_dataset, class_num = get_gta5_datasets()
    train_dataloader, val_dataloader, test_dataloader = get_data_loader(
        train_dataset, val_dataset, test_dataset, batch_size=BATCH_SIZE
    )
    print(
        f"Lengths of train_dataset, val_dataset, test_dataset: {len(train_dataset), len(val_dataset), len(test_dataset)}"
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = FocalLoss()

    model = UNet(
        class_num=class_num, intermediate_before_max_pool=INTERMEDIATE_BEFORE_MAX_POOL
    )
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(
            torch.load(MODEL_PATH, weights_only=False, map_location=device)
        )
        print("Loaded model")
    else:
        print("Initialized model")

    model.to(device)
    optimizer = optim.RMSprop(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        momentum=MOMENTUM,
        foreach=True,
    )
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=2
    )  # goal: minimize Dice score

    wandb_logger = wandb.init(project="Rico-U-Net", resume="allow", anonymous="must")
    wandb_logger.config.update(
        dict(
            epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE * ACCUMULATION_STEPS,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            training_size=len(train_dataset),
            intermediate_before_max_pool=INTERMEDIATE_BEFORE_MAX_POOL,
            save_checkpoint=SAVE_CHECKPOINTS,
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
        Intermediate_before_max_pool : {INTERMEDIATE_BEFORE_MAX_POOL}
        Save checkpoints:     {SAVE_CHECKPOINTS}
        Device:          {device.type}
        Mixed Precision: {USE_AMP},
        Optimizer:       {str(optimizer)}
    """
    )
    try:
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
    except torch.cuda.OutOfMemoryError:
        logging.error(
            "Detected OutOfMemoryError! "
            "Enabling checkpointing to reduce memory usage, but this slows down training. "
            "Consider enabling USE_AMP for fast and memory efficient training"
        )
        model.use_checkpointing()
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

    train_acc, val_acc, test_acc = eval_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        device=device,
        class_num=class_num,
        visualize=True,
    )
    # , val_dataloader, test_dataloader, device, class_num, visualize: bool = False)
    wandb_logger.log(
        {
            "Stopped at epoch": epoch,
            "train accuracy: ": train_acc,
            "val accuracy: ": val_acc,
            "test accuracy: ": test_acc,
        }
    )

    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()
