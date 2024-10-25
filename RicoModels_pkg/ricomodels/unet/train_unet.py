#! /usr/bin/env python3
import torch
import torch.nn as nn
from ricomodels.utils.losses import dice_loss, DiceLoss, FocalLoss
from ricomodels.utils.data_loading import get_package_dir, get_data_loader, get_gta5_datasets, get_carvana_datasets
from ricomodels.utils.training_tools import check_model_image_channel_num, EarlyStopping, eval_model
from ricomodels.unet.unet import UNet
from ricomodels.utils.visualization import get_total_weight_norm, wandb_weight_histogram_logging, TrainingTimer
import time
import os
from torch import optim
from tqdm import tqdm
import wandb
import logging

BATCH_SIZE = 8
MODEL_PATH = os.path.join(get_package_dir(), "unet/unet_pascal.pth")
CHECKPOINT_DIR = os.path.join(get_package_dir(), "unet/checkpoints")
ACCUMULATION_STEPS = int(32/BATCH_SIZE)
NUM_EPOCHS=70
LEARNING_RATE=1e-5
SAVE_CHECKPOINT = True
SAVE_EVERY_N_EPOCH = 5
AMP = False
INTERMEDIATE_BEFORE_MAX_POOL = False
WEIGHT_DECAY = 1e-8
MOMENTUM = 0.999

# Check against example
def train_model(model, train_loader, 
                criterion, optimizer, scheduler, 
                num_training, wandb_logger, NUM_EPOCHS, device='cpu'):
    early_stopping = EarlyStopping()
    timer = TrainingTimer()
    for epoch in range(1, NUM_EPOCHS+1):
        model.train()
        epoch_loss = 0.0
        
        with tqdm(total = num_training, desc=f"Epoch [{epoch }/{NUM_EPOCHS}]", unit='img') as pbar:
            for i, (inputs, labels) in enumerate(train_loader):
                check_model_image_channel_num(img_channels=inputs.shape[1], model_channels=model.n_channels)
                inputs = inputs.to(device)
                labels = labels.to(device)
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels) /ACCUMULATION_STEPS
                pbar.update(inputs.shape[0])
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                loss.backward()
                if (i+1)%ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                epoch_loss += loss.item() 

            epoch_loss /= num_training
            
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step(metrics=epoch_loss)
            total_weight_norm = get_total_weight_norm(model)
            wandb_weight_histogram_logging(model, epoch)
            wandb_logger.log({
                'epoch loss': epoch_loss,
                'epoch': epoch,
                'learning rate': current_lr,
                'total_weight_norm': total_weight_norm,
                'elapsed_time': timer.lapse_time(),
            })

        if SAVE_CHECKPOINT and epoch % SAVE_EVERY_N_EPOCH==0:
            if not os.path.exists(CHECKPOINT_DIR):
                os.mkdir(CHECKPOINT_DIR)
            file_name = f"unet_epoch_{epoch}.pth"
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, file_name))
            print(f"Saved model {file_name}")
        
        if early_stopping(epoch_loss):
            break
    print('Training complete')
    return epoch

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    train_dataset, val_dataset, test_dataset, class_num = get_carvana_datasets()
    # train_dataset, val_dataset, test_dataset, class_num = get_gta5_datasets()
    train_dataloader, val_dataloader, test_dataloader = get_data_loader(train_dataset, val_dataset, test_dataset, batch_size=BATCH_SIZE)
    print(f"Lengths of train_dataset, val_dataset, test_dataset: {len(train_dataset), len(val_dataset), len(test_dataset)}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # TODO: let's see this imbalance dataset
    criterion = FocalLoss()

    model = UNet(class_num = class_num, 
                    intermediate_before_max_pool=INTERMEDIATE_BEFORE_MAX_POOL)
    # TODO
    print("norm: ", get_total_weight_norm(model))
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=False, map_location=device))
        print("Loaded model")
    else:
        print("Initialized model")

    model.to(device)
    optimizer = optim.RMSprop(model.parameters(),
                              lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM,
                              foreach=True)
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)  # goal: minimize Dice score

    wandb_logger = wandb.init(project='Rico-U-Net', resume='allow', anonymous='must')
    wandb_logger.config.update(
        dict(epochs=NUM_EPOCHS, batch_size=BATCH_SIZE * ACCUMULATION_STEPS, learning_rate=LEARNING_RATE,
             weight_decay = WEIGHT_DECAY,
             training_size = len(train_dataset),
             intermediate_before_max_pool = INTERMEDIATE_BEFORE_MAX_POOL,
             save_checkpoint=SAVE_CHECKPOINT, amp=AMP, 
             optimizer = str(optimizer)
             )
    )
    logging.info(f'''Starting training:
        Epochs:          {NUM_EPOCHS}
        Batch size:      {BATCH_SIZE}
        Learning rate:   {LEARNING_RATE}
        Weight decay:    {WEIGHT_DECAY}
        Training size:   {len(train_dataset)}
        Intermediate_before_max_pool : {INTERMEDIATE_BEFORE_MAX_POOL}
        Checkpoints:     {SAVE_CHECKPOINT}
        Device:          {device.type}
        Mixed Precision: {AMP},
        Optimizer:       {str(optimizer)}
    ''')
    try:
        epoch = train_model(model = model, train_loader = train_dataloader, 
                    criterion = criterion, 
                    optimizer = optimizer, scheduler = scheduler,
                            NUM_EPOCHS=NUM_EPOCHS, device=device, num_training = len(train_dataset),
                            wandb_logger=wandb_logger)
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        model.use_checkpointing()
        epoch = train_model(model = model, train_loader = train_dataloader, 
                    criterion = criterion, 
                    optimizer = optimizer, scheduler = scheduler,
                            NUM_EPOCHS=NUM_EPOCHS, device=device, num_training = len(train_dataset),
                            wandb_logger=wandb_logger)

    train_acc, val_acc, test_acc = eval_model(
        model=model, 
        train_dataloader=train_dataloader, val_dataloader=val_dataloader, test_dataloader=test_dataloader,
        device = device,
        class_num = class_num, visualize = True
    )
    # , val_dataloader, test_dataloader, device, class_num, visualize: bool = False) 
    wandb_logger.log({
        'Stopped at epoch': epoch,
        'train accuracy: ': train_acc,
        'val accuracy: ': val_acc,
        'test accuracy: ': test_acc,
    })

    # [optional] finish the wandb run, necessary in notebooks                                                                      
    wandb.finish()
