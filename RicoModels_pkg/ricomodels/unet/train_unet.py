#! /usr/bin/env python3
import torch
import torch.nn as nn
from ricomodels.utils.losses import dice_loss, DiceLoss
from ricomodels.utils.data_loading import VOCSegmentationClass, get_pkg_dir
from ricomodels.unet.unet import UNet
import time
import os
from torch import optim
from tqdm import tqdm
import wandb
import logging
import functools

BATCH_SIZE = 8
MODEL_PATH = os.path.join(get_pkg_dir(), "unet/unet_pascal.pth")
CHECKPOINT_DIR = os.path.join(get_pkg_dir(), "unet/checkpoints")
ACCUMULATION_STEPS = 32/BATCH_SIZE
NUM_EPOCHS=70
LEARNING_RATE=1e-5
SAVE_CHECKPOINT = True
SAVE_EVERY_N_EPOCH = 10
AMP = False
INTERMEDIATE_BEFORE_MAX_POOL = False
WEIGHT_DECAY = 1e-8
MOMENTUM = 0.999

@functools.cache
def _check_channel(model_channels, img_channels):
    if model_channels != img_channels:
        raise ValueError(
            f'Network has been defined with {model_channels} input channels, ' \
            f'but loaded images have {img_channels} channels. Please check that ' \
            'the images are loaded correctly.'
        )

def eval_model(model, test_loader, device, visualize: bool = False):
    # Evaluation phase
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        # TODO I AM ITERATING OVER TRAIN_LOADER, SO I'M MORE SURE
        for inputs_test, labels_test in test_loader:
            inputs_test = inputs_test.to(device)
            labels_test = labels_test.to(device)
            outputs_test = model(inputs_test)
            _, predicted_test = outputs_test.max(1)
            mask = (labels_test != 21)
            local_total = mask.sum().item()
            local_correct = (predicted_test.eq(labels_test) & mask).sum().item()
            total_test += local_total
            correct_test += local_correct

            if visualize:
                #TODO Remember to remove
                print(f'Rico: predicted test acc {100. * local_correct/local_total}%')
                for img, pred, lab in zip(inputs_test, predicted_test, labels_test):
                    print("pred uniq: ", torch.unique(pred), "lab uniq: ", torch.unique(lab))
                    visualize_image_and_target(img.cpu(), pred.cpu(), lab.cpu())

    test_acc = 100. * correct_test / total_test
    print(f'Test Acc: {test_acc:.2f}%')

def validate_model(model, val_loader, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            #make sure data is on CPU/GPU
            inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to the correct device
            outputs = model(inputs)
            val_loss += dice_loss(outputs=outputs, labels = labels).item()
    val_loss /= len(val_loader)
    return val_loss

# Check against example
def train_model(model, train_loader, val_loader, 
                criterion, optimizer, scheduler, 
                num_training, wandb_logger, NUM_EPOCHS, device='cpu'):
    for epoch in range(1, NUM_EPOCHS+1):
        # Training phase
        start = time.time()
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        with tqdm(total = num_training, desc=f"Epoch [{epoch }/{NUM_EPOCHS}]", unit='img') as pbar:
            for i, (inputs, labels) in enumerate(train_loader):
                _check_channel(img_channels=inputs.shape[1], model_channels=model.n_channels)
                inputs = inputs.to(device)
                labels = labels.to(device)
                # Forward pass
                # output: (m, output_channel, h, w)
                outputs = model(inputs)
                # print(f"output: {outputs.dtype}, labels: {labels.dtype}")
                # This is because torch.nn.CrossEntropyLoss(reduction='mean') is true, so to simulate a larger batch, we need to further divide
                loss = (criterion(outputs, labels) +\
                    dice_loss(outputs=outputs, labels=labels.float(), ))/ACCUMULATION_STEPS
                pbar.update(inputs.shape[0])
                pbar.set_postfix(**{'loss (batch)': loss.item()})

            # Backward pass and optimization
            loss.backward()
            if (i+1)%ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
    #         # Statistics
    #         running_loss += loss.item() * inputs.size(0)
    #         _, predicted = outputs.max(1)
    #         # print(predicted.shape)
    #         total_train += mask.sum().item()
    #         # print((predicted == labels).sum().item(), ((predicted == labels) & mask).sum().item())
    #         correct_train += ((predicted == labels) & mask).sum().item()
    #     epoch_acc = 100. * correct_train / total_train
    #     print("correct train: ", correct_train, " total train: ", total_train)
            
            current_lr = optimizer.param_groups[0]['lr']
            validation_loss = validate_model(model, val_loader, device)
            scheduler.step(metrics=validation_loss)  # TODO: disabled for Adam optimizer
            wandb_logger.log({
                'train loss': loss.item(),
                'epoch': epoch,
                'learning rate': current_lr
            })

        if SAVE_CHECKPOINT and epoch % SAVE_EVERY_N_EPOCH==0:
            if not os.path.exists(CHECKPOINT_DIR):
                os.mkdir(CHECKPOINT_DIR)
            file_name = f"unet_epoch_{epoch}.pth"
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, file_name))
            print(f"Saved model {file_name}")
    # # eval_model(model, test_loader=test_dataloader, device=device) 
    print('Training complete')
    return model

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    train_dataset = VOCSegmentationClass(image_set='train', year='2012')
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = BATCH_SIZE,
        shuffle=True,
        num_workers = 2,
        pin_memory = True
    ) 
    val_dataset = VOCSegmentationClass(image_set='val', year='2012')
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size = BATCH_SIZE,
        shuffle=False,  # A bit more deterministic here
        num_workers = 2,
        pin_memory = True
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_num = len(train_dataset.classes)
    # TODO: let's see this imbalance dataset
    # criterion = DiceLoss()
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    # momentum=0.9
    # LEARNING_RATE=0.001

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=False, map_location=device))
        print("loaded model")
    else:
        model = UNet(class_num = class_num, 
                     intermediate_before_max_pool=INTERMEDIATE_BEFORE_MAX_POOL)
        print("Initialized model")
    model.to(device)
    optimizer = optim.RMSprop(model.parameters(),
                              lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM,
                              foreach=True)
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)  # goal: minimize Dice score

    wandb_logger = wandb.init(project='Rico-U-Net', resume='allow', anonymous='must')
    wandb_logger.config.update(
        dict(epochs=NUM_EPOCHS, batch_size=BATCH_SIZE * ACCUMULATION_STEPS, learning_rate=LEARNING_RATE,
             weight_decay = WEIGHT_DECAY,
             training_size = len(train_dataset),
             validation_size = len(val_dataset),
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
        Validation size: {len(val_dataset)}
        Intermediate_before_max_pool : {INTERMEDIATE_BEFORE_MAX_POOL}
        Checkpoints:     {SAVE_CHECKPOINT}
        Device:          {device.type}
        Mixed Precision: {AMP},
        Optimizer:       {str(optimizer)}
    ''')
    try:
        train_model(model = model, train_loader = train_dataloader, 
                    val_loader = val_dataloader, criterion = criterion, 
                    optimizer = optimizer, scheduler = scheduler,
                            NUM_EPOCHS=NUM_EPOCHS, device=device, num_training = len(train_dataset),
                            wandb_logger=wandb_logger)
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        model.use_checkpointing()
        train_model(model = model, train_loader = train_dataloader, 
                    val_loader = val_dataloader, criterion = criterion, 
                    optimizer = optimizer, scheduler = scheduler,
                            NUM_EPOCHS=NUM_EPOCHS, device=device, num_training = len(train_dataset),
                            wandb_logger=wandb_logger)

    # [optional] finish the wandb run, necessary in notebooks                                                                      
    wandb.finish()