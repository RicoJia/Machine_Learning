#! /usr/bin/env python3
import torch
from ricomodels.utils.losses import dice_loss, DiceLoss
from ricomodels.utils.data_loading import VOCSegmentationClass, get_pkg_dir
from ricomodels.unet.unet import UNet
import time
import os
from torch import optim
import tqdm
import wandb
import logging

BATCH_SIZE = 16
MODEL_PATH = os.path.join(get_pkg_dir(), "unet/unet_pascal.pth")
ACCUMULATION_STEPS = 8
NUM_EPOCHS=70
LEARNING_RATE=0.002
SAVE_CHECKPOINT = True
AMP = False
INTERMEDIATE_BEFORE_MAX_POOL = False

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
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS=25, device='cpu'):
    model.to(device)
    for epoch in range(NUM_EPOCHS):
        # Training phase
        start = time.time()
        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}] ')
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            print(inputs.shape)
    #         inputs = inputs.to(device)
    #         labels = labels.to(device)
    #         # Forward pass
    #         outputs = model(inputs)
    #         # This is because torch.nn.CrossEntropyLoss(reduction='mean') is true, so to simulate a larger batch, we need to further divide
    #         # print(f"output: {outputs.dtype}, labels: {labels.dtype}")
    #         # loss = (criterion(outputs, labels) + dice_loss(outputs=outputs, labels=labels, ))/ACCUMULATION_STEPS
    #         loss = criterion(outputs, labels)

    #         # TODO
    #         # Backward pass and optimization
    #         loss.backward()
    #         if (i+1)%ACCUMULATION_STEPS == 0:
    #             optimizer.step()
    #             # Zero the parameter gradients
    #             optimizer.zero_grad()
    #             # break #TODO
    #         # Statistics
    #         running_loss += loss.item() * inputs.size(0)
    #         _, predicted = outputs.max(1)
    #         # print(predicted.shape)
    #         mask = (labels != 21)
    #         total_train += mask.sum().item()
    #         # print((predicted == labels).sum().item(), ((predicted == labels) & mask).sum().item())

    #         correct_train += ((predicted == labels) & mask).sum().item()

    #     # adjust after every epoch
    #     current_lr = optimizer.param_groups[0]['lr']
    #     print(f"Current learning rate: {current_lr}")
    #     epoch_loss = running_loss / len(train_loader.dataset)
    #     epoch_acc = 100. * correct_train / total_train
    #     print("correct train: ", correct_train, " total train: ", total_train)
    #     end = time.time()
        
    #     validation_loss = validate_model(model, val_loader, device)
    #     scheduler.step(metrics=validation_loss)  # TODO: disabled for Adam optimizer
    #     print("elapsed: ", end-start)

    #     print(f"Saving Models")
    #     torch.save(model.state_dict(), MODEL_PATH)
    #     print(f"epoch: {epoch}, saved the model. "
    #           f'Train Loss: {epoch_loss:.4f} '
    #           f'Train Acc: {epoch_acc:.2f}% ')
    # # eval_model(model, test_loader=test_dataloader, device=device) 
    print('Training complete')
    return model

if __name__ == "__main__":
    train_dataset = VOCSegmentationClass(image_set='train', year='2007')
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = BATCH_SIZE,
        shuffle=True,
        num_workers = 2,
        pin_memory = True
    ) 
    val_dataset = VOCSegmentationClass(image_set='val', year='2007')
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
    criterion = DiceLoss()
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    weight_decay = 0.0001
    # momentum=0.9
    # LEARNING_RATE=0.001

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=False, map_location=device))
        print("loaded model")
    else:
        model = UNet(class_num = class_num, 
                     intermediate_before_max_pool=INTERMEDIATE_BEFORE_MAX_POOL)
        print("Initialized model")
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)  # goal: minimize Dice score
    model.to(device)

    experiment = wandb.init(project='Rico-U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=NUM_EPOCHS, batch_size=BATCH_SIZE * ACCUMULATION_STEPS, learning_rate=LEARNING_RATE,
             training_size = len(train_dataset),
             validation_size = len(val_dataset),
             intermediate_before_max_pool = INTERMEDIATE_BEFORE_MAX_POOL,
             save_checkpoint=SAVE_CHECKPOINT, amp=AMP)
    )
    logging.info(f'''Starting training:
        Epochs:          {NUM_EPOCHS}
        Batch size:      {BATCH_SIZE}
        Learning rate:   {LEARNING_RATE}
        Training size:   {len(train_dataset)}
        Validation size: {len(val_dataset)}
        Intermediate_before_max_pool : {INTERMEDIATE_BEFORE_MAX_POOL}
        Checkpoints:     {SAVE_CHECKPOINT}
        Device:          {device.type}
        Mixed Precision: {AMP}
    ''')
    train_model(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler,
                        NUM_EPOCHS=NUM_EPOCHS, device=device)

    # [optional] finish the wandb run, necessary in notebooks                                                                      
    wandb.finish()