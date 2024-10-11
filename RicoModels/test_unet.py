from numpy import ndim
import os
from utils.losses import dice_loss, DiceLoss
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from torchvision.transforms import v2, CenterCrop
import matplotlib.pyplot as plt
from torchvision.transforms.functional import InterpolationMode
from functools import cached_property
import torch.nn.functional as F
import time

###############################################################
# Data Loading
###############################################################

DATA_DIR='./data'
# BATCH_SIZE = 4
BATCH_SIZE = 16
IGNORE_INDEX = 0

def replace_tensor_val(tensor, a, b):
    tensor[tensor==a] = b
    return tensor

def is_extracted(dataset_dir, year='2007'):
    extracted_train_path = os.path.join(dataset_dir, 'VOCdevkit', f'VOC{year}', 'ImageSets', 'Segmentation', 'train.txt')
    return os.path.exists(extracted_train_path)


image_seg_transforms = transforms.Compose([
   v2.Resize((256, 256)),
    # Becareful because you want to rotate your transforms by the same amount
    # v2.RandomHorizontalFlip(),
    # v2.RandomVerticalFlip(),
    # v2.RandomRotation(degrees=15),
   v2.ToTensor(),
   v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

target_seg_transforms = transforms.Compose([
    v2.Resize((256, 256), interpolation=InterpolationMode.NEAREST),
    v2.PILToTensor(),
    v2.Lambda(lambda tensor: tensor.squeeze()),
    v2.Lambda(lambda x: replace_tensor_val(x.long(), 255, IGNORE_INDEX)),
])

class VOCSegmentationClass(Dataset):
   def __init__(self, image_set): 
        # Load PASCAL VOC 2007 dataset for segmentation
        self._dataset = datasets.VOCSegmentation(
            root=DATA_DIR,  # Specify where to store the data
            year='2007',    # Specify the year of the dataset (2007 in this case)
            image_set=image_set,  # You can use 'train', 'val', or 'trainval'
            download=not is_extracted(dataset_dir=DATA_DIR),  # Automatically download if not available
            transform=image_seg_transforms,  # Apply transformations to the images
            target_transform=target_seg_transforms  # Apply transformations to the masks
        )
        self._classes = set()
   @cached_property
   def classes(self):
       if len(self._classes) == 0: 
           for image, target in self._dataset:
            self._classes.update(torch.unique(target).tolist())
       return self._classes
   def __getitem__(self, index): 
       # return an image and a label. In this case, a label is an image with int8 values
       return self._dataset[index]
       # TODO: more transforms?
   def __len__(self):
        return len(self._dataset)

train_dataset = VOCSegmentationClass(image_set='train')
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size = BATCH_SIZE,
    shuffle=True,
    num_workers = 2,
    pin_memory = True
)

val_dataset = VOCSegmentationClass(image_set='val')
val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size = BATCH_SIZE,
    shuffle=True,
    num_workers = 2,
    pin_memory = True
)

test_dataset = VOCSegmentationClass(image_set='test')
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size = BATCH_SIZE,
    shuffle=True,
    num_workers = 2,
    pin_memory = True
)

def visualize_image_and_target(image, target=None, labels=None):
    # # See torch.Size([3, 281, 500]) torch.Size([1, 281, 500])
    # # print(image.shape, target.shape)

    plt.subplot(1,3,1)
    # Making channels the last dimension
    plt.imshow(image.permute(1,2,0))
    plt.title('image')

    if target is not None:
        plt.subplot(1,3,2)
        # Making channels the last dimension
        plt.imshow(target)
        plt.title('mask')

    if labels is not None:
        plt.subplot(1,3,3)
        # Making channels the last dimension
        plt.imshow(labels)
        plt.title('labels')

    # See tensor([  0,   1,  15, 255], dtype=torch.uint8)
    plt.show()
    tiempo = int(time.time() * 1000)
    plt.savefig(str(tiempo)+".png")

# print("classes: ", train_dataset.classes)
# for image, target in train_dataset:
#     visualize_image_and_target(image, labels=target)
#     break

###############################################################
# Model Definition
###############################################################

# This is a regular conv block
from torch import nn as nn
from collections import deque

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu( self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(x)))
        # x = self.relu( self.conv1(x))
        # return self.relu(self.conv2(x))

class Encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # This should include the bottleneck.
        self._layers = nn.ModuleList([ConvBlock(in_channels[i], in_channels[i+1]) for i in range(len(in_channels) - 1)])
        self._maxpool = nn.MaxPool2d(2, stride=2)
    def forward(self, x):
        # returns unpooled output from each block:
        # [intermediate results ... ], but we don't want to return
        intermediate_outputs = deque([])
        for i in range(len(self._layers) - 1):
            x = self._layers[i](x)
            intermediate_outputs.appendleft(x)
            x = self._maxpool(x)
        x = self._layers[-1](x)
        return x, intermediate_outputs

class Decoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self._upward_conv_blocks = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels = channels[i], out_channels = channels[i+1],
                kernel_size=2, stride=2
            ) for i in range(len(channels) - 1)
        ])
        # Then, there's a concat step in between
        self._conv_blocks = nn.ModuleList([
            ConvBlock(in_channels= channels[i], out_channels=channels[i+1])
            for i in range(len(channels) - 1)
        ])

    def forward(self, x, skip_inputs):
        if len(skip_inputs) != len(self._conv_blocks):
            raise ValueError("Please check implementation. Length of skip inputs and _conv_blocks should be the same!",
                             f"skip inputs, blocks inputs: {len(skip_inputs), len(self._conv_blocks)}")
        # x is smaller than skip inputs, because there's no padding in the conv layers
        for skip_input, up_block, conv_block in zip(skip_inputs, self._upward_conv_blocks, self._conv_blocks):
            # print("x shape before upsampling: ", x.shape)
            x = up_block(x)
            # print(skip_input.shape, x.shape)
            # TODO: here's a small detail. The paper didn't specify if we want to append or prepend. This might cause trouble
            skip_input = self.crop(skip_input=skip_input, x=x)
            x = torch.cat((skip_input, x), 1)
            # TODO, I'm really not sure if we need to crop.
            x = conv_block(x)
        return x

    def crop(self, skip_input, x):
        _, _, H, W = x.shape
        return CenterCrop((H,W))(skip_input)

class UNet(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        encoder_in_channels= [3, 64, 128, 256, 512, 1024]    # bottleneck is 128
        decoder_channels = [1024, 512, 256, 128, 64] #?
        self._encoder = Encoder(in_channels=encoder_in_channels)
        self._decoder = Decoder(channels=decoder_channels)
        # 1x1
        self._head = nn.Conv2d(in_channels=decoder_channels[-1], out_channels=class_num, kernel_size=1)
        self._init_weight()

    def forward(self, x):
        _, _, H, W = x.shape
        x, intermediate = self._encoder(x)
        output = self._decoder(x, intermediate)
        output = self._head(output)
        output = torch.nn.functional.interpolate(output, size=(H,W),  mode='nearest')
        return output

    def _init_weight(self):
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    print(f"{type(m)}, he initialization")
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)
                    print(f"{type(m)}, const init")

def forward_pass_poc():
    image, target = train_dataset[0]
    print(target.shape)
    class_num = len(train_dataset.classes)
    image = image.unsqueeze(0)
    _, _, H, W = image.shape
    enc = Encoder([3, 16, 32, 64])
    # # print(image.shape)
    x, intermediate_outputs = enc.forward(image)
    dec = Decoder(channels=[64, 32, 16])
    # torch.Size([1, 16, 216, 216])
    output = dec(x, intermediate_outputs)
    # 1x1
    head = nn.Conv2d(
        in_channels=16,
        out_channels=class_num,
        kernel_size=1,
    )
    output = head(output)
    output = torch.nn.functional.interpolate(output, size=(H,W),  mode='nearest')
    print(output.shape)
# forward_pass_poc()


###############################################################
# Model Eval 
###############################################################

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


###############################################################
# Model Training
###############################################################


import time
from torch import optim

# Define the training function
MODEL_PATH = 'unet_pascal.pth'
ACCUMULATION_STEPS = 8

# Check against example
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=25, device='cpu'):
    model.to(device)
    for epoch in range(num_epochs):
        # Training phase
        start = time.time()
        print(f'Epoch [{epoch + 1}/{num_epochs}] ')
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(inputs)
            # This is because torch.nn.CrossEntropyLoss(reduction='mean') is true, so to simulate a larger batch, we need to further divide
            # print(f"output: {outputs.dtype}, labels: {labels.dtype}")
            # loss = (criterion(outputs, labels) + dice_loss(outputs=outputs, labels=labels, ))/ACCUMULATION_STEPS
            loss = criterion(outputs, labels)

            # TODO
            # Backward pass and optimization
            loss.backward()
            if (i+1)%ACCUMULATION_STEPS == 0:
                optimizer.step()
                # Zero the parameter gradients
                optimizer.zero_grad()
                # break #TODO
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            # print(predicted.shape)
            mask = (labels != 21)
            total_train += mask.sum().item()
            # print((predicted == labels).sum().item(), ((predicted == labels) & mask).sum().item())

            correct_train += ((predicted == labels) & mask).sum().item()

        # adjust after every epoch
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr}")
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100. * correct_train / total_train
        print("correct train: ", correct_train, " total train: ", total_train)
        end = time.time()
        
        validation_loss = validate_model(model, val_loader, device)
        scheduler.step(metrics=validation_loss)  # TODO: disabled for Adam optimizer
        print("elapsed: ", end-start)

        print(f"Saving Models")
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"epoch: {epoch}, saved the model. "
              f'Train Loss: {epoch_loss:.4f} '
              f'Train Acc: {epoch_acc:.2f}% ')
    # eval_model(model, test_loader=test_dataloader, device=device) 
    print('Training complete')
    return model

class_num = len(train_dataset.classes)

# TODO: let's see this imbalance dataset
zero_class_weight = 0.01
other_class_weight = (1-zero_class_weight)/(len(train_dataset.classes)-1)
criterion = DiceLoss()
# criterion = nn.CrossEntropyLoss()
# criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
weight_decay = 0.0001
# momentum=0.9
# learning_rate=0.001
learning_rate=0.002
num_epochs=70
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50], gamma=0.1)
# optimizer = optim.SGD(model. parameters(),lr=0.01, momentum=0.9, weight_decay=0.0005)
# optimizer = optim.SGD(model. parameters(),lr=0.01, momentum=0.9, weight_decay=0.0005)
# multiply learning rate by 0.1 after 30% of epochs

# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(0.3*num_epochs), gamma=0.1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)  # goal: minimize Dice score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Train the model
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=False, map_location=device))
    print("loaded model")
else:
    model = UNet(class_num = class_num)
    print("Initialized model")

model.to(device)


# model = train_model(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler,
#                     num_epochs=num_epochs, device=device)

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
