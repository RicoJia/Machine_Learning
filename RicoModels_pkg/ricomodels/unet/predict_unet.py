#! /usr/bin/env python3
import torch
from ricomodels.utils.data_loading import VOCSegmentationClass

BATCH_SIZE = 16

if __name__ == "__main__":
    test_dataset = VOCSegmentationClass(image_set='test')
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = BATCH_SIZE,
        shuffle=True,
        num_workers = 2,
        pin_memory = True
    )