#! /usr/bin/env python3
import logging

import torch
import torchvision
from ricomodels.utils.data_loading import (get_carvana_datasets,
                                           get_data_loader, get_gta5_datasets,
                                           get_VOC_segmentation_datasets)
from ricomodels.utils.training_tools import eval_model

BATCH_SIZE = 16


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    train_dataset, val_dataset, test_dataset, class_num = (
        get_VOC_segmentation_datasets()
    )
    train_dataloader, val_dataloader, test_dataloader = get_data_loader(
        train_dataset, val_dataset, test_dataset, batch_size=BATCH_SIZE
    )
    print(
        f"Lengths of train_dataset, val_dataset, test_dataset: {len(train_dataset), len(val_dataset), len(test_dataset)}"
    )

    # Load a pre-trained DeepLabV3 model
    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    eval_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        device=device,
        class_num=class_num,
        visualize=True,
    )
