#! /usr/bin/env python3
import torch
from ricomodels.utils.data_loading import (
    get_data_loader,
    get_gta5_datasets,
    get_carvana_datasets,
)
from ricomodels.utils.training_tools import eval_model
from ricomodels.unet.train_unet import MODEL_PATH, INTERMEDIATE_BEFORE_MAX_POOL
from ricomodels.unet.unet import UNet
import logging

BATCH_SIZE = 16

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # train_dataset, val_dataset, test_dataset, class_num = get_gta5_datasets()
    train_dataset, val_dataset, test_dataset, class_num = get_carvana_datasets()
    train_dataloader, val_dataloader, test_dataloader = get_data_loader(
        train_dataset, val_dataset, test_dataset, batch_size=BATCH_SIZE
    )
    print(
        f"Lengths of train_dataset, val_dataset, test_dataset: {len(train_dataset), len(val_dataset), len(test_dataset)}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading state dict")
    state_dict = torch.load(MODEL_PATH, map_location=device)
    print("Created barebone unet")
    model = UNet(
        class_num=class_num, intermediate_before_max_pool=INTERMEDIATE_BEFORE_MAX_POOL
    )
    print("Model loading")
    model.load_state_dict(state_dict)
    model.to(device)

    eval_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        device=device,
        class_num=class_num,
        visualize=False,
    )
