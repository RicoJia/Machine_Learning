#! /usr/bin/env python3
import logging
import os

import torch
import torchvision
from ricomodels.deeplabv3.train_deeplabv3 import MODEL_PATH
from ricomodels.utils.data_loading import (PredictionDataset,
                                           get_carvana_datasets,
                                           get_data_loader, get_gta5_datasets,
                                           get_VOC_segmentation_datasets)
from ricomodels.utils.prediction_tools import predict
from ricomodels.utils.training_tools import eval_model

BATCH_SIZE = 16


def load_pretrained_model():
    """
    For testing the efficacy of deeplabv3 only
    """
    if os.path.exists(MODEL_PATH):
        model = torchvision.models.segmentation.deeplabv3_resnet101(
            weights=None, aux_loss=True
        )
        state_dict = torch.load(
            MODEL_PATH, map_location="cpu"
        )  # Adjust 'map_location' as needed
        model.load_state_dict(state_dict)
        logging.info(f"Loaded model in {MODEL_PATH}")
    else:
        model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
        torch.save(model.state_dict(), MODEL_PATH)
        logging.info(f"Saved model in {MODEL_PATH}")

    return model


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

    model = load_pretrained_model()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # eval_model(
    #     model=model,
    #     train_dataloader=train_dataloader,
    #     val_dataloader=val_dataloader,
    #     test_dataloader=test_dataloader,
    #     device=device,
    #     class_num=class_num,
    #     visualize=True,
    # )

    prediction_dataset = PredictionDataset()
    predict(dataset=prediction_dataset, device=device, model=model)
