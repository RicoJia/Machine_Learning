#!/usr/bin/env python3

import os
from typing import List, Tuple

import torch
from ricomodels.backbones.mobilenetv2.mobilenet_v2 import MobileNetV2
from ricomodels.utils.data_loading import (
    CLASSIFICATION_PRED_TRANSFORMS,
    COCODataset,
    PredictDataset,
    TaskMode,
    get_package_dir,
)
from ricomodels.utils.losses import MULTICLASS_CLASSIFICATION_THRE
from ricomodels.utils.training_tools import load_model
from ricomodels.utils.visualization import visualize_image_class_names, model_summary
from tqdm import tqdm


class BackbonePredictBench:
    def __init__(self, model, dataloader, device) -> None:
        """
        Args:
            model : model that's loaded or downloaded
        """
        self.device = device
        # Will download the model for the first time, takes about 10s
        self.model = model
        self.model.to(self.device)
        self.model.eval()
        self.dataloader = dataloader

    @torch.inference_mode()
    def predict(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Predict, and apply mask to mask out specified masked_classes

        Args:
            images (List): list of images. TODO: to make batch prediction compatible if necessary
            masked_classes (list, optional): list to be masked out

        Returns:
            Tuple[List, List]: (outputs, masked_outputs). masked_outputs (H, W, C) could be a list of Nones
        """
        torch.cuda.empty_cache()
        outputs = []
        # TODO: small inefficiency for the sake of code cleanliness,
        images = []
        with tqdm(total=len(dataset), desc=f"Prediction", unit="images") as pbar:
            for augmented_batch, original_size, original_image_batch in self.dataloader:
                augmented_batch = augmented_batch.to(self.device)
                with torch.autocast(device_type=str(self.device), dtype=torch.float16):
                    output = self.model(augmented_batch)
                    output = (output > MULTICLASS_CLASSIFICATION_THRE).bool()
                input_image = original_image_batch.squeeze()
                outputs.append(output.cpu())
                images.append(input_image.cpu())
                pbar.update(1)
        return outputs, images


if __name__ == "__main__":
    val_dataset = COCODataset(
        split="val", task_mode=TaskMode.MULTI_LABEL_IMAGE_CLASSIFICATION
    )
    class_names = val_dataset.class_names
    print(f"class_names: {class_names}")
    dataset = PredictDataset(
        images_dir="/home/ricojia/Downloads/pics/test",
        transforms=CLASSIFICATION_PRED_TRANSFORMS,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    model = MobileNetV2(num_classes=len(class_names), output_stride=4)
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu"),)
    # device = torch.device("cpu")
    MODEL_PATH = os.path.join(
        get_package_dir(), "backbones/mobilenetv2/mobilenetv2.pth"
    )
    load_model(model_path=MODEL_PATH, model=model, device=device)
    model_summary(
        model=model,
        input_size=(
            [
                256,
            ]
        ),
        batch_dim=0,
    )
    bench = BackbonePredictBench(dataloader=dataloader, model=model, device=device)
    outputs, images = bench.predict()
    for img, pred in zip(images, outputs):
        visualize_image_class_names(
            image=img, pred_cat_ids=pred.squeeze(), class_names=class_names
        )
