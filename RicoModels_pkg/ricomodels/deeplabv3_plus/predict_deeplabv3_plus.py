#!/usr/bin/env python3

from ricomodels.utils.data_loading import (
    get_package_dir,
    PredictDataset,
    PRED_SEG_AUGMENTATION_TRANSFORMS,
)
from ricomodels.utils.visualization import visualize_image_target_mask
from ricomodels.utils.predict_tools import resize_prediction, mask
from ricomodels.utils.training_tools import load_model
import torch
from torch.nn import functional as F
from torchvision import models
from typing import List, Tuple
from tqdm import tqdm
import numpy as np

pascal_voc_classes = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv/monitor",
]


class PredictBench:
    def __init__(self, model, dataloader) -> None:
        """
        Args:
            model : model that's loaded or downloaded
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Will download the model for the first time, takes about 10s
        self.model = model
        self.model.to(self.device)
        self.model.eval()
        self.dataloader = dataloader

    @torch.inference_mode()
    def predict(self, masked_classes=[]) -> Tuple[List, List]:
        """Predict, and apply mask to mask out specified masked_classes

        Args:
            images (List): list of images. TODO: to make batch prediction compatible if necessary
            masked_classes (list, optional): list to be masked out

        Returns:
            Tuple[List, List]: (outputs, masked_outputs). masked_outputs (H, W, C) could be a list of Nones
        """
        torch.cuda.empty_cache()
        outputs = []
        masked_outputs = []
        with tqdm(total=len(dataset), desc=f"Prediction", unit="images") as pbar:
            for predict_batch, original_size, original_image in self.dataloader:
                predict_batch = predict_batch.to(self.device)
                with torch.autocast(device_type=str(self.device), dtype=torch.float16):
                    output = self.model(predict_batch)
                output = output["out"]
                output = output.squeeze().argmax(0)
                output = resize_prediction(
                    prediction=output, original_size=original_size
                )
                input_image = resize_prediction(
                    prediction=predict_batch[0], original_size=original_size
                )
                if masked_classes:
                    masked_output = (
                        mask(
                            labels=output,
                            input_image=input_image,
                            class_names=self.class_names,
                            # convert C, H, W to  H,W,C for visualization.
                            classes=masked_classes,
                        )
                        .permute(1, 2, 0)
                        .cpu()
                        .numpy()
                    )
                else:
                    masked_output = None
                outputs.append(output.cpu().numpy())
                masked_outputs.append(masked_output)
                pbar.update(1)
        return outputs, masked_outputs

    @property
    def class_names(self):
        return pascal_voc_classes


if __name__ == "__main__":
    from PIL import Image
    import matplotlib.pyplot as plt

    dataset = PredictDataset(
        images_dir="/home/ricojia/Downloads/pics",
        transforms=PRED_SEG_AUGMENTATION_TRANSFORMS,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    bench = PredictBench(
        dataloader=dataloader,
        model=models.segmentation.deeplabv3_resnet101(pretrained=True),
    )
    outputs, masked_outputs = bench.predict()
    for seg_res, masked_batch in zip(outputs, masked_outputs):
        # Not visualizing masked_batch because it's none
        plt.imshow(seg_res)
        plt.show()
