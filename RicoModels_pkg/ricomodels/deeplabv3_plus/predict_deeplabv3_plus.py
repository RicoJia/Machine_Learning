#!/usr/bin/env python3

import os
from ricomodels.utils.data_loading import (
    get_package_dir,
    PredictDataset
)
from ricomodels.utils.visualization import visualize_image_target_mask
from ricomodels.utils.training_tools import (
    load_model
)
import torch
from torchvision import models
from typing import List
from tqdm import tqdm
import numpy as np

pascal_voc_classes = [
    'background',
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'potted plant',
    'sheep',
    'sofa',
    'train',
    'tv/monitor'
]

class PredictBench:
    def __init__(self, model) -> None:
        """
        Args:
            model : model that's loaded or downloaded
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Will download the model for the first time, takes about 10s
        self.model = model
        self.model.to(self.device)
        self.model.eval()
    @torch.inference_mode()
    def predict(self, images: List):
        torch.cuda.empty_cache()
        dataset = PredictDataset(images)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        outputs = []
        with tqdm(total=len(dataset), desc=f"Prediction", unit="images") as pbar:
            for predict_batch in dataloader:
                predict_batch = predict_batch.to(self.device)
                with torch.autocast(device_type=str(self.device), dtype=torch.float16):
                    output = self.model(predict_batch)
                output = output["out"].cpu()
                output = output.squeeze().argmax(0).numpy()
                outputs.append(output)
                pbar.update(1)
        return outputs



if __name__ == "__main__":
    from PIL import Image
    import matplotlib.pyplot as plt
    # import cv2
    # image = np.asarray(Image.open("/home/ricojia/Downloads/man_car.jpg").convert("RGB"))
    image = np.asarray(Image.open("/home/ricojia/Downloads/dinesh.jpg").convert("RGB"))
    bench = PredictBench(
        # aux_loss: If True, include an auxiliary classifier
        model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    )
    outputs = bench.predict([image])
    for output_batch in outputs:
        #TODO Remember to remove
        print(f'output_batch.shape: {output_batch.shape}')
        # cv2.imshow("pic", output_batch)
        # cv2.waitKey(0)
        plt.imshow(output_batch)
        plt.show()
    
