#!/usr/bin/env python3

import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from ricomodels.utils.visualization import visualize_image_target_mask
from torchvision import transforms
from tqdm import tqdm


# Define color palette for DeepLabV3
def decode_segmap(image, nc=21):
    label_colors = np.array(
        [
            (0, 0, 0),  # 0=background
            (128, 0, 0),  # 1=aeroplane
            (0, 128, 0),  # 2=bicycle
            (128, 128, 0),  # 3=bird
            (0, 0, 128),  # 4=boat
            (128, 0, 128),  # 5=bottle
            (0, 128, 128),  # 6=bus
            (128, 128, 128),  # 7=car
            (64, 0, 0),  # 8=cat
            (192, 0, 0),  # 9=chair
            (64, 128, 0),  # 10=cow
            (192, 128, 0),  # 11=dining table
            (64, 0, 128),  # 12=dog
            (192, 0, 128),  # 13=horse
            (64, 128, 128),  # 14=motorbike
            (192, 128, 128),  # 15=person
            (0, 64, 0),  # 16=potted plant
            (128, 64, 0),  # 17=sheep
            (0, 192, 0),  # 18=sofa
            (128, 192, 0),  # 19=train
            (0, 64, 128),  # 20=tv/monitor
        ]
    )

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb


@torch.inference_mode()
def predict(dataset: torch.utils.data.Dataset, device, model):
    torch.cuda.empty_cache()
    model.eval()
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    with tqdm(
        desc=f"Inferencing in batches",
        total=len(dataset),
        unit="im",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for original_image, input_batch in dataset:
            input_batch = input_batch.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.float16):
                outputs_test = model(input_batch)
            if "out" in outputs_test:
                outputs_test = outputs_test["out"]
            output_predictions = outputs_test.argmax(1).squeeze().cpu().numpy()
            segmentation_rgb = decode_segmap(output_predictions)
            visualize_image_target_mask(image=original_image, target=segmentation_rgb)
            bar.update(1)
