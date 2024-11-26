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
from torch.nn import functional as F
from torchvision import models
from typing import List, Tuple
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

def mask(labels, input_image, class_names, classes = []):
    """Masking out pixels belonging to classes

    Args:
        labels (torch.tensor): predicted mask (currently just 1 image)
        input_image (torch.Tensor): currently just 1 image
        class_names (list): sorted classes of the predictor
        classes (list, optional): list of classes to be masked OUT

    Raises:
        ValueError: Unsupported input image dimensions

    Returns:
        masked_image
    """
    # labels: torch.Size([480, 640]), input_image: torch.Size([1, 3, 256, 256])
    if isinstance(labels, np.ndarray):
        # This ensures the array is contiguous in memory and eliminates any negative strides. 
        # It works without creating an unnecessary copy if the array is already contiguous.
        input_image = np.ascontiguousarray(input_image)
        labels = torch.from_numpy(labels)
    # Convert input_image to torch.Tensor if it's an ndarray
    if isinstance(input_image, np.ndarray):
        input_image = torch.from_numpy(input_image)
    # Create a mask where the label is in the specified classes
    if classes:
        mask = torch.ones_like(labels, dtype=torch.bool)
        for cl in classes:
            c = class_names.index(cl)
            mask &= (labels != c)
    else:
        # If no classes are specified, include all labels
        mask = torch.ones_like(labels, dtype=torch.bool) 
    # Expand the mask to match the input image dimensions
    if input_image.dim() == 3:
        if input_image.shape[0] == 3:  # (3, H, W)
            mask = mask.unsqueeze(0)  # (1, H, W)
        else:  # (H, W, 3)
            mask = mask.unsqueeze(-1)  # (H, W, 1)
    elif input_image.dim() == 2:
        pass  # Mask already matches the dimensions
    else:
        raise ValueError("Unsupported input image dimensions")
     # Apply the mask to the input image
    masked_image = input_image * mask
    return masked_image

def resize_prediction(prediction, original_size):
    """
    Resize prediction tensor to the original image size.

    Parameters:
    - prediction (torch.Tensor): Predicted mask of shape (B, C, H', W') or (H', W').
    - original_size (tuple): Original image size (H, W).

    Returns:
    - torch.Tensor: Resized prediction of shape (B, C, H, W) or (H, W).
    """
    if prediction.dim() == 4:  # Batch mode (B, C, H', W')
        resized_prediction = F.interpolate(
            prediction, size=original_size, mode="bilinear", align_corners=False
        )
    elif prediction.dim() == 3:  # Single image with channels (C, H', W')
        resized_prediction = F.interpolate(
            prediction.unsqueeze(0), size=original_size, mode="bilinear", align_corners=False
        ).squeeze(0)
    elif prediction.dim() == 2:  # Single image (assuming it's a mask) without channels (H', W')
        resized_prediction = F.interpolate(
            prediction.unsqueeze(0).unsqueeze(0).float(),
            size=original_size,
            mode="nearest",
        ).squeeze(0).squeeze(0).long()
    else:
        raise ValueError("Unsupported prediction shape")

    return resized_prediction

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
    def predict(self, images: List, masked_classes = []) -> Tuple[List, List]:
        """Predict, and apply mask to mask out specified masked_classes 

        Args:
            images (List): list of images. TODO: to make batch prediction compatible if necessary
            masked_classes (list, optional): list to be masked out

        Returns:
            Tuple[List, List]: (outputs, masked_outputs). masked_outputs (H, W, C) could be a list of Nones
        """
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
        masked_outputs = []
        with tqdm(total=len(dataset), desc=f"Prediction", unit="images") as pbar:
            for predict_batch, original_size in dataloader:
                predict_batch = predict_batch.to(self.device)
                with torch.autocast(device_type=str(self.device), dtype=torch.float16):
                    output = self.model(predict_batch)
                output = output["out"]
                output = output.squeeze().argmax(0)
                output = resize_prediction(prediction=output, original_size=original_size)
                input_image = resize_prediction(prediction=predict_batch[0], original_size=original_size)
                if masked_classes:
                    masked_output = mask(labels=output, 
                                         input_image=input_image, 
                                         class_names=self.class_names, 
                                         # convert C, H, W to  H,W,C for visualization.
                                         classes=masked_classes).permute(1, 2, 0).cpu().numpy()
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
    image = np.asarray(Image.open("/home/ricojia/Downloads/dinesh.jpg").convert("RGB"))
    bench = PredictBench(
        # aux_loss: If True, include an auxiliary classifier
        model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    )
    outputs = bench.predict([image])
    for output_batch in outputs:
        # cv2.imshow("pic", output_batch)
        # cv2.waitKey(0)
        plt.imshow(output_batch)
        plt.show()
    
