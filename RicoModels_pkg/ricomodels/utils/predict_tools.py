#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn.functional as F


def allclose_replace_nan(tensor1, tensor2, rtol=1e-05, atol=1e-08, sentinel=0.0):
    """
    torch.allclose() does not handle nan. This is to replace nan with sentinel
    """
    tensor1_replaced = torch.where(
        torch.isnan(tensor1), torch.full_like(tensor1, sentinel), tensor1
    )
    tensor2_replaced = torch.where(
        torch.isnan(tensor2), torch.full_like(tensor2, sentinel), tensor2
    )
    return tensor1_replaced, tensor2_replaced


def mask(labels, input_image, class_names, classes=[]):
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
            mask &= labels != c
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
            prediction.unsqueeze(0),
            size=original_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    elif (
        prediction.dim() == 2
    ):  # Single image (assuming it's a mask) without channels (H', W')
        resized_prediction = (
            F.interpolate(
                prediction.unsqueeze(0).unsqueeze(0).float(),
                size=original_size,
                mode="nearest",
            )
            .squeeze(0)
            .squeeze(0)
            .long()
        )
    else:
        raise ValueError("Unsupported prediction shape")

    return resized_prediction
