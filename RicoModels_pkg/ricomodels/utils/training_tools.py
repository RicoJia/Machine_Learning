#!/usr/bin/env python3

import functools
import logging

import numpy as np
import torch
from ricomodels.utils.losses import DiceLoss, dice_loss, focal_loss
from ricomodels.utils.data_loading import TaskMode
from ricomodels.utils.visualization import (
    get_total_weight_norm,
    visualize_image_target_mask,
    visualize_image_class_names
)
from tqdm import tqdm
import os
from typing import List

@functools.cache
def check_model_image_channel_num(model_channels, img_channels):
    if model_channels != img_channels:
        raise ValueError(
            f"Network has been defined with {model_channels} input channels, "
            f"but loaded images have {img_channels} channels. Please check that "
            "the images are loaded correctly."
        )


def validate_model(model, val_loader, device, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            # make sure data is on CPU/GPU
            inputs, labels = inputs.to(device), labels.to(
                device
            )  # Move inputs and labels to the correct device
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
    val_loss /= len(val_loader)
    return val_loss

def load_model(model_path, model, device):
    if os.path.exists(model_path):
        model.load_state_dict(
            torch.load(model_path, weights_only=False, map_location=device)
        )
        print("Loaded model")
    else:
        print("Initialized model")

class EarlyStopping:
    def __init__(self, delta=1e-8, patience=5, verbose=False) -> None:
        """
        patience (int): How many epochs to wait after last validation loss improvement.
            Default: 7
        delta (float): Minimum change in validation loss to consider an improvement.
            Default: 0
        """
        self.patience = patience
        self.counter = 0  # Counts epochs with no improvement
        self.best_score = None  # Best validation loss score
        self.early_stop = False  # Flag to indicate early stopping
        self.verbose = verbose
        self.delta = delta

    def __call__(self, val_loss) -> bool:
        score = -val_loss  # Negative because lower loss is better

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1  # No improvement
            if self.verbose:
                logging.info(
                    f"EarlyStopping counter: {self.counter} out of {self.patience} 👀"
                )
            if self.counter >= self.patience:
                self.early_stop = True  # Trigger early stopping
        else:
            self.best_score = score
            self.counter = 0  # Reset counter if improvement

        return self.early_stop


@torch.inference_mode()
def _eval_model(
    model, test_dataloader, device, class_num, task_mode: TaskMode, visualize: bool = False, msg: str = "", class_names=[]
):
    """
    class_names: optional, only required for TaskMode.MULTI_LABEL_IMAGE_CLASSIFICATION. 
    """
    if test_dataloader is None:
        return float('nan')
    torch.cuda.empty_cache()
    # Evaluation phase
    num_images = len(test_dataloader)
    model.eval()
    correct_test = 0
    total_test = 0
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    i = 0
    # TODO I AM ITERATING OVER TRAIN_LOADER, SO I'M MORE SURE
    with tqdm(total=num_images, desc=f"{msg}", unit="batch") as pbar:
        for inputs_test, labels_test in test_dataloader:
            inputs_test = inputs_test.to(device)
            labels_test = labels_test.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.float16):
                outputs_test = model(inputs_test)
            
            if task_mode == TaskMode.IMAGE_SEGMENTATION:
                _, predicted_test = outputs_test.max(1)
                # Returning an int
                local_total = labels_test.numel()
                local_correct = (predicted_test == labels_test).sum()
            elif task_mode == TaskMode.MULTI_LABEL_IMAGE_CLASSIFICATION:
                # MULTI_LABLE_THRE = 0.5, TODO: can find the best threshold as part of training.
                # [1, 1, 0...]
                predicted_test = torch.where(outputs_test > 0.4, 1, 0).bool()
                # This is basically recall with true_positives 
                local_correct = (predicted_test & labels_test.bool()).sum()
                local_total = torch.count_nonzero(labels_test)
            else:
                raise RuntimeError(f"Evaluation for task mode {task_mode} has NOT been implemented yet")
                
            # Not doing item() here because that's an implicit synchronization call
            # .cpu(), .numpy() have synchronization calls, too
            total_test += local_total
            correct_test += local_correct

            # labels_test: (m, h, w)
            if visualize:
                if task_mode == TaskMode.IMAGE_SEGMENTATION:
                    for img, pred, lab in zip(inputs_test, predicted_test, labels_test):
                        # print("pred uniq: ", torch.unique(pred), "lab uniq: ", torch.unique(lab))
                        visualize_image_target_mask(
                            image=img.cpu(), target=pred.cpu(), labels=lab.cpu()
                        )
                elif task_mode == TaskMode.MULTI_LABEL_IMAGE_CLASSIFICATION:
                    debug_class_count = 0
                    for img, pred, lab in zip(inputs_test, predicted_test, labels_test):
                        visualize_image_class_names(image=img.cpu(), pred_cat_ids=pred, ground_truth_cat_ids=lab, class_names=class_names)
                    

            # 100 is to make the prob close to 1 after softmax
            pbar.update(1)
            # pbar.set_postfix(**{'batch accuracy': f'{100. * local_correct/local_total}%'})
    correct_test = correct_test.cpu().item()
    test_acc = 100.0 * correct_test / total_test
    logging.info(
        f"""{msg}
                Total weight norm: {get_total_weight_norm(model)}
                Accuracy: {test_acc:.2f}% 
                """
    )
    return test_acc


def eval_model(
    model,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    device,
    class_num: int,
    task_mode: TaskMode,
    class_names: List[str] = [],
    visualize: bool = False,
):
    logging.info("Evaluating the model ... ")
    train_acc = _eval_model(
        model=model,
        test_dataloader=train_dataloader,
        device=device,
        visualize=False,
        class_num=class_num,
        task_mode = task_mode,
        class_names=class_names,
        msg="Train Loader",
    )
    val_acc = _eval_model(
        model=model,
        test_dataloader=val_dataloader,
        device=device,
        visualize=False,
        class_num=class_num,
        task_mode = task_mode,
        class_names=class_names,
        msg="Validate Loader",
    )
    test_acc = _eval_model(
        model=model,
        test_dataloader=test_dataloader,
        device=device,
        visualize=visualize,
        class_num=class_num,
        task_mode = task_mode,
        class_names=class_names,
        msg="Test Loader",
    )
    return train_acc, val_acc, test_acc
