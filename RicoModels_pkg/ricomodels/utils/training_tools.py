#!/usr/bin/env python3

import functools
import logging

import numpy as np
import torch
from ricomodels.utils.losses import DiceLoss, dice_loss, focal_loss, F1ScoreCounter, AccuracyCounter
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
    model, test_dataloader, device, class_num, task_mode: TaskMode, visualize: bool = False, msg: str = "", 
    class_names=[], multiclass_thre=0.5
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
    if task_mode == TaskMode.MULTI_LABEL_IMAGE_CLASSIFICATION:
        performance_counter = F1ScoreCounter()
        print("multiclass_thre: ", multiclass_thre)
    else:
        performance_counter = AccuracyCounter()

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
                performance_counter.update(
                    epoch_correct=(predicted_test == labels_test).sum(), 
                    epoch_total=labels_test.numel())
            elif task_mode == TaskMode.MULTI_LABEL_IMAGE_CLASSIFICATION:
                # [1, 1, 0...]
                predicted_test = torch.where(outputs_test > multiclass_thre, 1, 0).bool()
                performance_counter.update(
                    true_positives=(predicted_test & labels_test.bool()).sum(),
                    actual_positives=torch.count_nonzero(labels_test),
                    pred_positives=torch.count_nonzero(predicted_test)
                )
            else:
                raise RuntimeError(f"Evaluation for task mode {task_mode} has NOT been implemented yet")
                
            # labels_test: (m, h, w)
            if visualize:
                if task_mode == TaskMode.IMAGE_SEGMENTATION:
                    for img, pred, lab in zip(inputs_test, predicted_test, labels_test):
                        visualize_image_target_mask(
                            image=img.cpu(), target=pred.cpu(), labels=lab.cpu()
                        )
                elif task_mode == TaskMode.MULTI_LABEL_IMAGE_CLASSIFICATION:
                    debug_class_count = 0
                    for img, pred, lab in zip(inputs_test, predicted_test, labels_test):
                        visualize_image_class_names(image=img.cpu(), pred_cat_ids=pred, ground_truth_cat_ids=lab, class_names=class_names)
                    

            # 100 is to make the prob close to 1 after softmax
            pbar.update(1)
    
    logging.info(
        f"""{msg}
            Total weight norm: {get_total_weight_norm(model)}
        """
    )
    performance_counter.print_result()


def eval_model(
    model,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    device,
    class_num: int,
    task_mode: TaskMode,
    class_names: List[str] = [],
    multiclass_thre=0.5,
    visualize: bool = False,
):
    logging.info("Evaluating the model ... ")
    _eval_model(
        model=model,
        test_dataloader=train_dataloader,
        device=device,
        visualize=False,
        class_num=class_num,
        task_mode = task_mode,
        class_names=class_names,
        multiclass_thre=multiclass_thre,
        msg="Train Loader",
    )
    _eval_model(
        model=model,
        test_dataloader=val_dataloader,
        device=device,
        visualize=False,
        class_num=class_num,
        task_mode = task_mode,
        class_names=class_names,
        multiclass_thre=multiclass_thre,
        msg="Validate Loader",
    )
    _eval_model(
        model=model,
        test_dataloader=test_dataloader,
        device=device,
        visualize=visualize,
        class_num=class_num,
        task_mode = task_mode,
        class_names=class_names,
        multiclass_thre=multiclass_thre,
        msg="Test Loader",
    )

def find_best_multi_classification_score(
    model,
    train_dataloader,
    device,
    class_num: int,
    task_mode: TaskMode,
    class_names: List[str] = [],
):
    """
    If the multiclass classifier has not defined a decision threshold, use this.
    """
    best_score = 0
    best_threshold = 0.5
    for i in np.arange(0.1, 0.9, 0.1):
        print(f'============')
        f1 = _eval_model(model, train_dataloader, device, class_num, task_mode, msg = "Finding best threshold", 
                         class_names=class_names, multiclass_thre=i)
        if f1 > best_score:
            best_score = f1
            best_threshold = i
        print(f'Thre under testing: {i}, f1 score: {f1}, current best: {best_score}, current best thre: {best_threshold}')
    print(f'Final best: {best_score}, final thre: {best_threshold}')
    return best_threshold
        