#!/usr/bin/env python3

import functools
import logging
import os
from typing import List

import numpy as np
import torch
from ricomodels.utils.data_loading import TaskMode
from ricomodels.utils.losses import (
    AccuracyCounter,
    F1ScoreCounter,
)
from ricomodels.utils.visualization import (
    get_total_weight_norm,
    visualize_image_class_names,
    visualize_image_target_mask,
)
from tqdm import tqdm


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


def load_model_and_optimizer(model, optimizer, path, device):
    print(f"Rico: model_path: {path}")
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        print(f"Model loaded from {path}, last trained epoch: {epoch}")

        # Move optimizer state to the correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        return model, optimizer, epoch
    else:
        print("Model is fresh-initialized.")
        return model, optimizer, 0


def save_model_and_optimizer(model, optimizer, epoch, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    print(f"Model saved to {path}")


def load_model(model_path, model, device):
    if os.path.exists(model_path):
        model.load_state_dict(
            torch.load(model_path, weights_only=False, map_location=device)
        )
        print("Loaded model")
    else:
        print("Initialized model")


def get_scheduled_probability(start_p, end_p, d):
    """
    start_p * (end_p / start_p) ^ d
    Args:
        start_p (_type_): starting probability
        end_p (_type_): ending probability
        d (_type_): exponential term in [1, 0]
    """

    def check_in_range(var, var_name):
        if var < 0.0 or var > 1.0:
            raise ValueError(f"{var_name} must be in [0, 1], but it got value {var}")

    # TODO This is hacky
    if end_p < 0.0001:
        end_p = 0.0001
    check_in_range(start_p, "start_p")
    check_in_range(end_p, "end_p")
    check_in_range(d, "d")
    return start_p * (end_p / start_p) ** d


def clip_gradients(model, gradient_clipped_norm_max):
    """Gradient clipping, should be called after unscaling

    Args:
        model (_type_): _description_
    """
    need_clipping = False
    for name, param in model.named_parameters():
        if torch.isinf(param.grad).any():
            print("inf: ", name)
            need_clipping = True
        if torch.isnan(param.grad).any():
            print("nan: ", name)
            need_clipping = True
    if need_clipping:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=gradient_clipped_norm_max
        )
        print(f"Applied gradient clipping to norm :{gradient_clipped_norm_max}")


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
    model,
    test_dataloader,
    device,
    class_num,
    task_mode: TaskMode,
    visualize: bool = False,
    msg: str = "",
    class_names=[],
    multiclass_thre=0.5,
):
    """
    class_names: optional, only required for TaskMode.MULTI_LABEL_IMAGE_CLASSIFICATION.
    """
    if test_dataloader is None:
        return float("nan")
    torch.cuda.empty_cache()
    # Evaluation phase
    num_images = len(test_dataloader)
    model.eval()
    if task_mode == TaskMode.MULTI_LABEL_IMAGE_CLASSIFICATION:
        performance_counter = F1ScoreCounter(device=device)
        print("multiclass_thre: ", multiclass_thre)
    else:
        performance_counter = AccuracyCounter(device=device)

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    # TODO I AM ITERATING OVER TRAIN_LOADER, SO I'M MORE SURE
    with torch.autograd.set_grad_enabled(True):
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
                        epoch_total=labels_test.numel(),
                    )
                elif task_mode == TaskMode.MULTI_LABEL_IMAGE_CLASSIFICATION:
                    # [1, 1, 0...]
                    predicted_test = (outputs_test > multiclass_thre).bool()
                    performance_counter.update(
                        true_positives=(predicted_test & labels_test.bool()).sum(),
                        actual_positives=torch.count_nonzero(labels_test),
                        pred_positives=torch.count_nonzero(predicted_test),
                    )
                else:
                    raise RuntimeError(
                        f"Evaluation for task mode {task_mode} has NOT been implemented yet"
                    )

                # labels_test: (m, h, w)
                if visualize:
                    if task_mode == TaskMode.IMAGE_SEGMENTATION:
                        for img, pred, lab in zip(
                            inputs_test, predicted_test, labels_test
                        ):
                            visualize_image_target_mask(
                                image=img.cpu(), target=pred.cpu(), labels=lab.cpu()
                            )
                    elif task_mode == TaskMode.MULTI_LABEL_IMAGE_CLASSIFICATION:
                        for img, pred, lab in zip(
                            inputs_test, predicted_test, labels_test
                        ):
                            visualize_image_class_names(
                                image=img.cpu(),
                                pred_cat_ids=pred,
                                ground_truth_cat_ids=lab,
                                class_names=class_names,
                            )

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
        task_mode=task_mode,
        class_names=class_names,
        multiclass_thre=multiclass_thre,
        msg="Train Loader",
    )
    _eval_model(
        model=model,
        test_dataloader=val_dataloader,
        device=device,
        visualize=visualize,
        class_num=class_num,
        task_mode=task_mode,
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
        task_mode=task_mode,
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
        print(f"============")
        f1 = _eval_model(
            model,
            train_dataloader,
            device,
            class_num,
            task_mode,
            msg="Finding best threshold",
            class_names=class_names,
            multiclass_thre=i,
        )
        if f1 > best_score:
            best_score = f1
            best_threshold = i
        print(
            f"Thre under testing: {i}, f1 score: {f1}, current best: {best_score}, current best thre: {best_threshold}"
        )
    print(f"Final best: {best_score}, final thre: {best_threshold}")
    return best_threshold
