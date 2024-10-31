#!/usr/bin/env python3

import argparse
import functools
import logging

import numpy as np
import torch
from ricomodels.utils.losses import DiceLoss, dice_loss, focal_loss
from ricomodels.utils.visualization import (get_total_weight_norm,
                                            visualize_image_target_mask)
from tqdm import tqdm


def parse_args():
    """
    Parse args and set global input args
    """
    parser = argparse.ArgumentParser(description="Set training options")
    parser.add_argument(
        "--use_amp",
        "-u",
        action="store_true",
        help="Enable automatic mixed precision (AMP)",
    )
    parser.add_argument(
        "--save_checkpoints",
        "-s",
        action="store_true",
        help="Enable saving model checkpoints",
    )
    args = parser.parse_args()
    return args


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
    model, test_dataloader, device, class_num, visualize: bool = False, msg: str = ""
):
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
            # This is for deeplab
            if "out" in outputs_test:
                outputs_test = outputs_test["out"]
            _, predicted_test = outputs_test.max(1)
            # Returning an int
            local_total = labels_test.numel()
            # Not doing item() here because that's an implicit synchronization call
            # .cpu(), .numpy() have synchronization calls, too
            local_correct = (predicted_test == labels_test).sum()
            total_test += local_total
            correct_test += local_correct

            # labels_test: (m, h, w)
            if visualize:
                # print(f'Predicted test acc {100. * local_correct/local_total}%')
                for img, pred, lab in zip(inputs_test, predicted_test, labels_test):
                    # print("pred uniq: ", torch.unique(pred), "lab uniq: ", torch.unique(lab))
                    visualize_image_target_mask(
                        image=img.cpu(), target=pred.cpu(), labels=lab.cpu()
                    )

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
    class_num,
    visualize: bool = False,
):
    logging.info("Evaluating the model ... ")
    train_acc = _eval_model(
        model=model,
        test_dataloader=train_dataloader,
        device=device,
        visualize=False,
        class_num=class_num,
        msg="Train Loader",
    )
    val_acc = _eval_model(
        model=model,
        test_dataloader=val_dataloader,
        device=device,
        visualize=False,
        class_num=class_num,
        msg="Validate Loader",
    )
    test_acc = _eval_model(
        model=model,
        test_dataloader=test_dataloader,
        device=device,
        visualize=visualize,
        class_num=class_num,
        msg="Test Loader",
    )
    return train_acc, val_acc, test_acc
