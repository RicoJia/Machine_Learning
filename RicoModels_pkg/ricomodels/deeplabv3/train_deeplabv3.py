#! /usr/bin/env python3
import argparse
import logging
import os

from ricomodels.utils.data_loading import (get_carvana_datasets,
                                           get_data_loader, get_gta5_datasets,
                                           get_package_dir,
                                           get_VOC_segmentation_datasets)

from ricomodels.utils.training_tools import (EarlyStopping,
                                             check_model_image_channel_num,
                                             eval_model, parse_args)

USE_AMP = False
SAVE_CHECKPOINTS = False
MODEL_PATH = os.path.join(get_package_dir(), "deeplabv3/deeplabv3.pth")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    USE_AMP = args.use_amp
    SAVE_CHECKPOINTS = args.save_checkpoints
