#! /usr/bin/env python3
import argparse
import logging
import os

from ricomodels.utils.training_tools import (EarlyStopping,
                                             check_model_image_channel_num,
                                             eval_model, parse_args)

USE_AMP = False
SAVE_CHECKPOINTS = False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    USE_AMP = args.use_amp
    SAVE_CHECKPOINTS = args.save_checkpoints
