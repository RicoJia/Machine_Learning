#! /usr/bin/env python3
import os
import importlib.util
from torch.utils.data import Dataset
from functools import cached_property
from torchvision import transforms, datasets
from torchvision.transforms import v2, CenterCrop
from torchvision.transforms.functional import InterpolationMode
import torchvision
import torch
import logging

DATA_DIR = 'data'
IGNORE_INDEX = 0

def replace_tensor_val(tensor, a, b):
    tensor[tensor == a] = b
    return tensor

def get_pkg_dir():
    spec = importlib.util.find_spec("ricomodels")

    # Get the absolute path of the package
    if spec:
        ricomodels_init = spec.origin
        return os.path.dirname(ricomodels_init)
    else:
        raise FileNotFoundError("Package 'ricomodels' not found")

class VOCSegmentationClass(Dataset):
    def __init__(self, image_set, year):
        image_seg_transforms = transforms.Compose([
            v2.Resize((256, 256)),
            # Becareful because you want to rotate your transforms by the same amount
            # v2.RandomHorizontalFlip(),
            # v2.RandomVerticalFlip(),
            # v2.RandomRotation(degrees=15),
            v2.ToTensor(),  # to 0-1
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        target_seg_transforms = transforms.Compose([
            v2.Resize((256, 256), interpolation=InterpolationMode.NEAREST),
            v2.PILToTensor(),
            v2.Lambda(lambda tensor: tensor.squeeze()),
            v2.Lambda(lambda x: replace_tensor_val(
            x.long(), 255, IGNORE_INDEX)),
        ])
        self._dataset = datasets.VOCSegmentation(
            root=DATA_DIR,
            year=year,
            image_set=image_set,
            download=not self._is_extracted(
                dataset_dir=os.path.join(get_pkg_dir(), DATA_DIR), year=year),
            transform=image_seg_transforms,
            target_transform=target_seg_transforms
        )
        self._classes = set()
        #TODO Remember to remove
        print(f'Data {image_set} Successfully Loaded')

    def _is_extracted(self, dataset_dir, year):
        """
        Checking if our data has been extracted
        """
        extracted_train_path = os.path.join(dataset_dir, 'VOCdevkit', f'VOC{year}', 'ImageSets', 'Segmentation', 'train.txt')
        return os.path.exists(extracted_train_path)

    @cached_property
    def classes(self): 
        """
        Initialize classes in a lazy manner
        """
        if len(self._classes) == 0: 
            logging.info("Getting VOC classes") 
            for image, target in self._dataset:
                self._classes.update(torch.unique(target).tolist())
        return self._classes
    def __getitem__(self, index): 
        # return an image and a label. In this case, a label is an image with int8 values
        return self._dataset[index]
    def __len__(self):
        return len(self._dataset)