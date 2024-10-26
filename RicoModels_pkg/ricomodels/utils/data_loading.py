#! /usr/bin/env python3
import importlib.util
import logging
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cache, cached_property
from typing import List, Tuple

import albumentations as A
# pytorch is a file but not a registered module, so it has to be imported separately
import albumentations.pytorch as At
import cv2
import numpy as np
import requests
import torch
from PIL import Image
from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms
from torchvision.transforms import CenterCrop, v2
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm


def replace_tensor_val(tensor, a, b):
    # albumentations could pass in extra args
    tensor[tensor == a] = b
    return tensor


@cache
def get_package_dir():
    spec = importlib.util.find_spec("ricomodels")

    # Get the absolute path of the package
    if spec:
        ricomodels_init = spec.origin
        return os.path.dirname(ricomodels_init)
    else:
        raise FileNotFoundError("Package 'ricomodels' not found")


DATA_DIR = os.path.join(get_package_dir(), "data")
IGNORE_INDEX = 0

np.random.seed(42)
# transforms for mask and image
AUGMENTATION_TRANSFORMS = A.Compose(
    [
        A.Resize(
            height=256,
            width=256,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
        ),
        A.HorizontalFlip(p=0.5),
        A.RandomCrop(width=256, height=256),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.ElasticTransform(p=1.0),
        A.Lambda(
            mask=lambda x, **kwargs: replace_tensor_val(x, 255, IGNORE_INDEX).astype(
                np.int64
            )
        ),
        # need to convert from uint8 to float32
        # A.Normalize(
        #     mean=(0.485, 0.456, 0.406),
        #     std=(0.229, 0.224, 0.225)
        # ),
        A.Lambda(image=lambda x, **kwargs: x.astype(np.float32) / 255.0),
        # Converts to [C, H, W] after all augmentations
        At.ToTensorV2(transpose_mask=True),
    ]
)


def download_file(url, dest_path, chunk_size=1024):
    """
    A generic function for downloading from an url to dest_path
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for HTTP errors
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = chunk_size  # 1 KB

        # Create the destination directory if it doesn't exist
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "wb") as file, tqdm(
            desc=f"Downloading {os.path.basename(dest_path)}",
            total=total_size_in_bytes,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                file.write(data)
                bar.update(len(data))
        print(f"Downloaded: {dest_path}\n")
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred while downloading {url}: {http_err}")
    except Exception as err:
        print(f"An error occurred while downloading {url}: {err}")


def extract_zip(zip_path, extract_to):
    """
    Extracts a ZIP file to a specified directory.

    Args:
        zip_path (str): The path to the ZIP file.
        extract_to (str): The directory where files will be extracted.
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            print(f"Extracting {zip_path} to {extract_to}...")
            zip_ref.extractall(extract_to)
        print(f"Extraction completed: {extract_to}\n")
    except zipfile.BadZipFile:
        print(f"Error: {zip_path} is not a zip file or it is corrupted.")
    except Exception as e:
        print(f"An error occurred while extracting {zip_path}: {e}")


class BaseDataset(Dataset):
    """
    Load data -> applies augmentation on masks and images
    """

    def __init__(self, images_dir, labels_dir, manual_find_class_num=False):
        self._images_dir = images_dir
        self._labels_dir = labels_dir
        # call this after initializing these variables
        self.images = sorted(os.listdir(self._images_dir))
        self.labels = sorted(os.listdir(self._labels_dir))
        assert len(self.images) == len(
            self.labels
        ), "Number of images and labels should be equal."

        self._max_class = 0 if manual_find_class_num else None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # return an image and a label. In this case, a label is an image with int8 values
        img_path = os.path.join(self._images_dir, self.images[idx])
        label_path = os.path.join(self._labels_dir, self.labels[idx])

        # Open image and label
        image = np.asarray(
            Image.open(img_path).convert("RGB")
        )  # Ensure image is in RGB
        label = np.asarray(Image.open(label_path))

        augmented = AUGMENTATION_TRANSFORMS(image=image, mask=label)
        image = augmented["image"]
        label = augmented["mask"]

        if self._max_class is not None:
            unique_values = np.unique(label)
            self._max_class = max(max(unique_values), self._max_class)
        return image, label


class GTA5Dataset(BaseDataset):
    def __init__(self):
        """
        GTA5/
            ├── images/
            │   ├── train/
            │   │   ├── GTA5_0000.png
            │   └── val/
            │       ├── GTA5_val_0000.png
            ├── labels/
            │   ├── train/
            │   │   ├── GTA5_0000.png
            │   └── val/
            │       ├── GTA5_val_0000.png
        """
        IMAGES_URL = (
            "http://download.visinf.tu-darmstadt.de/data/from_games/data/01_images.zip"
        )
        LABELS_URL = (
            "http://download.visinf.tu-darmstadt.de/data/from_games/data/01_labels.zip"
        )
        images_dir = os.path.join(get_package_dir(), DATA_DIR, "gta5", "images")
        labels_dir = os.path.join(get_package_dir(), DATA_DIR, "gta5", "labels")
        self.download_and_extract(
            url=IMAGES_URL, dir_name=images_dir, zip_name="01_images.zip"
        )
        self.download_and_extract(
            url=LABELS_URL, dir_name=labels_dir, zip_name="01_labels.zip"
        )

        tasks_args = [
            (IMAGES_URL, images_dir, "01_images.zip"),
            (LABELS_URL, labels_dir, "01_labels.zip"),
        ]

        # TODO: this is not creating a pool?
        # Use ThreadPoolExecutor to download and extract concurrently
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit all tasks to the executor
            future_to_task = {
                executor.submit(self.download_and_extract, url, dir_name, zip_name): (
                    url,
                    dir_name,
                    zip_name,
                )
                for url, dir_name, zip_name in tasks_args
            }
            # Process as tasks complete
            for future in as_completed(future_to_task):
                try:
                    future.result()
                except Exception as exc:
                    print(f"An error occurred with {url}: {exc}")
        super().__init__(images_dir=images_dir, labels_dir=labels_dir)

    def download_and_extract(self, url, dir_name, zip_name):
        dest_path = os.path.join(dir_name, zip_name)
        if not os.path.exists(dir_name):
            download_file(url=url, dest_path=dest_path)
            extract_zip(zip_path=dest_path, extract_to=dir_name)
        else:
            print(f"{dir_name} already exists")

    @cached_property
    def classes(self):
        return set(range(35))


class CarvanaDataset(BaseDataset):
    def __init__(self, dataset_name):
        """
        carvana/
            ├── train/
            │   ├── fff9b3a5373f_16.jpg
            ├── train_masks/
            │   ├── fff9b3a5373f_16_mask.jpg/
        Download the Kaggle dataset by:
        1. Create an account on Kaggle
        2. kaggle competitions download -c carvana-image-masking-challenge -f train.zip
        3. kaggle competitions download -c carvana-image-masking-challenge -f train_masks.zip
        kaggle competitions download -c carvana-image-masking-challenge -f test.zip
        4. unzip train.zip
        5. unzip train_masks.zip
        6. unzip test.zip
        """
        if dataset_name not in ("test", "train"):
            raise FileNotFoundError(
                "Carvana dataset can only have 'test' or 'train' sub datasets!"
            )
        images_dir = os.path.join(get_package_dir(), DATA_DIR, "carvana", dataset_name)
        labels_dir = os.path.join(
            get_package_dir(), DATA_DIR, "carvana", dataset_name + "_masks"
        )

        super().__init__(images_dir=images_dir, labels_dir=labels_dir)

    @cached_property
    def classes(self):
        # 0 = background, 1 = car
        return set([0, 1])


class VOCSegmentationDataset(Dataset):
    def __init__(self, image_set, year):
        IMAGE_SET = ("train", "trainval", "val")
        YEARS = ("2007", "2012")
        if image_set not in IMAGE_SET:
            raise ValueError(f"VOC: Image_set '{image_set}' must be one of {IMAGE_SET}")
        if year not in YEARS:
            raise ValueError(f"VOC: Year '{year}' must be one of {YEARS}")

        self._dataset = datasets.VOCSegmentation(
            root=DATA_DIR,
            year=year,
            image_set=image_set,
            download=not self._is_extracted(
                dataset_dir=os.path.join(get_package_dir(), DATA_DIR), year=year
            ),
        )
        self._classes = set()
        print(f"Data {image_set} Successfully Loaded")

    def _is_extracted(self, dataset_dir, year):
        """
        Checking if our data has been extracted
        """
        extracted_train_path = os.path.join(
            dataset_dir,
            "VOCdevkit",
            f"VOC{year}",
            "ImageSets",
            "Segmentation",
            "train.txt",
        )
        return os.path.exists(extracted_train_path)

    @cached_property
    def classes(self):
        """
        Initialize classes in a lazy manner
        """
        if len(self._classes) == 0:
            logging.info("Getting VOC classes")
            for _, target in self._dataset:
                self._classes.update(torch.unique(target).tolist())
        return self._classes

    def __getitem__(self, index):
        # return an image and a label. In this case, a label is an image with int8 values
        image, label = self._dataset[index]

        # Convert to NumPy arrays for Albumentations compatibility
        image = np.array(image)
        label = np.array(label)

        augmented = AUGMENTATION_TRANSFORMS(image=image, mask=label)
        image = augmented["image"]
        label = augmented["mask"]
        return image, label

    def __len__(self):
        return len(self._dataset)


##################################################################
## Tool Functions
##################################################################


def split_dataset(
    main_dataset: Dataset, train_dev_test_split: List[float]
) -> Tuple[Dataset, int]:
    dataset_size = len(main_dataset)
    assert (
        len(train_dev_test_split) == 3
    ), "Please have 3 floats in train_dev_test_split"
    train_dev_test_split = np.array(train_dev_test_split) / np.sum(train_dev_test_split)

    train_size, dev_size, test_size = (train_dev_test_split * dataset_size).astype(int)
    # addressing rounding errors
    test_size = dataset_size - (train_size + dev_size)
    shuffled_main_dataset = torch.utils.data.Subset(
        main_dataset, torch.randperm(dataset_size)
    )
    train_dataset, dev_dataset, test_dataset = random_split(
        shuffled_main_dataset, [train_size, dev_size, test_size]
    )
    class_num = len(main_dataset.classes)
    return train_dataset, dev_dataset, test_dataset, class_num


def get_data_loader(train_dataset, val_dataset, test_dataset, batch_size):
    def data_loader(dataset, shuffle: bool):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
        )

    return (
        data_loader(train_dataset, shuffle=True),
        data_loader(val_dataset, shuffle=False),
        data_loader(test_dataset, shuffle=False),
    )


def get_gta5_datasets():
    main_dataset = GTA5Dataset()
    train_dataset, val_dataset, test_dataset, class_num = split_dataset(
        main_dataset=main_dataset, train_dev_test_split=[0.8, 0.1, 0.1]
    )
    return train_dataset, val_dataset, test_dataset, class_num


def get_carvana_datasets():
    main_dataset = CarvanaDataset(
        dataset_name="train",
    )
    # Pytorch asks for equal lengths of val and test_datasets
    train_dataset, val_dataset, test_dataset, class_num = split_dataset(
        main_dataset=main_dataset, train_dev_test_split=[0.8, 0.1, 0.1]
    )
    # test_dataset = CarvanaDataset(dataset_name="test", )
    return train_dataset, val_dataset, test_dataset, class_num


def get_VOC_segmentation_datasets():
    year = "2012"
    train_dataset = VOCSegmentationDataset(
        image_set="train",
        year=year,
    )
    val_dataset = VOCSegmentationDataset(
        image_set="trainval",
        year=year,
    )
    test_dataset = VOCSegmentationDataset(
        image_set="val",
        year=year,
    )
    class_num = len(train_dataset)
    return train_dataset, val_dataset, test_dataset, class_num


if __name__ == "__main__":
    # rm -rf results/ && python3 data_loading.py && mv /tmp/results/ .
    # eog results/$(ls results/ | head -n1)
    from ricomodels.utils.visualization import visualize_image_target_mask

    # dataset = GTA5Dataset()
    dataset = VOCSegmentationDataset(image_set="train", year="2007")
    for i in range(15):
        image, label = dataset[i]
        img = torch.Tensor(image)
        visualize_image_target_mask(img, target=None, labels=label)
