#! /usr/bin/env python3

"""
Let's keep this file small, guys. Here are some generaly guidelines:
- Each Dataset class should have below attributes:
    - self.task_mode

"""

import importlib.util
import json
import logging
import os
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
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
from pycocotools.coco import COCO
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


def replace_mask_values(mask, ignore_index):
    return replace_tensor_val(mask, 255, ignore_index).astype(np.int64)


DATA_DIR = os.path.join(get_package_dir(), "data")
IGNORE_INDEX = 0

np.random.seed(42)

PRED_SEG_AUGMENTATION_TRANSFORMS = A.Compose(
    [
        A.Resize(
            height=256,
            width=256,
            interpolation=cv2.INTER_LINEAR,
        ),
        # need to convert from uint8 to float32
        A.Lambda(image=lambda x, **kwargs: x.astype(np.float32) / 255.0),
        # Converts to [C, H, W] after all augmentations
        At.ToTensorV2(),
    ]
)

# transforms for mask and image
SEG_AUGMENTATION_TRANSFORMS = A.Compose(
    [
        A.Resize(
            height=256,
            width=256,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
        ),
        A.HorizontalFlip(p=0.5),
        A.RandomCrop(width=256, height=256),
        A.Affine(
            scale=(0.8, 1.2), translate_percent=(0.2, 0.2), rotate=(-30, 30), p=0.5
        ),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        # normalize will divide pixel values by 255 as well
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.ElasticTransform(p=1.0),
        A.Lambda(
            mask=lambda x, **kwargs: replace_tensor_val(x, 255, IGNORE_INDEX).astype(
                np.int64
            )
        ),
        # need to convert from uint8 to float32
        A.Lambda(image=lambda x, **kwargs: x.astype(np.float32) / 255.0),
        # Converts to [C, H, W] after all augmentations
        At.ToTensorV2(transpose_mask=True),
    ]
)

CLASSIFICATION_AUGMENTATION_TRANSFORMS = A.Compose(
    [
        A.Resize(
            height=256,
            width=256,
            interpolation=cv2.INTER_LINEAR,
        ),
        A.HorizontalFlip(p=0.5),
        A.RandomCrop(width=256, height=256),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        # Comment this out if the dataset is NOT "natural" objects
        # A.Lambda(image=lambda x, **kwargs: x.astype(np.float32) / 255.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        # Converts to [C, H, W] after all augmentations
        At.ToTensorV2(transpose_mask=False),
    ]
)

CLASSIFICATION_PRED_TRANSFORMS = A.Compose(
    [
        A.Resize(
            height=256,
            width=256,
            interpolation=cv2.INTER_LINEAR,
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        # Converts to [C, H, W] after all augmentations
        At.ToTensorV2(transpose_mask=False),
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


class TaskMode(Enum):
    SINGLE_LABEL_IMAGE_CLASSIFICATION = 1
    MULTI_LABEL_IMAGE_CLASSIFICATION = 2
    IMAGE_SEGMENTATION = 3


class PredictDataset(Dataset):
    def __init__(self, images_dir, transforms):
        """We assume images have been in np array and in RGB format"""
        super().__init__()
        image_paths = [
            os.path.join(images_dir, name) for name in os.listdir(images_dir)
        ]
        self.images = [
            np.asarray(Image.open(image_path).convert("RGB"))
            for image_path in image_paths
        ]
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        original_size = image.shape[:2]  # (H, W)
        augmented = self.transforms(image=image)
        augmented = augmented["image"]
        # Returning original size so the images could be scaled back up
        return augmented, original_size, image


class BaseDataset(Dataset):
    """
    Load data -> applies augmentation on masks and images
    """

    def __init__(self, images_dir, labels_dir, task_mode, manual_find_class_num=False):
        self._images_dir = images_dir
        self._labels_dir = labels_dir
        # call this after initializing these variables
        self.images = sorted(os.listdir(self._images_dir))
        self.labels = sorted(os.listdir(self._labels_dir))
        assert len(self.images) == len(
            self.labels
        ), "Number of images and labels should be equal."

        self._max_class = 0 if manual_find_class_num else None
        self.task_mode = task_mode

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

        augmented = SEG_AUGMENTATION_TRANSFORMS(image=image, mask=label)
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
        super().__init__(
            images_dir=images_dir,
            labels_dir=labels_dir,
            task_mode=TaskMode.IMAGE_SEGMENTATION,
        )

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

        super().__init__(
            images_dir=images_dir,
            labels_dir=labels_dir,
            task_mode=TaskMode.IMAGE_SEGMENTATION,
        )

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
        self.task_mode = TaskMode.IMAGE_SEGMENTATION
        print(f"Data {image_set} Successfully Loaded")

    def _is_extracted(self, dataset_dir, year):
        """
        Checking if our data has been extracted
        """
        extracted_train_path = os.path.join(
            dataset_dir,
            # "VOCdevkit",
            "2012_VOCdevkit/" f"VOC{year}",
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

        augmented = SEG_AUGMENTATION_TRANSFORMS(image=image, mask=label)
        image = augmented["image"]
        label = augmented["mask"]
        return image, label

    def __len__(self):
        return len(self._dataset)


def download_and_unzip(url, path, dataset_dir, msg):
    os.makedirs(dataset_dir, exist_ok=True)
    if not os.path.exists(path):
        logging.warning(msg)
        subprocess.run(["wget", url, "-O", path + ".zip"], check=True)
        subprocess.run(["unzip", path + ".zip", "-d", dataset_dir], check=True)
        subprocess.run(["rm", "-rf", path + ".zip"], check=True)


class COCODataset:
    """
    coco/
    ├── images/
    │   ├── train2017/
    │   └── val2017/
    └── annotations/
        ├── instances_train2017.json
        └── instances_val2017.json
    """

    def __init__(
        self,
        split: str,
        task_mode: TaskMode,
    ) -> None:
        """
        stop_at : if not -1, the length of the dataset will be set to the specified value. Defaults to -1.
        """
        SUPPORTED_SPLITS = ("train", "val")
        if split not in SUPPORTED_SPLITS:
            raise ValueError(f"Split '{split}' is not in {SUPPORTED_SPLITS}")
        self.split = split
        self.task_mode = task_mode
        self.set_effective_length_if_necessary(-1)
        train_path, val_path, annotations_path = self._download_and_unzip_coco()
        self._load_annotations(train_path, val_path, annotations_path)

    def set_effective_length_if_necessary(self, stop_at: int):
        """
        This function can be called at any time of the program, to expose part of the dataset to the dataloader
        """
        self.stop_at = stop_at

    def _download_and_unzip_coco(self):
        # Download COCO
        dataset_dir = os.path.join(get_package_dir(), DATA_DIR)
        train_images_url = "http://images.cocodataset.org/zips/train2017.zip"
        val_images_url = "http://images.cocodataset.org/zips/val2017.zip"
        annotations_url = (
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        )
        COCO_path = os.path.join(dataset_dir, "coco")
        # These names shall NOT be changed, because they correspond to the names after unzipping
        train_path = os.path.join(COCO_path, "train2017")
        val_path = os.path.join(COCO_path, "val2017")
        annotations_path = os.path.join(COCO_path, "annotations")

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_train = executor.submit(
                download_and_unzip,
                train_images_url,
                train_path,
                COCO_path,
                "Downloading and extracting train images",
            )
            future_val = executor.submit(
                download_and_unzip,
                val_images_url,
                val_path,
                COCO_path,
                "Downloading and extracting validation images",
            )
            future_annotations = executor.submit(
                download_and_unzip,
                annotations_url,
                annotations_path,
                COCO_path,
                "Downloading and extracting annotation",
            )

            for future in as_completed([future_train, future_val, future_annotations]):
                try:
                    future.result()
                except Exception as e:
                    print(f"Downloading encounters an exception: {e}")

        return train_path, val_path, annotations_path

    def _load_annotations(self, train_path, val_path, annotations_path):
        if self.split == "train":
            annotation_file = os.path.join(annotations_path, "instances_train2017.json")
            images_path = train_path
        else:
            annotation_file = os.path.join(annotations_path, "instances_val2017.json")
            images_path = val_path
        self.coco = COCO(annotation_file)
        self.image_ids = self.coco.getImgIds()
        self.images_path = images_path
        self.cat_ids = self.coco.getCatIds()
        self.cat_ids_2_labels = {cat_id: idx for idx, cat_id in enumerate(self.cat_ids)}
        self.labels_2_cat_ids = {idx: cat_id for idx, cat_id in enumerate(self.cat_ids)}
        self.num_classes = len(self.cat_ids)

        # Extract class names
        with open(annotation_file, "r") as file:
            data = json.load(file)
        categories = data["categories"]
        class_names = [category["name"] for category in categories]
        # (Optional) Sort class names by their category IDs for consistency
        self.class_names = sorted(
            class_names,
            key=lambda x: [cat["id"] for cat in categories if cat["name"] == x][0],
        )

    def __len__(self):
        if self.stop_at == -1:
            return len(self.image_ids)
        else:
            return self.stop_at

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # Load image
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.images_path, img_info["file_name"])
        image = np.asarray(Image.open(img_path).convert("RGB"))
        image = CLASSIFICATION_AUGMENTATION_TRANSFORMS(image=image)["image"]

        if self.task_mode == TaskMode.MULTI_LABEL_IMAGE_CLASSIFICATION:
            target = self._get_item_multi_label_image_classification(anns)
            return image, target
        else:
            raise NotImplementedError(
                f"__getitem()__ has not been implemented for task {self.task_mode} yet 🥺"
            )

    def _get_item_multi_label_image_classification(self, anns) -> torch.Tensor:
        # creating a multi-hot vector
        target = torch.zeros(self.num_classes)
        for ann in anns:
            cat_id = ann["category_id"]
            label_id = self.cat_ids_2_labels[cat_id]
            target[label_id] = 1
        return target


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
        if dataset is None:
            return None
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2,
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


def get_coco_classification_datasets():
    train_dataset = COCODataset(
        split="train", task_mode=TaskMode.MULTI_LABEL_IMAGE_CLASSIFICATION
    )
    val_dataset = COCODataset(
        split="val", task_mode=TaskMode.MULTI_LABEL_IMAGE_CLASSIFICATION
    )
    class_num = train_dataset.num_classes
    return train_dataset, val_dataset, None, class_num
