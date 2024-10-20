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
from tqdm import tqdm
import requests
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from PIL import Image

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


IMAGE_SEG_TRANSFORMS = transforms.Compose([
    v2.Resize((256, 256)),
    # Becareful because you want to rotate your transforms by the same amount
    # v2.RandomHorizontalFlip(),
    # v2.RandomVerticalFlip(),
    # v2.RandomRotation(degrees=15),
    v2.ToTensor(),  # to 0-1
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
TARGET_SEG_TRANSFORMS = transforms.Compose([
    v2.Resize((256, 256), interpolation=InterpolationMode.NEAREST),
    v2.PILToTensor(),
    v2.Lambda(lambda tensor: tensor.squeeze()),
    v2.Lambda(lambda x: replace_tensor_val(
        x.long(), 255, IGNORE_INDEX)),
])


def download_file(url, dest_path, chunk_size=1024):
    """
    dest_path is the zip file path
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for HTTP errors
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = chunk_size  # 1 KB

        # Create the destination directory if it doesn't exist
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, 'wb') as file, tqdm(
            desc=f"Downloading {os.path.basename(dest_path)}",
            total=total_size_in_bytes,
            unit='iB',
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
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            print(f"Extracting {zip_path} to {extract_to}...")
            zip_ref.extractall(extract_to)
        print(f"Extraction completed: {extract_to}\n")
    except zipfile.BadZipFile:
        print(f"Error: {zip_path} is not a zip file or it is corrupted.")
    except Exception as e:
        print(f"An error occurred while extracting {zip_path}: {e}")


class BaseDataset(Dataset):
    def __init__(self, manual_find_class_num=False):
        # call this after initializing these variables
        self.images = sorted(os.listdir(self._images_dir))
        self.labels = sorted(os.listdir(self._labels_dir))
        assert len(self.images) == len(self.labels), "Number of images and labels should be equal."

        self._max_class = 0 if manual_find_class_num else None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # return an image and a label. In this case, a label is an image with int8 values
        img_path = os.path.join(self._images_dir, self.images[idx])
        label_path = os.path.join(self._labels_dir, self.labels[idx])

        # Open image and label
        image = Image.open(img_path).convert('RGB')  # Ensure image is in RGB
        label = Image.open(label_path)

        image = IMAGE_SEG_TRANSFORMS(image)
        label = TARGET_SEG_TRANSFORMS(label)
        label = np.array(label).astype(np.int64)

        if self._max_class is not None:
            unique_values = np.unique(label)
            self._max_class = max(max(unique_values), self._max_class)
            print("max: ", self._max_class)
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
        IMAGES_URL = "http://download.visinf.tu-darmstadt.de/data/from_games/data/01_images.zip"
        LABELS_URL = "http://download.visinf.tu-darmstadt.de/data/from_games/data/01_labels.zip"
        self._images_dir = os.path.join(get_pkg_dir(), DATA_DIR, 'gta5', 'images')
        self._labels_dir = os.path.join(get_pkg_dir(), DATA_DIR, 'gta5', 'labels')
        self.download_and_extract(url=IMAGES_URL, dir_name=self._images_dir, zip_name="01_images.zip")
        self.download_and_extract(url=LABELS_URL, dir_name=self._labels_dir, zip_name="01_labels.zip")

        tasks_args = [
            (IMAGES_URL, self._images_dir, "01_images.zip"),
            (LABELS_URL, self._labels_dir, "01_labels.zip")
        ]

        # Use ThreadPoolExecutor to download and extract concurrently
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit all tasks to the executor
            future_to_task = {
                executor.submit(self.download_and_extract, url, dir_name, zip_name): (url, dir_name, zip_name)
                for url, dir_name, zip_name in tasks_args
            }
            # Process as tasks complete
            for future in as_completed(future_to_task):
                try:
                    future.result()
                except Exception as exc:
                    print(f"An error occurred with {url}: {exc}")
        super().__init__()

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
            raise FileNotFoundError("Carvana dataset can only have 'test' or 'train' sub datasets!")
        self._images_dir = os.path.join(get_pkg_dir(), DATA_DIR, 'carvana', dataset_name)
        self._labels_dir = os.path.join(get_pkg_dir(), DATA_DIR, 'carvana', dataset_name + "_masks")

        super().__init__()

    @cached_property
    def classes(self):
        # 0 = background, 1 = car
        return set([0, 1])


class VOCSegmentationClass(Dataset):
    def __init__(self, image_set, year):
        self._dataset = datasets.VOCSegmentation(
            root=DATA_DIR,
            year=year,
            image_set=image_set,
            download=not self._is_extracted(
                dataset_dir=os.path.join(get_pkg_dir(), DATA_DIR), year=year),
            transform=IMAGE_SEG_TRANSFORMS,
            target_transform=TARGET_SEG_TRANSFORMS
        )
        self._classes = set()
        # TODO Remember to remove
        print(f'Data {image_set} Successfully Loaded')

    def _is_extracted(self, dataset_dir, year):
        """
        Checking if our data has been extracted
        """
        extracted_train_path = os.path.join(
            dataset_dir,
            'VOCdevkit',
            f'VOC{year}',
            'ImageSets',
            'Segmentation',
            'train.txt')
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


if __name__ == '__main__':
    dataset = GTA5Dataset()
