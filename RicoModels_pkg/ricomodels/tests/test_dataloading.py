#!/usr/bin/env python3

# rm -rf results/ && python3 data_loading.py && mv /tmp/results/ .
# eog results/$(ls results/ | head -n1)
from ricomodels.utils.visualization import visualize_image_target_mask
from ricomodels.utils.data_loading import *

def test_voc():
    # dataset = GTA5Dataset()
    dataset = VOCSegmentationDataset(image_set="train", year="2012")
    for i in range(15):
        image, label = dataset[i]
        img = torch.Tensor(image)
        # visualize_image_target_mask(img, target=None, labels=label)

def test_gta5():
    dataset = GTA5Dataset()
    for i in range(15):
        image, label = dataset[i]
        img = torch.Tensor(image)
        # visualize_image_target_mask(img, target=None, labels=label)

def test_coco():
    dataset = COCODataset(split="val", task_mode=TaskMode.MULTI_LABEL_IMAGE_CLASSIFICATION)
    for i in range(15):
        image, target = dataset[i]
        img = torch.Tensor(image)
        print(image.shape, target.shape)
        # visualize_image_target_mask(img, target=None, labels=None)

