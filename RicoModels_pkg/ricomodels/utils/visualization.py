#! /usr/bin/env python3

import matplotlib.pyplot as plt
import os
import time

RESULTS_DIR = "/tmp/results/"

def visualize_image_target_mask(image, target=None, labels=None):
    # # See torch.Size([3, 281, 500]) torch.Size([1, 281, 500])
    # # print(image.shape, target.shape)

    plt.subplot(1,3,1)
    # Making channels the last dimension
    plt.imshow(image.permute(1,2,0))
    plt.title('image')

    if target is not None:
        plt.subplot(1,3,2)
        # Making channels the last dimension
        plt.imshow(target)
        plt.title('mask')

    if labels is not None:
        plt.subplot(1,3,3)
        # Making channels the last dimension
        plt.imshow(labels)
        plt.title('labels')

    # See tensor([  0,   1,  15, 255], dtype=torch.uint8)
    plt.show()
    tiempo = int(time.time() * 1000)


    if not os.path.exists(RESULTS_DIR):
        os.mkdir(RESULTS_DIR)

    plt.savefig(RESULTS_DIR + str(tiempo)+".png")
