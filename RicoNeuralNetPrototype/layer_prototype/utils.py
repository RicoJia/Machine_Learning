#!/usr/bin/env python3

import numpy as np
import torch


def mnist_preprocess(x, y):
    from keras.utils import to_categorical

    # Normalize the data
    x = x / 255.0
    # x = x.reshape(x.shape[0], -1)
    # y = y.reshape(y.shape[0], 1)
    y = to_categorical(y, 10)  # One-hot encode labels
    return x, y


def load_mnist():
    X_TRAIN_FILE, X_TEST_FILE, Y_TRAIN_FILE, Y_TEST_FILE = (
        "x_train.npy",
        "x_test.npy",
        "y_train.npy",
        "y_test.npy",
    )
    try:
        x_train = np.load("data/x_train.npy")
        y_train = np.load("data/y_train.npy")
        x_test = np.load("data/x_test.npy")
        y_test = np.load("data/y_test.npy")
    except FileNotFoundError:
        print(f"Downloading mnist")
        import tensorflow as tf

        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        np.save("data/x_train.npy", x_train)
        np.save("data/y_train.npy", y_train)
        np.save("data/x_test.npy", x_test)
        np.save("data/y_test.npy", y_test)
        print(
            f"Saved mnist data to {X_TEST_FILE, Y_TEST_FILE, X_TRAIN_FILE, Y_TRAIN_FILE}"
        )
    else:
        print(
            f"Loaded mnist data from {X_TEST_FILE, Y_TEST_FILE, X_TRAIN_FILE, Y_TRAIN_FILE}"
        )
    # training set: (60000, 28, 28), (60000, )
    # test set: (10000, 28, 28), (10000, )

    x_train, y_train = mnist_preprocess(x_train, y_train)
    x_test, y_test = mnist_preprocess(x_test, y_test)
    return x_train, y_train, x_test, y_test


def create_mini_batches(x, y, batch_size: int, for_torch: bool = False):
    if for_torch:
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
    total_batch_size = x.shape[0]
    random_sequnce = np.random.permutation(total_batch_size)
    x_shuffled = x[random_sequnce]
    y_shuffled = y[random_sequnce]
    mini_batches = [
        (x_shuffled[i : i + batch_size], y_shuffled[i : i + batch_size])
        for i in range(0, total_batch_size, batch_size)
    ]
    return mini_batches
