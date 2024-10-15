#!/usr/bin/env python3

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_blobs, make_circles


def generate_xor_data(n_points):
    X = np.random.randint(0, 2, (n_points, 2))
    y = np.logical_xor(X[:, 0], X[:, 1]).astype(np.float32)
    # y = np.reshape(y, (-1, 1))
    # Add some noise
    X = (X + np.random.normal(0, 0.1, X.shape)).astype(np.float32)
    return X, y


def generate_spiral_data(n_points, classes):
    X = np.zeros((n_points * classes, 2)).astype(np.float32)
    y = np.zeros(n_points * classes).astype(np.float32)
    for j in range(classes):
        ix = range(n_points * j, n_points * (j + 1))
        r = np.linspace(0.0, 1, n_points)
        t = np.linspace(j * 4, (j + 1) * 4, n_points) + np.random.randn(n_points) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    return X, y


def generate_gaussian_mixtures(n_points, classes):
    X, y = make_blobs(
        n_samples=n_points, centers=classes, n_features=2, random_state=42
    )
    return X.astype(np.float32), y.astype(np.float32)


def generate_circles_within_circles(n_points, classes):
    """Create a circle with number of n_points APPROXIMATELY being n_points"""
    X = []
    y = []
    for i in range(classes):
        X_part, _ = make_circles(
            n_samples=n_points // classes, noise=0.05, factor=0.2 * i
        )
        X.append(X_part)
        y.append(np.full(n_points // classes, i))
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    return X, y


def partition_data(
    X: np.ndarray, y: np.ndarray, training_set=0.7, test_set=0.2, validation_set=0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not np.allclose(training_set + test_set + validation_set, 1.0):
        raise ValueError("The sum of data split proportions must be close to 1.0")
    if len(X) != len(y):
        raise ValueError("X length and y length should be the same")

    # Give indices a nice shuffle
    indices = np.arange(0, len(X))
    np.random.shuffle(indices)
    # Determine the cutting point
    # Split the data
    total = len(X)
    n_training_set = int(total * training_set)
    n_test_set = int(total * test_set)

    X_train = X[indices[:n_training_set]]
    Y_train = y[indices[:n_training_set]]
    X_test = X[indices[n_training_set : n_training_set + n_test_set]]
    y_test = y[indices[n_training_set : n_training_set + n_test_set]]
    X_validation = X[indices[n_training_set + n_test_set :]]
    y_validation = y[indices[n_training_set + n_test_set :]]
    return X_train, Y_train, X_test, y_test, X_validation, y_validation


def to_tensor(*nd_arrays):
    return_ls = []
    for arr in nd_arrays:
        return_ls.append(torch.from_numpy(arr))
    return tuple(return_ls)


def to_one_hot(arr):
    if arr.ndim != 1:
        raise ValueError(f"arr dim should be 1D, but now, its shape is {arr.shape}")
    # from 1D array to 2D
    num_classes = np.max(arr) + 1
    one_hot = np.eye(int(num_classes))[arr.astype(int)].astype(arr.dtype)
    return one_hot


def from_one_hot_to_labels(arr):
    return np.argmax(arr, axis=-1)


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


def visualize_2D_data(X, y, title, skip_visualize: bool = False):
    plt.close()  # to prevent from the previous images to open again
    plt.figure(figsize=(8, 8))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", s=50, alpha=0.8)
    plt.title(title)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.grid(True)
    if not skip_visualize:
        plt.show()


if __name__ == "__main__":
    # X, y = generate_gaussian_mixtures(n_points=20, classes=4)
    X, y = generate_xor_data(n_points=20)
    one_hot = to_one_hot(y)
