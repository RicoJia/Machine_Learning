#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from sklearn.datasets import make_blobs, make_circles

def generate_spiral_data(n_points, classes):
    X = np.zeros((n_points*classes, 2))
    y = np.zeros(n_points*classes, dtype='uint8')
    for j in range(classes):
        ix = range(n_points*j, n_points*(j+1))
        r = np.linspace(0.0, 1, n_points)
        t = np.linspace(j*4, (j+1)*4, n_points) + np.random.randn(n_points)*0.2
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j
    return X, y

def generate_gaussian_mixtures(n_points, classes):
    X, y = make_blobs(n_samples=n_points, centers=5, n_features=classes, random_state=42)
    return X,y

def generate_circles_within_circles(n_points, classes):
    """Create a circle with number of n_points APPROXIMATELY being n_points"""
    X = []
    y = []
    for i in range(classes):
        X_part, _ = make_circles(n_samples=n_points//classes, noise=0.05, factor=0.2*i)
        X.append(X_part)
        y.append(np.full(n_points//classes, i))
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    return X, y

def partition_data(X: np.ndarray, y: np.ndarray, 
                   training_set=0.7, 
                   holdout_set = 0.2, test_set=0.1) \
    -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not np.allclose(training_set+holdout_set+test_set, 1.0):
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
        n_holdout_set = int(total * holdout_set)

        X_train = X[indices[:n_training_set]]
        Y_train = y[indices[:n_training_set]]
        X_holdout = X[indices[n_training_set: n_training_set + n_holdout_set]]
        y_holdout = y[indices[n_training_set: n_training_set + n_holdout_set]]
        X_test = X[indices[n_training_set + n_holdout_set:]]
        y_test = y[indices[n_training_set + n_holdout_set:]]
        return X_train, Y_train, X_holdout, y_holdout, X_test, y_test
    
def visualize_2D_data(X,y,title, skip_visualize: bool = False):
    plt.figure(figsize=(8, 8))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.8)
    plt.title(title)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.grid(True)
    if not skip_visualize:
        plt.show()