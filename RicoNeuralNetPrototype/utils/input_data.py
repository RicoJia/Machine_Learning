#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles

def generate_spiral_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for j in range(classes):
        ix = range(points*j, points*(j+1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(j*4, (j+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j
    return X, y

def generate_gaussian_mixtures(points, classes):
    X, y = make_blobs(n_samples=points, centers=5, n_features=classes, random_state=42)
    return X,y

def generate_circles_within_circles(points, classes):
    X = []
    y = []
    for i in range(classes):
        X_part, _ = make_circles(n_samples=points//classes, noise=0.05, factor=0.2*i)
        X.append(X_part)
        y.append(np.full(points//classes, i))
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    return X, y

if __name__ == "__main__":
    # Create a scatter plot
    def vis(X,y,title):
        plt.figure(figsize=(8, 8))
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.8)
        plt.title(title)
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.grid(True)
        plt.show()
    X, y = generate_spiral_data(points=200, classes=5)
    # vis(X,y,"Spiral Data Visualization")

    X,y = generate_gaussian_mixtures(200, classes=5)
    vis(X,y,"Gaussian Mixture Visualization")
    
    X,y = generate_circles_within_circles(points=200, classes=3)
    # vis(X,y,"Circles Within Circles Visualization")