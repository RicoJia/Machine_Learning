#!/usr/bin/env python3

"""
In this file, we have a collection of utils for visualizations:
- Demo Cost function visualization
"""
import matplotlib.pyplot as plt
import numpy as np

########################################################
# Demo Cost function visualization
########################################################


def relu(z, derivative=False):
    if derivative:
        return np.where(z > 0, 1, 0)
    return np.maximum(0, z)


def tanh(z, derivative=False):
    if derivative:
        return 1 - np.tanh(z) ** 2
    else:
        return np.tanh(z)


def sigmoid(z, derivative=False):
    if derivative:
        s = 1 / (1 + np.exp(-z))
        return s * (1 - s)
    return 1 / (1 + np.exp(-z))


def plot_func(func):
    x = np.linspace(-5, 5, 100)
    y = func(x)
    y_derivative = func(x, derivative=True)
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(x, y_derivative)
    plt.xlabel("x")
    plt.ylabel("y derivative")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_func(sigmoid)
    # plot_func(tanh)
    # plot_func(relu)
