#!/usr/bin/env python3
from collections import deque
from typing import List

import numpy as np


################################
# Activation Functions
################################
def tanh(z, derivative=False):
    if derivative:
        return 1 - np.tanh(z) ** 2
    else:
        return np.tanh(z)


def relu(z, derivative=False):
    if derivative:
        return np.where(z > 0, 1, 0)
    else:
        # np.maximum returns element wise max
        return np.maximum(z, 0)


def binary_cross_entropy(z):
    pass


def rounding_thresholding(z):
    return np.round(z)


def sigmoid(z, derivative=False):
    if derivative:
        s = 1 / (1 + np.exp(-z))
        return s * (1 - s)
    return 1 / (1 + np.exp(-z))


################################
# Cost Functions
################################
def mean_squared_error(
    targets: np.ndarray, forward_output: np.ndarray, derivative=False
):
    # The final cost is a single value, across all features, and samples sum (y-a) / m
    # targets = [[output vector1], [output vector2] ...], m x p
    if targets.shape != forward_output.shape:
        raise ValueError(
            "Targets must have the same shape as forward_output. "
            f"Target shape: {targets.shape}, forward output shape: {forward_output.shape}"
        )
    if not derivative:
        return np.mean((targets - forward_output) ** 2)
    else:
        # del J / del a = [a-y]/m, where a is m x n. return value is 1 x n
        return forward_output - targets


################################
# Weight Initialization Functions
################################


def xavier_init(shape):
    return np.random.randn(*shape) * np.sqrt(2 / (shape[0] + shape[1]))


# TODO: what are these init functions?
def he_init(shape):
    return np.random.randn(*shape) * np.sqrt(2 / shape[0])


def bias_init(shape):
    return np.zeros(shape)


################################
# Neural Network Impl
################################
class RicoNeuralNet:
    def __init__(self, io_dimensions: List, learning_rate: float, momentum: float):
        # io_dimensions is something like [2,3,1], where the input dimension is 2, output dimension is 1
        # w : [np.array([]), ...], b: [np.arrays([1,2,3]), ...]
        if len(io_dimensions) < 2:
            raise ValueError(
                "io_dimensions must be 2 or more to build at least one layer"
            )
        self._io_dimensions = io_dimensions
        self._momentum = momentum
        self._learning_rate = learning_rate
        # each weight is p*n: [[node1_1, node1_2, node1_3], [node2_1, ...]]. The number of nodes at each layer is determined by
        # the number of the layer ouptuts
        # Bias: [bias_node_1, bias_node_2, ...]
        self._weights = [
            he_init((io_dimensions[i], io_dimensions[i - 1]))
            for i in range(1, len(self._io_dimensions))
        ]
        self._biases = [
            bias_init((io_dimensions[i], 1)) for i in range(1, len(self._io_dimensions))
        ]

    def forward(self, inputs):
        # [[x1, x2, x3], [layer2]], n x m
        self.a = [np.asarray(inputs).T]
        self.z = []
        # weights: p x n, p is output size, where n is input size, bias: p*1
        for weights, bias in zip(self._weights, self._biases):
            self.z.append(weights @ self.a[-1] + bias)  # p * m
            self.a.append(sigmoid(self.z[-1], derivative=False))
        # returning [[a1], [a2]...], mxp
        return self.a[-1].T

    # Lessons learned: optimize your matrix operations is key. Another thing is if there's a tiny bit of mistake in your math,
    # checking it could be occuluded.
    def backward(self, targets):
        # targets: mxp -> pxm
        targets = np.asarray(targets).T
        del_j_del_a = mean_squared_error(targets, self.a[-1], derivative=True)  # pxm
        del_j_del_zs = [
            del_j_del_a * sigmoid(self.z[-1], derivative=True)
        ]  # elementwise multiplication, pxm
        LAYER_NUM = len(self._weights)
        for l in range(1, LAYER_NUM):
            # weights is the next layer, nxp. del_j_del_a is nxm. We need matrix multiplication so we get the sum of
            # all nodes on the next layer.
            del_j_del_a = self._weights[LAYER_NUM - l].T @ del_j_del_zs[-1]
            # z needs to be on the current layer for sigmoid
            del_j_del_zs.append(
                del_j_del_a * sigmoid(self.z[LAYER_NUM - l - 1], derivative=True)
            )
        del_j_del_zs.reverse()

        for l in range(LAYER_NUM):
            # p x m @ m x n
            del_j_del_w = del_j_del_zs[l] @ self.a[l].T / self.a[l].shape[1]
            self._weights[l] -= self._learning_rate * del_j_del_w
            bias_gradient = np.mean(del_j_del_zs[l], axis=1, keepdims=True)
            #     # keepdims will make sure it's (p,1) array, not a (p, ) array
            self._biases[l] -= self._learning_rate * bias_gradient

    def predict(self, inputs):
        # input is mxn
        return self.forward(inputs=inputs)


if __name__ == "__main__":
    rico_neural_net = RicoNeuralNet(
        io_dimensions=[2, 3, 1], learning_rate=0.1, momentum=0.01
    )
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([[0], [1], [1], [0]])
    epochs = 50000
    for epoch in range(epochs):
        outputs = rico_neural_net.forward(inputs)
        rico_neural_net.backward(targets)
        if epoch % 1000 == 0:
            loss = mean_squared_error(targets, outputs, derivative=False)
            print(f"epoch: {epoch}, loss: {loss}")
    pred = rico_neural_net.predict(inputs=inputs)
    # TODO Remember to remove
    print(f"Final prediction: \n{pred}")
