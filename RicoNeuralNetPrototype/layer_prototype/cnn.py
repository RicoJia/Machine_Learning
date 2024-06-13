#!/usr/bin/env python3
from typing import List

import numpy as np
import scipy
import scipy.signal


def he_init_cnn(out_channels, in_channels, kernel_size):
    # For ReLU, this is for [output_channels, input_channels, kernel, kernel]
    return np.random.randn(out_channels, in_channels, *kernel_size) * np.sqrt(
        2 / in_channels
    )


def he_init(shape):
    # For ReLU
    return np.random.randn(*shape) * np.sqrt(2 / shape[0])


################################
# Optimizer
################################
class SGD:
    def __init__(self, layers: List, lr=1e-3):
        self.lr = lr
        self.layers = layers

    def step(
        self,
    ):
        for layer in self.layers:
            layer.weights -= self.lr * layer.weights_gradient
            # sum across batch, image dimensions. Because bias is applied output per channel
            layer.biases -= self.lr * np.sum(layer.output_gradient, axis=(0, 2, 3))


################################
# Activation Functions
################################
class Sigmoid:
    def __call__(self, input: np.ndarray) -> np.ndarray:
        pass


class ReLU:
    def __call__(self, input: np.ndarray) -> np.ndarray:
        # Input: [batch_numer, input_channels, height, weight]
        self.input = input
        return np.maximum(input, 0)

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        # del a / del z
        return output_gradient * (self.input > 0)


################################
# Cost Functions
################################
class MSELoss:
    def __call__(self, output: np.ndarray, target: np.ndarray) -> float:
        self.output = output
        self.target = target
        return np.mean((target - output) ** 2, axis=None)

    def backward(self) -> np.ndarray:
        n = np.prod(self.target.shape)
        return -2 / n * (self.target - self.output)


################################
# Layers
################################
class MaxPool2D:
    def __init__(self, kernel_size, stride=1):
        self.kernel_size = kernel_size
        self.stride = stride

    def __call__(self, input: np.ndarray) -> np.ndarray:
        # Input: [batch_numer, input_channels, height, weight]
        # Output: [batch_numer, input_channels, new_height, new_weight]

        # This is output shape of each max_windows
        # [number of row, number of matrices per row; number of rows in matrix, number of elements per row]
        output_height = (input.shape[2] - self.kernel_size[0]) // self.stride + 1
        output_width = (input.shape[3] - self.kernel_size[1]) // self.stride + 1
        output_shape = (
            output_height,
            output_width,
            self.kernel_size[0],
            self.kernel_size[1],
        )
        # This is strides in number of bytes.
        # [number of bytes to switch to next row, number of bytes to switch to next column (a matrix),
        # number of bytes to switch to next row in matrix, number of bytes to switch to next element in matrix,
        # ]
        strides = (
            self.stride * input[0][0].strides[0],
            self.stride * input[0][0].strides[1],
            input[0][0].strides[0],
            input[0][0].strides[1],
        )
        output = np.zeros(
            (input.shape[0], input.shape[1], output_shape[0], output_shape[1])
        )
        self.max_indices = np.zeros_like(input, dtype=bool)
        for b in range(input.shape[0]):
            for i in range(input.shape[1]):
                # number of matrices along each axis
                max_windows = np.lib.stride_tricks.as_strided(
                    input[b, i], shape=output_shape, strides=strides
                )
                output[b, i] = max_windows.max(axis=(2, 3))

                # Find the index of the max in each patch of the image. A patch is h * self.stride h * self.stride + +self.kernel_size[0]
                # Max ID:
                for h in range(output_height):
                    for w in range(output_width):
                        window = input[
                            b,
                            i,
                            h * self.stride : h * self.stride + +self.kernel_size[0],
                            w * self.stride : w * self.stride + self.kernel_size[1],
                        ]
                        # returns the index of the flatten view of the array, then converts max_index back to 2D index
                        max_id = np.unravel_index(
                            np.argmax(window, axis=None), window.shape
                        )
                        self.max_indices[
                            b,
                            i,
                            h * self.stride + max_id[0],
                            w * self.stride + max_id[1],
                        ] = True
        return output

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        input_gradient = np.zeros_like(self.max_indices, dtype=np.float32)
        batch_num, out_channels, output_height, output_width = output_gradient.shape

        for b in range(batch_num):
            for o in range(out_channels):
                for h in range(output_height):
                    for w in range(output_width):
                        window_slices = (
                            slice(b, b + 1),
                            slice(o, o + 1),
                            slice(
                                h * self.stride, h * self.stride + +self.kernel_size[0]
                            ),
                            slice(
                                w * self.stride, w * self.stride + self.kernel_size[1]
                            ),
                        )
                        # max_indices is the same size as the input, which is a larger than the output gradient
                        input_gradient[window_slices] = (
                            self.max_indices[window_slices]
                            * output_gradient[b, o, h, w]
                        )
        return input_gradient


class Flatten:
    def __init__(
        self,
    ):
        pass

    def __call__(self, input: np.ndarray) -> np.ndarray:
        # Input: [batch_numer, input_channels, height, weight]
        self.input_shape = input.shape
        return input.reshape(self.input_shape[0], self.input_shape[1], -1)

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        return output_gradient.reshape(self.input_shape)


class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size, padding=0) -> None:
        # n is the number of inputs, p is the number of outputs
        # nxn, [output_channels, input_channels, kernel, kernel]
        self.weights = he_init_cnn(
            out_channels=out_channels, in_channels=in_channels, kernel_size=kernel_size
        )
        self.biases = np.zeros(out_channels)
        self.stride = 1
        self.kernel_size = np.asarray(kernel_size)
        self.padding = padding

    def pad_input(self, x):
        if self.padding > 0:
            # Here (0,0) for the first axis as that's the batch dimension, then input channel,
            # then (self.padding, self.padding) for the rows, and columns
            return np.pad(
                x,
                (
                    (
                        (0, 0),
                        (0, 0),
                        (self.padding, self.padding),
                        (self.padding, self.padding),
                    )
                ),
                mode="constant",
            )
        return x

    def __call__(self, x):
        # Forward pass: input [batch_numer, input_channels, height, weight]
        out_channel_num, input_channel_num = (
            self.weights.shape[0],
            self.weights.shape[1],
        )
        batch_num = x.shape[0]
        input_image_size = np.asarray((x.shape[2], x.shape[3]))
        output_size = (
            (input_image_size + self.padding * 2 - self.kernel_size) / self.stride + 1
        ).astype(int)
        self.output = np.zeros(
            [batch_num, out_channel_num, output_size[0], output_size[1]]
        )

        if x.shape[1] != input_channel_num:
            raise ValueError(
                f"Number of input channel must be {input_channel_num}, but now it is {x.shape[1]}"
            )
        x = self.pad_input(x)
        self.input = x
        for b in range(batch_num):
            for o in range(out_channel_num):
                for i in range(input_channel_num):
                    self.output[b, o] += scipy.signal.correlate2d(
                        x[b][i], self.weights[o][i], mode="valid"
                    )
                self.output[b, o] += self.biases[o]
        return self.output

    def backward(self, output_gradient):
        if output_gradient.shape != self.output.shape:
            raise ValueError(
                f"Output Gradient Shape {output_gradient.shape} must be equal to output shape {self.output.shape}"
            )
        out_channel_num, input_channel_num = (
            self.weights.shape[0],
            self.weights.shape[1],
        )
        self.output_gradient = output_gradient
        self.input_gradient = np.zeros(self.input.shape)
        # in this case, weights is kernel
        self.weights_gradient = np.zeros(self.weights.shape)  # delJ/delK

        batch_num = self.input.shape[0]
        for b in range(batch_num):
            for o in range(out_channel_num):
                for i in range(input_channel_num):
                    self.weights_gradient[o, i] = scipy.signal.correlate2d(
                        self.input[b, i], output_gradient[b, o], mode="valid"
                    )
                    self.input_gradient[i] += scipy.signal.convolve2d(
                        output_gradient[b, o], self.weights[o][i], mode="full"
                    )

        if self.padding > 0:
            # Just keep the unpadded portion, which is consistent with pytorch
            self.input_gradient = self.input_gradient[
                :, :, self.padding : -self.padding, self.padding : -self.padding
            ]


class Linear:
    def __init__(self, input_size: int, output_size: int) -> None:
        self.weights = he_init((output_size, input_size))
        self.bias = np.zeros((output_size, 1))

    def __call__(self, x):
        # x: [batch, input_size]
        self.input = np.asarray(x).T  # [input_size, batch_size]
        self.z = self.weights @ self.input + self.bias  # [output_size, batch_size]
        return self.z.T

    def backward(self, output_gradient):
        # output_gradient = [batch_size, output_size], dj/dz.
        self.input_gradient = output_gradient @ self.weights
        self.weights_gradient = (
            self.input @ output_gradient
        ).T  # [output_size, input_size]
        self.bias_gradient = np.mean(output_gradient, axis=0)
        return self.input_gradient


if __name__ == "__main__":
    pass
