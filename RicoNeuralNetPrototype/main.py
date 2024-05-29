#!/usr/bin/env python3
from collections import deque
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical

# npz files are compressed file for multiple numpy arrays
# npy files are binary files
MODEL_WEIGHTS_FILE = "rico_net_weights.npz"


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


def binary_cross_entropy(
    targets: np.ndarray, forward_output: np.ndarray, derivative=False
):
    p = sigmoid(forward_output)
    if derivative:
        return p - targets
    else:
        # clip the predicted values to avoid log(0) error
        p = np.clip(p, 1e-10, 1e-10)
        return -np.mean(targets * np.log(p) + (1 - targets) * np.log(1 - p))


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
    def __init__(
        self,
        io_dimensions: List,
        learning_rate: float,
        momentum: float,
        activation_func=sigmoid,
        cost_func=mean_squared_error,
        skip_output_activation=False,
    ):
        # io_dimensions is something like [2,3,1], where the input dimension is 2, output dimension is 1
        # w : [np.array([]), ...], b: [np.arrays([1,2,3]), ...]
        if len(io_dimensions) < 2:
            raise ValueError(
                "io_dimensions must be 2 or more to build at least one layer"
            )
        self._io_dimensions = io_dimensions
        self._momentum = momentum
        self._learning_rate = learning_rate
        self._activation_func = activation_func
        self._cost_func = cost_func
        self._skip_output_activation = skip_output_activation

        # each weight is p*n: [[node1_1, node1_2, node1_3], [node2_1, ...]]. The number of nodes at each layer is determined by
        # the number of the layer ouptuts
        # Bias: [bias_node_1, bias_node_2, ...]
        try:
            self.load_model()
        except FileNotFoundError:
            self._weights = [
                he_init((io_dimensions[i], io_dimensions[i - 1]))
                for i in range(1, len(self._io_dimensions))
            ]
            self._biases = [
                bias_init((io_dimensions[i], 1))
                for i in range(1, len(self._io_dimensions))
            ]
            print(f"Weights initialized randomly.")

    def forward(self, inputs):
        # [[x1, x2, x3], [layer2]], n x m
        self.a = [np.asarray(inputs).T]
        self.z = []
        LAYER_NUM = len(self._weights)
        # weights: p x n, p is output size, where n is input size, bias: p*1
        for i, (weights, bias) in enumerate(zip(self._weights, self._biases)):
            self.z.append(weights @ self.a[-1] + bias)  # p * m
            if i == LAYER_NUM - 1:
                self.a.append(self.z[-1])
            else:
                self.a.append(self._activation_func(self.z[-1], derivative=False))
        # returning [[a1], [a2]...], mxp
        return self.a[-1].T

    # Lessons learned: optimize your matrix operations is key. Another thing is if there's a tiny bit of mistake in your math,
    # checking it could be occuluded.
    def backward(self, targets):
        # targets: mxp -> pxm
        targets = np.asarray(targets).T
        del_j_del_a = self._cost_func(targets, self.a[-1], derivative=True)  # pxm
        del_j_del_zs = [
            del_j_del_a * self._activation_func(self.z[-1], derivative=True)
        ]  # elementwise multiplication, pxm
        LAYER_NUM = len(self._weights)
        for l in range(1, LAYER_NUM):
            # weights is the next layer, nxp. del_j_del_a is nxm. We need matrix multiplication so we get the sum of
            # all nodes on the next layer.
            del_j_del_a = self._weights[LAYER_NUM - l].T @ del_j_del_zs[-1]
            # z needs to be on the current layer for sigmoid
            if self._skip_output_activation and l == LAYER_NUM - 1:
                del_j_del_zs.append(del_j_del_a)
            else:
                del_j_del_zs.append(
                    del_j_del_a
                    * self._activation_func(self.z[LAYER_NUM - l - 1], derivative=True)
                )
        del_j_del_zs.reverse()

        # TODO Remember to remove
        print(f"Rico: ======================")
        for l in range(LAYER_NUM):
            # p x m @ m x n
            del_j_del_w = del_j_del_zs[l] @ self.a[l].T / self.a[l].shape[1]
            self._weights[l] -= self._learning_rate * del_j_del_w
            bias_gradient = np.mean(del_j_del_zs[l], axis=1, keepdims=True)
            # keepdims will make sure it's (p,1) array, not a (p, ) array
            self._biases[l] -= self._learning_rate * bias_gradient
            # TODO Remember to remove
            print(f"del_j_del_zs:\n{del_j_del_zs}")
            # TODO Remember to remove
            print(f"w grad: {del_j_del_w}")

    def save_model(self):
        np.savez(MODEL_WEIGHTS_FILE, weights=self._weights, biases=self._biases)
        # TODO Remember to remove
        print(f"Rico: weights and biases are saved to {MODEL_WEIGHTS_FILE}")

    def load_model(self):
        data = np.load(MODEL_WEIGHTS_FILE, allow_pickle=True)
        self._weights = data["weights"]
        self._biases = data["biases"]
        print(f"Weights initialized with pretrained data.")

    def predict(self, input):
        return self.forward(input)


################################
# Test Functions
################################
def plot_images(images, labels, num_rows, num_cols):
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
    for i in range(num_rows * num_cols):
        ax = axes[i // num_cols, i % num_cols]
        ax.imshow(images[i], cmap="gray")
        ax.set_title(f"Label: {labels[i]}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def test_with_xor():
    rico_neural_net = RicoNeuralNet(
        io_dimensions=[2, 3, 1],
        learning_rate=0.1,
        momentum=0.01,
        activation_func=relu,
        cost_func=mean_squared_error,
        skip_output_activation=True,
    )
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([[0], [1], [1], [0]])
    EPOCH_NUM = 50000
    for epoch in range(EPOCH_NUM):
        outputs = rico_neural_net.forward(inputs)
        rico_neural_net.backward(targets)
        if epoch % 10 == 0:
            loss = mean_squared_error(targets, outputs, derivative=False)
            print(f"epoch: {epoch}, loss: {loss}")
    pred = rico_neural_net.predict(inputs=inputs)
    # TODO Remember to remove
    print(f"Final prediction: \n{pred}")


def mnist_preprocess(x, y):
    # Normalize the data
    x = x / 255.0
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], 1)
    y = to_categorical(y, 10)  # One-hot encode labels
    return x, y


def create_mini_batches(x, y, batch_size: int):
    total_batch_size = x.shape[0]
    random_sequnce = np.random.permutation(total_batch_size)
    x_shuffled = x[random_sequnce]
    y_shuffled = y[random_sequnce]
    mini_batches = [
        (x_shuffled[i : i + batch_size], y_shuffled[i : i + batch_size])
        for i in range(0, total_batch_size, batch_size)
    ]
    return mini_batches


def test_with_mnist():
    X_TRAIN_FILE, X_TEST_FILE, Y_TRAIN_FILE, Y_TEST_FILE = (
        "x_train.npy",
        "x_test.npy",
        "y_train.npy",
        "y_test.npy",
    )
    try:
        x_train = np.load("x_train.npy")
        y_train = np.load("y_train.npy")
        x_test = np.load("x_test.npy")
        y_test = np.load("y_test.npy")
    except FileNotFoundError:
        print(f"Downloading mnist")
        import tensorflow as tf

        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        np.save("x_train.npy", x_train)
        np.save("y_train.npy", y_train)
        np.save("x_test.npy", x_test)
        np.save("y_test.npy", y_test)
        # TODO Remember to remove
        print(
            f"Saved mnist data to {X_TEST_FILE, Y_TEST_FILE, X_TRAIN_FILE, Y_TRAIN_FILE}"
        )
    else:
        print(
            f"Loaded mnist data from {X_TEST_FILE, Y_TEST_FILE, X_TRAIN_FILE, Y_TRAIN_FILE}"
        )
    # training set: (60000, 28, 28), (60000, )
    # test set: (10000, 28, 28), (10000, )

    # Plot 16 images from the training set
    # plot_images(x_train, y_train, num_rows=4, num_cols=4)

    x_train, y_train = mnist_preprocess(x_train, y_train)
    x_test, y_test = mnist_preprocess(x_test, y_test)

    # TODO Remember to remove
    print(f"{x_test[:10].shape}")

    TEST_BATCH_SIZE = 50
    EPOCH_NUM = 100
    rico_neural_net = RicoNeuralNet(
        io_dimensions=[784, 128, 64, 10],
        learning_rate=0.005,
        momentum=0.01,
        activation_func=sigmoid,
        cost_func=mean_squared_error,
    )
    for epoch in range(EPOCH_NUM):
        batches = create_mini_batches(x_train, y_train, batch_size=60)
        for inputs, targets in batches:
            outputs = rico_neural_net.forward(inputs)
            rico_neural_net.backward(targets)
        if epoch % 4 == 0:
            loss = binary_cross_entropy(targets, outputs, derivative=False)
            print(f"epoch: {epoch}, loss: {loss}")
    rico_neural_net.save_model()
    pred = np.argmax(rico_neural_net.predict(inputs=x_test[:TEST_BATCH_SIZE]), axis=1)
    # pred = np.argmax(rico_neural_net.predict(inputs=x_train[:TEST_BATCH_SIZE]), axis=1)
    # #TODO Remember to remove
    # print(f'Rico: ')
    print(f"Final prediction: \n{pred}")
    print(f"test labels: \n{np.argmax(y_test[:TEST_BATCH_SIZE], axis=1)}")


if __name__ == "__main__":
    # test_with_xor()
    test_with_mnist()
