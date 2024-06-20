import os
import time

import numpy as np
import torch

from RicoNeuralNetPrototype.layer_prototype.cnn import (
    SGD,
    Conv2d,
    Flatten,
    Linear,
    MaxPool2D,
    MSELoss,
    ReLU,
    RicoCalculationLayer,
)
from RicoNeuralNetPrototype.layer_prototype.utils import create_mini_batches, load_mnist

DENSE_IO_DIMS = [(28 * 28, 128), (128, 64), (64, 10)]
# CNN_IO_DIMS = [(28*28,128), (128, 64), (64,10)]
LR = 0.01
EPOCH_NUM = 9
WEIGHTS_FILE = "rico_nn_weights.npz"


################################################################
# Neural Nets
################################################################
class RicoNNBase:
    def load_model(
        self,
    ):
        if not os.path.exists(WEIGHTS_FILE):
            # TODO Remember to remove
            print(f"No weights loaded, using default initialized weight")
            return
        with np.load(WEIGHTS_FILE, allow_pickle=True) as data:
            weights_ls = data["weights_ls"]
            bias_ls = data["bias_ls"]
            rico_calculation_layers = [
                l for l in self.layers if isinstance(l, RicoCalculationLayer)
            ]
            if not len(weights_ls) == len(bias_ls) == len(rico_calculation_layers):
                raise ValueError(
                    f"Array lengths should be equal: weights_ls - {len(weights_ls)}",
                    f"bias_ls: {len(bias_ls) }"
                    f"rico_calculation_layers: {len(rico_calculation_layers)}",
                )
            for layer, weights, bias in zip(
                rico_calculation_layers, weights_ls, bias_ls
            ):
                layer.weights = weights
                layer.bias = bias
        print(f"Model loaded successfully")

    def save_model(self):
        """Saving weights and bias into a file handle
        # weights, bias
        np.ndarray, np.ndarray
        """
        with open(WEIGHTS_FILE, "wb") as weights_fh:
            weights_ls = [
                layer.weights for layer in self.layers if hasattr(layer, "weights")
            ]
            bias_ls = [layer.bias for layer in self.layers if hasattr(layer, "bias")]
            np.savez(weights_fh, weights_ls=weights_ls, bias_ls=bias_ls)


class RicoDenseNN(RicoNNBase):
    # 1000 epochs, 89.5%
    def __init__(self) -> None:
        self.layers = [
            Flatten(),
            Linear(*DENSE_IO_DIMS[0]),
            ReLU(),
            Linear(*DENSE_IO_DIMS[1]),
            ReLU(),
            Linear(*DENSE_IO_DIMS[2]),
        ]

    def __call__(self, x):
        # Forward
        for layer in self.layers:
            x = layer(x)
        return x

    def debug_gradient_check(self):
        pass


class TorchDenseNN(torch.nn.Module):
    def __init__(self):
        super(TorchDenseNN, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(*DENSE_IO_DIMS[0])
        self.fc2 = torch.nn.Linear(*DENSE_IO_DIMS[1])
        self.fc3 = torch.nn.Linear(*DENSE_IO_DIMS[2])

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class RicoCNN(RicoNNBase):
    def __init__(self) -> None:
        self.layers = [
            Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3)),
            ReLU(),
            MaxPool2D(
                kernel_size=(2, 2),
                stride=2,
            ),
            Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)),
            ReLU(),
            MaxPool2D(
                kernel_size=(2, 2),
                stride=2,
            ),
            Flatten(),
            Linear(64 * 5 * 5, 128),
            ReLU(),
            Linear(128, 10),
        ]

    def __call__(self, x):
        # Forward
        for layer in self.layers:
            x = layer(x)
        return x

    def debug_gradient_check(self):
        raise NotImplementedError


class TorchCNN(torch.nn.Module):
    def __init__(self):
        super(TorchCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0
        )
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(64 * 5 * 5, 128)
        self.fc2 = torch.nn.Linear(128, 10)
        self.relu = torch.nn.ReLU()
        self.debug_calculation_layers = [self.conv1, self.conv2, self.fc1, self.fc2]

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 5 * 5)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


################################################################
# Test Benches
################################################################


def test_torch(x_train, y_train, x_test, y_test, model, model_name: str):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    for epoch in range(EPOCH_NUM):
        batches = create_mini_batches(x_train, y_train, batch_size=60, for_torch=True)
        optimizer.zero_grad()  # Zero the gradient buffers
        for inputs, targets in batches:
            inputs = inputs.unsqueeze(1)  # This is probably for cnn only
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, targets)  # Compute the loss
            loss.backward()
            optimizer.step()  # Update the weights
            print(f"Epoch: {epoch}, Loss: {loss.item():.4f}")

    torch.save(model, model_name)
    # Testing the neural network
    print("Testing on sample data:")
    with torch.no_grad():  # TODO: what's this?
        x_test = x_test.unsqueeze(1)  # This is probably for cnn only
        output = model(x_test)
        output = (output > 0.5).float()
        _, predicted = torch.max(output, 1)
        _, labels = torch.max(y_test, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        accuracy = correct / total
        # TODO Remember to remove
        print(f"accuracy: {accuracy}")


def test_rico_NN(x_train, y_train, x_test, y_test, model: RicoNNBase):
    model.load_model()
    criterion = MSELoss()
    optimizer = SGD(layers=model.layers, criterion=criterion, lr=LR)
    for epoch in range(EPOCH_NUM):
        t1 = time.time()
        # TODO Remember to remove
        print(f"Epoch {epoch}")
        batches = create_mini_batches(x_train, y_train, batch_size=60, for_torch=False)
        for i, (input, target) in enumerate(batches):
            output = model(input)  # Forward pass
            loss = criterion(output, target)
            optimizer.backward_and_step()
            print(f"Batch: {i}; Loss: {loss}")
        t2 = time.time()

        model.save_model()
        print(f"Rico: testing")
        output = model(x_test)
        predicted = np.argmax(output, axis=1)
        labels = np.argmax(y_test, axis=1)

        total = labels.shape[0]
        correct = np.sum(predicted == labels)

        accuracy = correct / total
        print(f"accuracy: {accuracy}")
        t3 = time.time()
        # TODO Remember to remove
        print(f"Totol time: {t3-t1}, training time; {t2-t1}")


################################################################
# Test Torch
################################################################


def one_step_rico(input, model, criterion, optimizer, target):
    output = model(input)  # Forward pass
    loss = criterion(output, target)
    optimizer.backward_and_step()
    # TODO Remember to remove
    print(f"Rico Loss: {loss}")


def one_step_torch(inputs, model, criterion, optimizer, targets):
    inputs = inputs.unsqueeze(1)
    outputs = model(inputs)  # Forward pass
    loss = criterion(outputs, targets)  # Compute the loss
    loss.backward()
    optimizer.step()  # Update the weights
    print(f"Torch Loss: {loss.item():.4f}")


def assign_weights_to_model(model, torch_model):
    # Only RicoCalculationLayer layers can be assigned
    if not hasattr(torch_model, "debug_calculation_layers"):
        raise ValueError("Need debug_calculation_layers in torch model")
    rico_calculation_layers = [
        layer for layer in model.layers if isinstance(layer, RicoCalculationLayer)
    ]
    if len(rico_calculation_layers) != len(torch_model.debug_calculation_layers):
        raise ValueError(
            "Need rico_calculation_layers to be of the same length as torch model's debug_calculation_layers"
        )
    for layer, torch_layer in zip(
        rico_calculation_layers, torch_model.debug_calculation_layers
    ):
        # shape is torch.Size
        if layer.weights.shape != torch_layer.weight.shape:
            raise ValueError(
                "weights shapes should be the same"
                f"layer type {type(layer)} weights shape: {layer.weights.shape}, torch layer shape: {torch_layer.weight.shape}"
            )
        else:
            layer.weights = torch_layer.weight.detach().numpy()
        # in rico's fully connected layer, this is [batch_size, 1], whereas in torch, it's [batch_size]
        # TODO: this is kind of ugly,
        bias = (
            torch_layer.bias
            if layer.bias.shape == torch_layer.bias.shape
            else torch_layer.bias.unsqueeze(1)
        )
        if layer.bias.shape != bias.shape:
            raise ValueError(
                "bias shapes should be the same"
                f"layer type {type(layer)} weights shape: {layer.bias.shape}, torch layer shape: {torch_layer.bias.shape}"
            )
        else:
            layer.bias = bias.detach().numpy()


def test_torch_model(x_test, y_test, model):
    print("Testing on sample data:")
    with torch.no_grad():  # TODO: what's this?
        x_test = torch.tensor(x_test, dtype=torch.float32).unsqueeze(
            1
        )  # This is probably for cnn only
        y_test = torch.tensor(y_test, dtype=torch.float32)
        output = model(x_test)
        output = (output > 0.5).float()
        _, predicted = torch.max(output, 1)
        _, labels = torch.max(y_test, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        accuracy = correct / total
        # TODO Remember to remove
        print(f"accuracy: {accuracy}")


def test_rico_NN_against_torch(
    x_train, y_train, x_test, y_test, model: RicoNNBase, torch_model
):
    model.load_model()
    criterion = MSELoss()
    optimizer = SGD(layers=model.layers, criterion=criterion, lr=LR)

    assign_weights_to_model(model=model, torch_model=torch_model)
    torch_criterion = torch.nn.MSELoss()
    torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=LR)
    for epoch in range(EPOCH_NUM):
        torch_optimizer.zero_grad()  # Zero the gradient buffers
        batches = create_mini_batches(x_train, y_train, batch_size=60, for_torch=True)
        for i, (input, target) in enumerate(batches):
            one_step_rico(
                input.detach().numpy(),
                model,
                criterion,
                optimizer,
                target.detach().numpy(),
            )
            one_step_torch(
                inputs=input,
                model=torch_model,
                criterion=torch_criterion,
                optimizer=torch_optimizer,
                targets=target,
            )
    test_torch_model(x_test=x_test, y_test=y_test, model=torch_model)


"""
REPORT
1. loss trend (torch): In first 10 batches, 0.1076, then over 30 batches, steadily down to 0.08
2. Our model should have the same weights array dims as the torch model.
    torch shape before 1st pool: torch.Size([60, 32, 28, 28])
    torch shape after 1st pool: torch.Size([60, 32, 14, 14])
    torch shape before 2nd pool: torch.Size([60, 64, 14, 14])
    torch shape after 2nd pool: torch.Size([60, 64, 7, 7])
"""

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_mnist()
    # test_torch( x_train, y_train, x_test, y_test, model = TorchDenseNN())
    # test_torch( x_train, y_train, x_test, y_test, model = TorchCNN(), model_name="torch_cnn")
    # test_rico_NN(x_train, y_train, x_test, y_test, RicoDenseNN())
    # test_rico_NN(x_train, y_train, x_test, y_test, RicoCNN())

    test_rico_NN_against_torch(
        x_train, y_train, x_test, y_test, model=RicoCNN(), torch_model=TorchCNN()
    )
