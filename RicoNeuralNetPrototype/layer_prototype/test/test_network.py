#!/usr/bin/env python3
"""
In this test, we will test forward propagation of a network with MNIST data. 
The main goal is: "what is the network output at each layer"
"""

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
from RicoNeuralNetPrototype.layer_prototype.rico_nn import RicoNNBase, LR, one_step_rico, one_step_torch
from RicoNeuralNetPrototype.layer_prototype.utils import create_mini_batches, load_mnist

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
        self.layer_outputs = []
        for layer in self.layers:
            x = layer(x)
            self.layer_outputs.append(x)
        return x

class TorchFlatten(torch.nn.Module):
    def forward(self, x):
        return torch.flatten(x, 1)

class TorchCNN(torch.nn.Module):
    def __init__(self):
        super(TorchCNN, self).__init__()
        # Needs this for nn parameters
        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0
        )
        self.fc1 = torch.nn.Linear(64 * 5 * 5, 128)
        self.fc2 = torch.nn.Linear(128, 10)

        # This is not the most efficient way to represent layers, because they can be reused.
        self.layers = [
            self.conv1,
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            self.conv2,
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            TorchFlatten(), 
            self.fc1,
            torch.nn.ReLU(),
            self.fc2,
        ]

    def forward(self, x):
        self.layer_outputs = []
        for layer in self.layers:
            x = layer(x)
            self.layer_outputs.append(x) 
        return x

def assign_weights_to_model(model, torch_model):
    # Only RicoCalculationLayer layers can be assigned
    rico_calculation_layers = [
        layer for layer in model.layers if isinstance(layer, RicoCalculationLayer)
    ]
    torch_calculation_layers = [
        layer for layer in torch_model.layers if isinstance(layer, torch.nn.Linear) \
        or isinstance(layer, torch.nn.Conv2d)
    ]
    if len(rico_calculation_layers) != len(torch_calculation_layers):
        raise ValueError(
            "Need rico_calculation_layers to be of the same length as torch model's debug_calculation_layers"
        )
    for layer, torch_layer in zip(
        rico_calculation_layers, torch_calculation_layers
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

########################################################################
# Actual Tests
########################################################################

def batch_setup():
    # Returns [x, y]
    x_train, y_train, x_test, y_test = load_mnist()
    batches = create_mini_batches(x_train, y_train, batch_size=2, for_torch=True)
    return batches[0]

def check_layer_outputs(torch_nn, rico_nn):
    assert len(torch_nn.layer_outputs) == len(rico_nn.layer_outputs)
    for layer_num, (rico_layer_output, torch_layer_output) in enumerate(zip(rico_nn.layer_outputs,torch_nn.layer_outputs)):
        #TODO Remember to remove
        print(f'Layer num: {layer_num}. layer output: {rico_layer_output.shape}, torch layer output: {torch_layer_output.shape}')
        #TODO Remember to remove
        print(f'rico_layer_output: {rico_layer_output}')
        #TODO Remember to remove
        print(f'torch layer output: {torch_layer_output}: ')
        assert np.allclose(rico_layer_output, torch_layer_output.detach().numpy(), rtol=1e-3, atol=1e-2)

def test_network():
    input, target = batch_setup()
    rico_nn = RicoCNN()
    torch_nn = TorchCNN()
    assign_weights_to_model(model=rico_nn, torch_model=torch_nn)

    rico_criterion = MSELoss()
    rico_optimizer = SGD(layers=rico_nn.layers, criterion=rico_criterion, lr=LR)
    torch_criterion = torch.nn.MSELoss()
    torch_optimizer = torch.optim.SGD(torch_nn.parameters(), lr=LR) 

    torch_optimizer.zero_grad()  # Zero the gradient buffers
    print(f'Training Rico Batch')
    one_step_rico(
        input.detach().numpy(),
        rico_nn,
        rico_criterion,
        rico_optimizer,
        target.detach().numpy(),
    )
    print(f'Training Torch Batch')
    one_step_torch(
        inputs=input,
        model=torch_nn,
        criterion=torch_criterion,
        optimizer=torch_optimizer,
        targets=target,
    )
    check_layer_outputs(
        torch_nn=torch_nn, rico_nn=rico_nn 
    )