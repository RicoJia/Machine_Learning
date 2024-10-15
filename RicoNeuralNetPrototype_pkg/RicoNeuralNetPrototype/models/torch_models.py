#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim


# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self, fcn_layers, initialization_func=nn.init.xavier_uniform_):
        super(SimpleNN, self).__init__()
        self.fcn_layers = fcn_layers
        [
            initialization_func(l.weight)
            for l in self.fcn_layers
            if isinstance(l, nn.Linear)
        ]
        # [nn.init.kaiming_uniform_(l.weight) for l in self.fcn_layers if isinstance(l, nn.Linear)]

        # Torch requires each layer to have a name
        for i, l in enumerate(self.fcn_layers):
            setattr(self, f"fcn_layer_{i}", l)

    def forward(self, x):
        for l in self.fcn_layers:
            x = l(x)
        # x must be longer than 1 because it represents probabilities
        # x = torch.softmax(x, dim=1)
        return x


def predict(model, X, use_argmax=False):
    with torch.no_grad():
        output = model(X)
        if use_argmax:
            output = torch.argmax(output, dim=-1)
    return output
