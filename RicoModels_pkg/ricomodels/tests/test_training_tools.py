#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from ricomodels.utils.training_tools import *
from torch.utils.data import DataLoader, TensorDataset


# Define a simple feedforward neural network for testing
class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def test_validate_model():
    """
    The purpose of this test is to make sure it can run successfully on the current device
    """
    # Generate some random data for testing
    input_size = 10
    hidden_size = 5
    output_size = 2
    batch_size = 4
    inputs = torch.randn(
        100, input_size
    )  # 100 samples, each with `input_size` features
    labels = torch.randint(
        0, output_size, (100,)
    )  # 100 random labels in [0, output_size)

    # Create a TensorDataset and DataLoader for validation data
    val_dataset = TensorDataset(inputs, labels)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # Set the device to CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Instantiate the model and move it to the correct device
    model = SimpleModel(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size
    )
    model = model.to(device)

    # Define the loss criterion (e.g., CrossEntropyLoss for classification)
    criterion = nn.CrossEntropyLoss()
    val_loss = validate_model(model, val_loader, device, criterion)


def test_early_stopping():
    patience = 5
    early_stopping = EarlyStopping(patience=patience)
    early_stopping(val_loss=1.0)
    early_stopping(val_loss=0.5)
    early_stopping(val_loss=0.4)
    counter = 0
    for i in range(1000):
        counter += 1
        if early_stopping(0.4):
            break
    assert (
        counter == patience
    ), f"Please check early stopping logic. Counter value {counter} should be equal to patience {patience}"


def test_scheduled_probability():
    test_values = [
        # start_p, end_p, d, p
        [0.8, 0.7, 1.0, 0.8],
        [0.8, 0.7, 0.0, 0.7],
    ]
    for start_p, end_p, d, p_expected in test_values:
        p = get_scheduled_probability(start_p, end_p, d)
        assert p == p_expected
