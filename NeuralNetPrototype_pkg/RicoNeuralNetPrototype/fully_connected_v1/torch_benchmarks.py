import functools

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from RicoNeuralNetPrototype.models.torch_models import SimpleNN, predict
from RicoNeuralNetPrototype.utils.debug_tools import FCN2DDebugger, FCNDebuggerConfig
from RicoNeuralNetPrototype.utils.input_data import (
    create_mini_batches,
    generate_circles_within_circles,
    generate_gaussian_mixtures,
    generate_spiral_data,
    generate_xor_data,
    partition_data,
    to_one_hot,
    to_tensor,
    visualize_2D_data,
)


def test_with_model(
    X_train, y_train, X_test, y_test, X_validation=None, y_validation=None
):
    """
    Create a model, train it, and validate its performance.
    All inputs are expected to be in Torch tensors.
    """
    num_classes = np.min(y_test.shape)
    model = SimpleNN(
        fcn_layers=[
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 12),
            nn.ReLU(),
            nn.Linear(12, 4),
            nn.ReLU(),
            nn.Linear(4, num_classes),
            # Along the last dimension
            nn.Softmax(dim=-1),
        ]
    )

    loss_func = nn.MSELoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.02)
    optimizer = optim.Adam(model.parameters(), lr=0.02)

    debugger = FCN2DDebugger(
        model=model,
        config=FCNDebuggerConfig(),
        X_test=X_test,
        y_test=y_test,
        predict_func=functools.partial(predict, model=model),
    )
    epochs = 2000
    for epoch in range(epochs):
        mini_batches = create_mini_batches(X_train, y_train, batch_size=64)
        for X_train_mini_batch, y_train_mini_batch in mini_batches:
            optimizer.zero_grad()  # Zero the gradient buffers
            outputs = model(X_train_mini_batch)  # Forward pass
            loss = loss_func(outputs, y_train_mini_batch)  # Compute the loss
            loss.backward()
            debugger.record_and_calculate_backward_pass(loss=loss)
            optimizer.step()
        if (epoch + 1) % 100 == 0:
            # y here is one hot vector
            outputs = predict(model=model, X=X_validation)
            loss = loss_func(outputs, y_validation)
            print(f"Validation Loss: {loss}")

    debugger.plot_summary()


if __name__ == "__main__":
    # X, y = generate_xor_data(n_points=200)
    X, y = generate_spiral_data(n_points=1000, classes=5)
    # X, y = generate_gaussian_mixtures(1000, classes=5)
    # X, y = generate_circles_within_circles(n_points=200, classes=3)

    y = to_one_hot(y)
    X_train, y_train, X_test, y_test, X_validation, y_validation = to_tensor(
        *partition_data(X=X, y=y)
    )
    test_with_model(X_train, y_train, X_test, y_test, X_validation, y_validation)
