import torch
import torch.nn as nn
import torch.optim as optim

from RicoNeuralNetPrototype.models.torch_models import SimpleNN
from RicoNeuralNetPrototype.utils.debug_tools import (FCN2DDebugger,
                                                      FCNDebuggerConfig)
from RicoNeuralNetPrototype.utils.input_data import (
    create_mini_batches, generate_circles_within_circles,
    generate_gaussian_mixtures, generate_spiral_data, generate_xor_data,
    partition_data, to_tensor, visualize_2D_data)


def test_with_model(X_train, y_train, X_test, y_test, X_validation=None, y_validation=None):
    model = SimpleNN(
        fcn_layers=[
            nn.Linear(2, 4), 
            nn.ReLU(), 
            nn.Linear(4, 1), 
            nn.Sigmoid()
        ]
    )

    loss_func = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
    debugger = FCN2DDebugger(model=model, config=FCNDebuggerConfig(), X_test=X_test, y_test=y_test)
    epochs = 500
    for epoch in range(epochs):
        mini_batches = create_mini_batches(X_train, y_train, batch_size=64) 
        for X_train_mini_batch, y_train_mini_batch in mini_batches:
            optimizer.zero_grad()  # Zero the gradient buffers
            # TODO
            X_train_mini_batch = torch.Tensor([[0,0],[0,1], [1,0], [1,1]])
            y_train_mini_batch = torch.Tensor([0,1,1,0]).view(-1,1)
            outputs = model(X_train_mini_batch)     # Forward pass
            loss = loss_func(outputs, y_train_mini_batch)  # Compute the loss
            loss.backward()
            debugger.record_and_calculate_backward_pass(loss=loss)
            optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
            with torch.no_grad():
                output = model(X_validation)
                loss = loss_func(output, y_validation)  # Compute the loss
                #TODO Remember to remove
                print(f'Validation Loss: {loss}')

    debugger.plot_summary()

if __name__ == "__main__":
    X, y = generate_xor_data(n_points=200)
    X_train, y_train, X_test, y_test, X_validation, y_validation = to_tensor(
        *partition_data(X=X, y=y)
    )
    test_with_model(
        X_train, y_train, X_test, y_test, X_validation, y_validation
    )

    # Testing the neural network
    print("Testing on test data:")