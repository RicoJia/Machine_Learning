import numpy as np
import torch

from RicoNeuralNetPrototype.layer_prototype.cnn import (SGD, Conv2d, Flatten,
                                                        Linear, MSELoss, ReLU)
from RicoNeuralNetPrototype.layer_prototype.utils import (create_mini_batches,
                                                          load_mnist)

IO_DIMS = [(28*28,128), (128, 64), (64,10)]
LR = 0.01
EPOCH_NUM = 1000

class RicoNN:
    def __init__(self) -> None:
        self.layers = [
            Flatten(),
            Linear(*IO_DIMS[0]),
            ReLU(),
            Linear(*IO_DIMS[1]),
            ReLU(),
            Linear(*IO_DIMS[2]),
        ]

    def __call__(self, x):
        # Forward
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self):
        pass

    def debug_gradient_check(self):
        pass

class TorchNN(torch.nn.Module):
    def __init__(self):
        super(TorchNN, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(*IO_DIMS[0])
        self.fc2 = torch.nn.Linear(*IO_DIMS[1])
        self.fc3 = torch.nn.Linear(*IO_DIMS[2])
    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
def test_torch(x_train, y_train, x_test, y_test):
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32) 
    
    model = TorchNN()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    for epoch in range(EPOCH_NUM):
        batches = create_mini_batches(x_train, y_train, batch_size=60)
        optimizer.zero_grad()  # Zero the gradient buffers
        for inputs, targets in batches:
            outputs = model(inputs)     # Forward pass
            loss = criterion(outputs, targets)  # Compute the loss
            loss.backward()
            optimizer.step()       # Update the weights
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCH_NUM}, Loss: {loss.item():.4f}")

    # Testing the neural network
    print("Testing on sample data:")
    with torch.no_grad(): #TODO: what's this?
        output = model(x_test)
        output = (output > 0.5).float()
        _, predicted = torch.max(output, 1)
        _, labels = torch.max(y_test, 1)
        
        total = labels.size(0)
        correct = (predicted == labels).sum().item()

        accuracy = correct/total
        #TODO Remember to remove
        print(f'accuracy: {accuracy}')

def test_rico_NN(x_train, y_train, x_test, y_test):
    model = RicoNN()
    criterion = MSELoss()
    optimizer = SGD(layers=model.layers, criterion=criterion, lr=LR)
    for epoch in range(EPOCH_NUM):
        batches = create_mini_batches(x_train, y_train, batch_size=60)
        for input, target in batches:
            output = model(input)     # Forward pass
            loss = criterion(output, target)
            optimizer.backward_and_step()
            
        if epoch % 10 == 0:
            output = model(x_test)
            #TODO Remember to remove
            print(f'shape: {output.shape}, labels: {y_test.shape}')
            predicted = np.argmax(output, axis=1)
            labels = np.argmax(y_test, axis=1)
            
            total = labels.shape[0]
            correct = np.sum(predicted == labels)

            accuracy = correct/total
            #TODO Remember to remove
            print(f'accuracy: {accuracy}')

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_mnist()
    test_rico_NN(x_train, y_train, x_test, y_test)