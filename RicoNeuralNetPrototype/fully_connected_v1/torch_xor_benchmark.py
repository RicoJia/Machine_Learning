"""
How PyTorch Works: AutoGrad. AutoGrad generates a computation graph for a network.
- Each tensor is a node. Each tensor(requires_grad=True) has a "backward()"

    Trivial Example:
    Forward pass:
    x1, x2 -> a = x1*x2 -> y1=log(a), y2=sin(x2) -> w=y1+y2

    backward propapagtion:
    1. dw/dy1 = 1, dw/dy2 = 1
    2. dw/da = d2/dy1 * 1/a, dw/dx2 (intermediate) = dw/dy2 * cos(x2)
    3. dw/dx1 = dw/da * x2, dw/dx2 = dw/da * x1 (from a) + dw/dy2(intermediate)

    So, the computation graph records inputs, outputs at each layer, and the operation done.
    loss.backward() does this, and stores gradients in tensor.grad for the ones with (requires_grad=True)
"""

import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(2, 4)
        self.hidden2 = nn.Linear(4, 4)
        self.output = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

# Sample data (XOR problem)
X = torch.tensor([[0.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 0.0],
                  [1.0, 1.0]])

y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

# Initialize the neural network, loss function, and optimizer
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training the neural network
epochs = 10000
for epoch in range(epochs):
    optimizer.zero_grad()  # Zero the gradient buffers
    outputs = model(X)     # Forward pass
    loss = criterion(outputs, y)  # Compute the loss
    loss.backward()        # Backward pass
    optimizer.step()       # Update the weights
    
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Testing the neural network
print("Testing on sample data:")
with torch.no_grad():
    for x in X:
        output = model(x)
        output = (output > 0.5).float()
        print(f"Input: {x.numpy()}, Output: {output.numpy()}")