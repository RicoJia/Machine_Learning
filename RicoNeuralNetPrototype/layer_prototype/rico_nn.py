class RicoNN:
    def __init__(self) -> None:
        self.layers = [
            FullyConnectedLinear(2, 4),
            Sigmoid()
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

if __name__ == '__main__':
#     inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#     targets = np.array([[0], [1], [1], [0]])
#     EPOCH_NUM = 50000

#     model = RicoNN()
#     criterion = MSELoss()
#     optimizer = SGD(model, criterion, lr=0.1)

#     for epoch in range(EPOCH_NUM):
#         optimizer.zero_grad()  # Zero the gradient buffers
#         outputs = model(inputs)     # Forward pass
#         loss = criterion(outputs, targets)  # Compute the loss
#         model.backward()
#         optimizer.step()       # Update the weights
    
#         if (epoch + 1) % 1000 == 0:
#             print(f"Epoch {epoch+1}/{EPOCH_NUM}, Loss: {loss.item():.4f}")

#     # # Testing the neural network
#     # print("Testing on sample data:")
#     # with torch.no_grad(): #TODO: what's this?
#     #     for x in inputs:
#     #         output = model(x)
#     #         output = (output > 0.5).float()
#     #         print(f"Input: {x.numpy()}, Output: {output.numpy()}")