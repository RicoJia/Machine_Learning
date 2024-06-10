#!/usr/bin/env python3
import numpy as np
import scipy.signal
import scipy

def he_init_cnn(out_channels, in_channels, kernel_size):
    # For ReLU, this is for [output_channels, input_channels, kernel, kernel]
    return np.random.randn(out_channels, in_channels, *kernel_size) * np.sqrt(2 / in_channels)

################################
# Optimizer
################################
class SGD:
    def __init__(self, params, criterion, lr=1e-3):
        self.lr = lr
    def step(self,):
        pass
    def zero_grad(self):
        pass

################################
# Activation Functions
################################
class Sigmoid:
    def __call__(self, input: np.ndarray) -> np.ndarray:
        pass

################################
# Cost Functions
################################
class MSELoss:
    def __init__(self) -> None:
        pass

    def __call__(self, *args, **kwds) -> np.ndarray:
        pass

    def forward(input, target) -> np.ndarray:
        pass

################################
# Layer
################################
# TODO: to expand to multiple channels
class Conv2d:
    def __init__(self,in_channels, out_channels, kernel_size, padding=0) -> None:
        # n is the number of inputs, p is the number of outputs
        # nxn, [output_channels, input_channels, kernel, kernel]
        self.weights = he_init_cnn(out_channels=out_channels, in_channels=in_channels, kernel_size=kernel_size)
        self.bias = None
        self.stride = 1
        self.kernel_size = np.asarray(kernel_size)
        self.padding = padding

    def pad_input(self,x):
        if self.padding > 0:
            # Here (0,0) for the first axis as that's the batch dimension ? then (self.padding, self.padding) for the rows, and columns
            return np.pad(x, (((0, 0), (self.padding, self.padding), (self.padding, self.padding))), mode="constant")
        return x

    def __call__(self, x):
        # Forward pass: input [input_channels, height, weight]
        out_channel_num, input_channel_num = self.weights.shape[0], self.weights.shape[1]
        if self.bias is None:
            input_size = np.asarray((x.shape[1], x.shape[2]))
            output_size = ((input_size + self.padding * 2 -self.kernel_size)/self.stride + 1).astype(int)
            output_shape = [out_channel_num, output_size[0], output_size[1]]
            self.bias = np.zeros(output_shape)
            
        print(f'{self.weights.shape}')
        if x.shape[0] != input_channel_num:
            raise ValueError(f'Number of input channel must be {input_channel_num}, but now it is {x.shape[0]}')
        x = self.pad_input(x)
        self.input = x
        #TODO Remember to remove
        print(f'x: {x}')
        self.output = np.copy(self.bias)
        for o in range(out_channel_num):
            for i in range(input_channel_num):
                self.output[o] += scipy.signal.correlate2d(x[i], self.weights[o][i],mode='valid')
        return self.output

    def backward(self, output_gradient):
        if output_gradient.shape != self.output.shape:
            raise ValueError(f"Output Gradient Shape {output_gradient.shape} must be equal to output shape {self.output.shape}")
        out_channel_num, input_channel_num = self.weights.shape[0], self.weights.shape[1]
        self.output_gradient = output_gradient
        self.input_gradient = np.zeros(self.input.shape)
        self.kernel_gradient = np.zeros(self.weights.shape)    #delJ/delK

        for o in range(out_channel_num):
            for i in range(input_channel_num):
                self.kernel_gradient[o, i] = scipy.signal.correlate2d(self.input[i], output_gradient[o], mode="valid")
                #TODO Remember to remove
                print(f'============================================')
                print(f'output_gradient[o]:\n{output_gradient[o]}')
                #TODO Remember to remove
                print(f'self.weights[o][i]:\n{self.weights[o][i]}')
                self.input_gradient[i] += scipy.signal.convolve2d(output_gradient[o], self.weights[o][i],mode="full")

        if self.padding > 0:
            # Just keep the unpadded portion, which is consistent with pytorch
            self.input_gradient = self.input_gradient[:,self.padding:-self.padding, self.padding:-self.padding]
            

class FullyConnectedLinear:
    def __init__(self,in_channels, out_channels) -> None:
        pass
    def __call__(self, x):
        pass

    def grad(self):
        pass

    def reset_parameters(self):
        pass


if __name__ == "__main__":
    pass