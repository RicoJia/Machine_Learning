#!/usr/bin/env python3
from unittest import TestCase
from main import RicoNeuralNet
import numpy as np

IO_DIMENSIONS = [2,16,4,1]
np.set_printoptions(precision=3, suppress=True)
class TestRicoNeuralNet(TestCase):
    def setUp(self):
        self.nn = RicoNeuralNet(io_dimensions = IO_DIMENSIONS , learning_rate = 0.1, momentum = 0.01, debugging=True)

    def test_setup(self):
        i = 1
        for weights_per_layer in self.nn._weights:
            assert weights_per_layer.shape == (IO_DIMENSIONS[i], IO_DIMENSIONS[i-1])
            i+=1
        i = 1
        for biases_per_layer in self.nn._biases:
            assert biases_per_layer.shape == (IO_DIMENSIONS[i], 1)
            i +=1 

    def test_forward_pass(self):
        forward_output = self.nn.forward(
            inputs = [[0,0],[0,1], [1,0], [1,1]],
        )


    def test_backward_pass(self):
        def predict(inputs, targets):
            for input, target in zip(inputs, targets):
                output = self.nn.predict(input)
                #TODO Remember to remove
                print(f'input: {input}, output: {output}, target: {target}')

        self.nn = RicoNeuralNet(io_dimensions = IO_DIMENSIONS , learning_rate = 0.1, momentum = 0.01, debugging=False)
        inputs = [
                [0,0],
                [0,1],
                [1,0],
                [1,1]]
        targets = np.array([[0],
            [1],
            [1],
            [0]])
        for _ in range(10000):
            forward_output = self.nn.forward(inputs=inputs)
            self.nn.backward(targets=targets, forward_output=forward_output)
        # print(f'Post training: =============')
        predict(inputs, targets)