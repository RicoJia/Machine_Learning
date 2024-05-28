#!/usr/bin/env python3
from typing import List
import numpy as np
from collections import deque

################################
# Activation Functions
################################
def tanh(z):
    pass

def relu(z, derivative = False):
    if derivative:
        return np.where(z > 0, 1, 0)
    else:
        # np.maximum returns element wise max
        return np.maximum(z, 0)

def binary_cross_entropy(z):
    pass

def rounding_thresholding(z):
    return np.round(z)

def sigmoid(z, derivative = False):
    if derivative:
        s = 1 / (1 + np.exp(-z))
        return s * (1 - s)
    return 1/(1+np.exp(-z))

################################
# Cost Functions
################################
def mean_squared_error(targets: np.ndarray, forward_output: np.ndarray, derivative = False):
    # The final cost is a single value, across all features, and samples sum (y-a) / m
    # targets = [[output vector1], [output vector2] ...], m x p
    if targets.shape != forward_output.shape:
        raise ValueError("Targets must have the same shape as forward_output. "
                         f"Target shape: {targets.shape}, forward output shape: {forward_output.shape}")
    if not derivative:
        return np.mean((targets - forward_output) ** 2)
    else:
        # del J / del a = [a-y]/m, where a is m x n. return value is 1 x n
        return forward_output - targets

################################
# Neural Network Impl
################################
class RicoNeuralNet:
    def __init__(self, io_dimensions:List, learning_rate: float, momentum:float, debugging=False):
        # io_dimensions is something like [2,3,1], where the input dimension is 2, output dimension is 1
        # w : [np.array([]), ...], b: [np.arrays([1,2,3]), ...]
        if len(io_dimensions) < 2:
            raise ValueError("io_dimensions must be 2 or more to build at least one layer")
        self._io_dimensions = io_dimensions
        self._momentum = momentum 
        self._learning_rate = learning_rate
        self._last_change = []
        self._weights = []
        self._biases = []
        # each weight is p*n: [[node1_1, node1_2, node1_3], [node2_1, ...]]. The number of nodes at each layer is determined by 
        # the number of the layer ouptuts
        # Bias: [bias_node_1, bias_node_2, ...]
        if debugging:
            self._weights = [np.ones((io_dimensions[i], io_dimensions[i-1])) for i in range(1, len(self._io_dimensions))]
            self._biases = [np.ones((io_dimensions[i], 1)) for i in range(1, len(self._io_dimensions))]
        else:
            # self._weights = [np.random.rand(io_dimensions[i], io_dimensions[i-1]) for i in range(1, len(self._io_dimensions))]
            # self._biases = [np.random.rand(io_dimensions[i], 1) for i in range(1, len(self._io_dimensions))]
            self._weights = [np.zeros((io_dimensions[i], io_dimensions[i-1])) for i in range(1, len(self._io_dimensions))]
            self._biases = [np.zeros((io_dimensions[i], 1)) for i in range(1, len(self._io_dimensions))]
            # TODO
        self._gradients = [None] * len(io_dimensions)
        # [[inputs], [output_of_layer1], ... [output_of_output layer]]
        # outputs are in the form of [[output vector1], [output vector2] ...]
        self.io_list = []


    def forward(self, inputs, hidden_activation_func = sigmoid, output_activation_func = sigmoid, update_internal_variables = True) -> np.ndarray:
        # returns the outputs given input, 
        # 1. inputs: nice to have: match with the first io_dimensions, get transpose if necessary
        inputs = np.asarray(inputs)
        if update_internal_variables:
            self.io_list.append(inputs)
        a = inputs.transpose()  # n * m, where m is the mini-batch size
        # weights: p x n, where n is input size, p is output size, bias: p*1
        for layer_id, (bias, weights_per_layer) in enumerate(zip(self._biases, self._weights)):
            # p * m
            z = weights_per_layer @ a + bias
            if layer_id != len(self._io_dimensions) -2:
                a = hidden_activation_func(z)
            else:
                if output_activation_func is not None:
                    a = output_activation_func(z)
                else:
                    a = z
            
            if update_internal_variables:
                self.io_list.append(a.transpose())
        # m x n
        return a.transpose()
            

    def backward(self, targets, forward_output, error_func = mean_squared_error, hidden_activation_func = sigmoid, output_activation_func = sigmoid):
        def expand_multiply(del_j_del_z, del_z_del_w):
            M, N = del_z_del_w.shape
            M2, P = del_j_del_z.shape
            if M2 != M:
                raise ValueError("del_j_del_z's number of rows should be double checked")
            # Initialize an empty array for the result with shape (m, p, n)
            del_j_del_w = np.empty((M, P, N))

            # Loop through each row
            for m in range(M):
                # Multiply each element in row i of B with the entire row i of A
                for p in range(P):
                    # print(f'm: {m}, p: {p}, del_j_del_z[m, p]: {del_j_del_z[m, p]}, del_z_del_w[m, :]: {del_z_del_w[m, :]}')
                    del_j_del_w[m, p, :] = del_j_del_z[m, p] * del_z_del_w[m, :]
            
            # print(f'Rico: P (6): {P}, N (2): {N}')
            # M x P x N
            return del_j_del_w

        targets = np.asarray(targets)
        # both targets and forward output are m x p: [[output vector1], [output vector2] ...]
        LAYER_NUM = len(self._io_dimensions) - 1
        # Errors: del_j_del_z
        del_j_del_zs = [None] * LAYER_NUM
        del_j_del_ws = [None] * LAYER_NUM 
        # LAYER_NUM-1 ... 0
        for layer_id in reversed(range(LAYER_NUM)):
            print(f'====================={layer_id}================')
            if layer_id == LAYER_NUM-1:
                # m x p: [[L1_1, L1_2 ...], [batch_2]] each row is jacobian over all nodes of a layer, because del(j)/del(a) is a scalar for a single node 
                del_j_del_a = error_func(targets=targets, forward_output=forward_output, derivative=True)
                # #TODO Remember to remove
                # print(f'targets shape: {targets.shape}')
                # print(f'forward_output shape: {forward_output.shape}')
                print(f'del_j_del_a: {del_j_del_a}')
                #TODO Remember to remove
                print(f'Rico: error: {error_func(targets=targets, forward_output=forward_output, derivative=False)}')

                # m x p: [[L1_1, L1_2 ...], [batch_2]] because del(a)/del(z) is a scalar for a single node
                del_a_del_z = output_activation_func(z=forward_output, derivative=True)
                # print(f'del_a_del_z.shape: {del_a_del_z.shape}')

                # element wise multiplication, m x p
                del_j_del_z = del_j_del_a * del_a_del_z
                # print(f'del_j_del_z: {del_j_del_z}')
                
            else:
                # del_j_del_z^{L} = (del_j_del_z^{L+1} @ weights) elementwise_dot del_a_del_z^{L};
                # del_j_del_z^{L} is mxn, del_j_del_z^{L+1} is mxp, weights^{L+1} is pxn, del_a_del_z^{L}' is mxn,
                # n is the number of inputs at layer L+1, p is the number of outputs at layer L+1. So dimensions are consistent with the output layer case
                del_j_del_z_next_layer = del_j_del_zs[layer_id + 1]  #
                weights_next_layer = self._weights[layer_id + 1]
                output_L = self.io_list[layer_id + 1]
                del_a_del_z_this_layer = hidden_activation_func(z = output_L, derivative = True)

                del_j_del_z = (del_j_del_z_next_layer @ weights_next_layer)* del_a_del_z_this_layer
                #TODO Remember to remove
                # print(f'del_j_del_z_this_layer: {del_j_del_z}')
                # print(f'output_L : {output_L}')
                # print(f'del_a_del_z_this_layer : {del_a_del_z}')
                # print(f'del_j_del_z_this_layer: {del_j_del_z}')

            # m x n: [[x1_1, x1_2 ... x_1_n]], [batch 2] ... ] because del(z)/del(w) are inputs to this layer. And we want to keep
            # those for all batches, so we can sum them up at final update. Each layer shares the same input vector, which means
            # These are inputs to the layer, W_update = each p_per_node * input_vec
            del_z_del_w = self.io_list[layer_id]
            #TODO Remember to remove: 
            # print(f'del_z_del_w: {del_z_del_w}')

            # m x p x n - batch->layer->node, each row is a node's derivative
            del_j_del_w = expand_multiply(del_j_del_z = del_j_del_z, del_z_del_w = del_z_del_w)
            # print(f'del_j_del_w, should be mxpxn: {del_j_del_w}')

            del_j_del_zs[layer_id] = del_j_del_z
            del_j_del_ws[layer_id] = del_j_del_w

        self._update(del_j_del_ws=del_j_del_ws, del_j_del_zs=del_j_del_zs) 

    def _update(self, del_j_del_ws: List[np.ndarray], del_j_del_zs: List[np.ndarray]):
        # print(f'weights: {self._weights}') 
        for weights_per_layer, biases_per_layer, del_j_del_w, del_j_del_z in zip(self._weights, self._biases, del_j_del_ws, del_j_del_zs):
            # del_j_del_w = mxpxn
            total_del_j_del_w = np.mean(del_j_del_w, axis=0)
            weights_per_layer -= self._learning_rate * total_del_j_del_w
            # m x p
            total_del_j_del_z = np.mean(del_j_del_z, axis=0).reshape(-1, 1)
            biases_per_layer -= self._learning_rate * total_del_j_del_z
            # print(f'total_del_j_del_w\n: {total_del_j_del_w}')
            # print(f'total_del_j_del_z\n: {total_del_j_del_z}')

    def predict(self, input):
        # input is 1xn
        output = self.forward(inputs = [input], output_activation_func = None, update_internal_variables=False)
        return output

if __name__ == '__main__':
    rico_neural_net = RicoNeuralNet(io_dimensions = [2,3,1], learning_rate = 0.1, momentum = 0.01)