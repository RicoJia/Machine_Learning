#!/usr/bin/env python3
import numpy as np
import torch

from RicoNeuralNetPrototype.layer_prototype.cnn import Linear


def torch_forward_backward_test(a, l, dj_dout):
    input_tensor = torch.from_numpy(a).requires_grad_(True)
    output_tensor = l(input_tensor)
    output_tensor.backward(torch.from_numpy(dj_dout))
    return input_tensor, output_tensor


def rico_forward_backward_test(a, l, dj_dout):
    output = l(a)
    input_gradient = l.backward(dj_dout)
    return input_gradient, output


def test_dense():
    a = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.float32)
    batch_num, input_dim, output_dim = a.shape[0], a.shape[1], 1
    dj_dout = np.ones((batch_num, output_dim))
    rico_l = Linear(input_size=input_dim, output_size=output_dim)
    input_gradient, output = rico_forward_backward_test(a=a, l=rico_l, dj_dout=dj_dout)

    l = torch.nn.Linear(input_dim, output_dim)
    l.weight = torch.nn.Parameter(torch.from_numpy(rico_l.weights.copy()).float())
    input_tensor, output_tensor = torch_forward_backward_test(a=a, l=l, dj_dout=dj_dout)

    assert np.allclose(input_tensor.grad.detach().numpy(), input_gradient)
    assert np.allclose(l.weight.grad.detach().numpy(), rico_l.weights_gradient)
