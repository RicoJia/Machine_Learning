#!/usr/bin/env python3

import dataclasses
from typing import Any, Callable

import numpy as np


class Tensor:
    def __init__(self, data: Any, required_grad: bool = False):
        if isinstance(data, Tensor):
            self = data
            return
        for t in (int, float):
            if isinstance(data, t):
                data = np.array([data])
                break
        self._data = data.astype(np.float32)
        self._required_grad = required_grad
        # scalar to matrix gradient:
        self._grad = np.zeros_like(data, dtype=np.float32)
        self._prev_nodes = tuple()

    def backward(self):
        self._backward()
        print(f"Look closely: {self}")
        for node in self._prev_nodes:
            node.backward()

    def zero_grad(self):
        self._grad = np.zeros_like(self._grad, dtype=np.float32)

    def _backward(self):
        """
        Default backward propagation function that's used ON THE INPUT LAYER
        Each math operation needs to define their own _backward operation for their parents
        Don't forget that we need to accumulate gradients until zero_grad is called
        """
        pass
        # if self._required_grad:
        #     self._grad = np.ones_like(self._grad)

    def _binary_forward_operation(self, other, binary_op: Callable):
        """Forward operation that needs to be called for binary operations

        Returns:
            Tensor: output tensor
        """
        out = Tensor(
            data=binary_op(self._data, other._data),
            required_grad=(self._required_grad or other._required_grad),
        )
        out._prev_nodes = (self, other)
        return out

    # Element wise product. dC/dA = B
    def __mul__(self, other):
        # Here other already is a tensor, and we have created out.
        # This backward is called by out, but actually acts on the current nodes
        # TODO: Why we can pass this function without self right to out?
        other = Tensor(data=other)
        out = self._binary_forward_operation(
            other=other, binary_op=lambda this_data, other_data: this_data * other_data
        )

        def backward_func():
            if self._required_grad:
                self._grad += other._data * out._grad
            if other._required_grad:
                other._grad += self._data * out._grad

        out._backward = backward_func
        return out

    def __matmul__(self, other):
        other = Tensor(data=other)
        out = self._binary_forward_operation(
            other=other, binary_op=lambda this_data, other_data: this_data @ other_data
        )

        def backward_func():
            if self._required_grad:
                self._grad += out._grad @ other._data.T
            if other._required_grad:
                other._grad += self._data.T @ out._grad

        out._backward = backward_func
        return out

    def __sub__(self, other):
        other = Tensor(data=other)
        out = self._binary_forward_operation(
            other=other, binary_op=lambda this_data, other_data: this_data - other_data
        )

        def backward_func():
            if self._required_grad:
                self._grad += out._grad
            if other._required_grad:
                other._grad -= out._grad

        out._backward = backward_func
        return out

    def __add__(self, other):
        other = Tensor(data=other)
        out = self._binary_forward_operation(
            other=other, binary_op=lambda this_data, other_data: this_data + other_data
        )

        def backward_func():
            if self._required_grad:
                self._grad += out._grad
            if other._required_grad:
                other._grad += out._grad

        out._backward = backward_func
        return out

    def __pow__(self, pow):
        # element wise power
        out = Tensor(data=self._data**pow, required_grad=self._required_grad)

        def backward_func():
            if self._required_grad:
                self._grad += out._grad * pow * (self._data) ** (pow - 1)

        out._backward = backward_func
        out._prev_nodes = (self,)
        return out

    def __repr__(self) -> str:
        return f"data: {self._data}, grad: {self._grad}"


class MSELoss:
    def __init__(self) -> None:
        self._data = None
        self._prev_nodes = tuple()

    def __call__(self, output: Tensor, target: Tensor):
        self._prev_nodes = tuple()
        if not isinstance(output, Tensor) or not isinstance(target, Tensor):
            raise ValueError("Loss must take in tensor as inputs")
        self._data = np.mean((target._data - output._data) ** 2)
        self._model_output = output
        n = np.prod(self._model_output._data.shape)

        def backward_func():
            self._grad = 1.0
            self._model_output._grad = (
                -2.0 / n * (target._data - self._model_output._data)
            )

        self._backward = backward_func

    def backward(self):
        self._backward()
        self._model_output.backward()


# when calculating loss, out grad should be clearly set to 1. In other, it should be 0.
if __name__ == "__main__":

    @dataclasses.dataclass
    class TestInfo:
        k: float
        w1_val: float

    def _test_func(input, output, y, expected_grad):
        loss = MSELoss()
        loss(output=output, target=y)
        loss.backward()  # w1 = 2
        assert np.allclose(input._grad, expected_grad)

    test_info_ls = [
        TestInfo(
            k=0,
            w1_val=1,
        ),
        TestInfo(
            k=-1,
            w1_val=1,
        ),
        TestInfo(k=1, w1_val=-2),
    ]

    # Test suite 1
    true_grad = lambda self: (self.k - self.w1_val) * -2

    def test_y_squared_equals_k(w1_val, k, expected_grad):
        # Test 1: single number tests
        w1 = Tensor(np.array([[w1_val]]), required_grad=True)

        # (w1-y)^2, y is the constant here
        y = Tensor(np.array([k]), required_grad=False)
        _test_func(input=w1, output=w1, y=y, expected_grad=expected_grad)

    # TODO
    # for t in test_info_ls:
    #     test_y_squared_equals_k(w1_val=t.w1_val, k=t.k, expected_grad=true_grad(t))

    # (y- 3*(2w1+3))^2 -> 2 * (y- 3*(2w1+3)) * -6

    # Test suite 2
    true_grad = lambda self: -12 * (self.k - 3 * (2 * self.w1_val + 3))

    def test_y_squared_minus_first_order(w1_val, k, expected_grad):
        w1 = Tensor(np.array([[w1_val]]), required_grad=True)

        w2 = w1 * 2 + 3
        w3 = w2 * 3
        # 1 -> 2 -> 5 -> 15
        # (w1-y)^2, y is the constant here
        y = Tensor(np.array([k]), required_grad=False)
        _test_func(input=w1, output=w3, y=y, expected_grad=expected_grad)

    for t in test_info_ls:
        test_y_squared_minus_first_order(
            w1_val=t.w1_val, k=t.k, expected_grad=true_grad(t)
        )

    # Test Suite 3, 1D test for matrix multiplication, which is INCOMPLETE
    true_grad = lambda self: -12 * (self.k - 3 * (2 * self.w1_val + 3))

    def test_y_squared_minus_first_order(w1_val, k, expected_grad):
        w1 = Tensor(np.array([[w1_val]]), required_grad=True)

        w2 = w1 @ 2 + 3
        w3 = w2 @ 3
        # 1 -> 2 -> 5 -> 15
        # (w1-y)^2, y is the constant here
        y = Tensor(np.array([k]), required_grad=False)
        _test_func(input=w1, output=w3, y=y, expected_grad=expected_grad)

    for t in test_info_ls:
        test_y_squared_minus_first_order(
            w1_val=t.w1_val, k=t.k, expected_grad=true_grad(t)
        )
