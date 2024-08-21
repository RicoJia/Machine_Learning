#!/usr/bin/env python3

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
