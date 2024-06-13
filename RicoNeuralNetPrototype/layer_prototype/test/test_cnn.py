import numpy as np
import torch

from RicoNeuralNetPrototype.layer_prototype.cnn import (SGD, Conv2d, Flatten,
                                                        MaxPool2D, MSELoss,
                                                        ReLU)

a = np.array(
    [
        # Batch dimension
        [
            # input channel
            [[-1, 2, 3], [-4, 5, 6], [-7, 8, 9]]
        ]
    ],
    dtype=np.float32,
)
dl_dout = np.array(
    [
        [
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
            ]
        ]
    ],
    dtype=np.float32,
)


class TestsRicoNNUnits:
    def test_relu(self):
        l = ReLU()
        output = l(a)
        input_gradient = l.backward(dl_dout)

        torch_relu = torch.nn.ReLU()
        input_tensor = torch.from_numpy(a).requires_grad_(True)
        output_tensor = torch_relu(input_tensor)
        assert np.allclose(output, output_tensor.detach().numpy())
        output_tensor.backward(torch.from_numpy(dl_dout))
        assert np.allclose(input_tensor.grad, input_gradient)

    def test_flatten(self):
        l = Flatten()
        output = l(a)
        input_gradient = l.backward(dl_dout)

        torch_flatten = torch.nn.Flatten()
        input_tensor = torch.from_numpy(a).requires_grad_(True)
        output_tensor = torch_flatten(input_tensor)
        N, C, H, W = dl_dout.shape
        # dl/dout needs to combine all channels
        dl_dout_flatten = torch.from_numpy(dl_dout).reshape(N, -1)
        output_tensor.backward(dl_dout_flatten)
        assert np.allclose(output_tensor.detach().numpy(), output)
        assert np.allclose(input_tensor.grad.detach().numpy(), input_gradient)

    def test_max_pooling_flattening(self):
        x = np.array(
            [
                [
                    [
                        [1, 2, 3, 4, 5],
                        [6, 7, 8, 9, 10],
                        [16, 17, 18, 19, 20],
                        [21, 22, 23, 24, 25],
                        [11, 12, 13, 14, 15],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        l = MaxPool2D(kernel_size=(3, 3), stride=2)
        output = l(x)
        dx_doutput = np.ones_like(output)
        input_gradient = l.backward(dx_doutput)

        m = torch.nn.MaxPool2d((3, 3), stride=(2, 2))
        input_tensor = torch.from_numpy(x).requires_grad_(True)
        output_tensor = m(input_tensor)
        dx_doutput_tensor = torch.from_numpy(dx_doutput)
        output_tensor.backward(dx_doutput_tensor)
        # TODO Remember to remove
        print(f"input tensor grad: {input_tensor.grad}")
        print(f"input gradient:: {input_gradient}")
        assert np.allclose(output, output_tensor.detach().numpy())
        assert np.allclose(input_gradient, input_tensor.grad.detach().numpy())


def forward_prop_test(conv, torch_conv):
    conv.weights = np.array(
        [
            [
                [
                    [1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0],
                ]
            ]
        ],
        dtype=np.float32,
    )
    rico_output = conv(a)

    input_tensor = torch.from_numpy(a).requires_grad_(True)
    with torch.no_grad():
        torch_conv.weight = torch.nn.Parameter(torch.from_numpy(conv.weights.copy()))
        torch_conv.bias = torch.nn.Parameter(
            torch.zeros(torch_conv.out_channels)
        )  # Initialize bias to zero
    torch_output = torch_conv(input_tensor)
    assert np.allclose(rico_output, torch_output.detach().numpy())
    return input_tensor, torch_output


def backward_prop_test(conv, torch_conv, input_tensor, torch_output):
    conv.backward(dl_dout)
    torch_output.backward(torch.from_numpy(dl_dout))
    assert np.allclose(conv.weights_gradient, torch_conv.weight.grad)
    dL_dInput = input_tensor.grad
    assert np.allclose(dL_dInput, conv.input_gradient)


def optimize_test(conv, torch_conv):
    assert np.allclose(conv.weights, torch_conv.weight.detach().numpy())
    assert np.allclose(conv.bias, torch_conv.bias.detach().numpy())

    rico_optimizer = SGD(layers=[conv], lr=0.1)
    rico_optimizer.step()  # Update the weights

    optimizer = torch.optim.SGD(torch_conv.parameters(), lr=0.1)
    optimizer.step()  # Update the weights
    assert np.allclose(conv.weights, torch_conv.weight.detach().numpy())
    assert np.allclose(conv.bias, torch_conv.bias.detach().numpy())


def test_cnn():
    conv = Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), padding=1)
    torch_conv = torch.nn.Conv2d(
        in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=True
    )
    input_tensor, torch_output = forward_prop_test(conv, torch_conv)
    backward_prop_test(conv, torch_conv, input_tensor, torch_output)
    optimize_test(conv, torch_conv)
