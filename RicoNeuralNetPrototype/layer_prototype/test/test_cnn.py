from RicoNeuralNetPrototype.layer_prototype.cnn import Conv2d
import numpy as np
import torch 

def test_forward_prop():
    conv = Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3), padding=1)
    a = np.array(
        [
            # Batch dimension
            [
                [1, 2, 3], 
                [4, 5, 6], 
                [7, 8, 9]]         
        ], dtype=np.float32
    )
    conv.weights = np.array([
        [
            [
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
            ]
        ]
    ], dtype=np.float32)
    rico_output = conv(a)

    input_tensor = torch.from_numpy(a).requires_grad_(True)
    torch_conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
    with torch.no_grad():
        torch_conv.weight = torch.nn.Parameter(
            torch.from_numpy(conv.weights)
        )
    torch_output = torch_conv(input_tensor) 
    assert np.allclose(rico_output, torch_output.detach().numpy())

    dl_dout = np.array(
        [
            [
                [1,1,1],
                [1,1,1],
                [1,1,1],
            ]
        ], dtype=np.float32
    )
    conv.backward(dl_dout)
    torch_output.backward(torch.from_numpy(dl_dout))
    # grad is also numpy
    print("torch kernel gradient: ", torch_conv.weight.grad)
    print("rico kernel gradient: ", conv.kernel_gradient)
    assert np.allclose(conv.kernel_gradient, torch_conv.weight.grad)
    dL_dInput = input_tensor.grad
    print("torch input gradient: ", dL_dInput)
    print("rico input gradient: ", conv.input_gradient)
    assert np.allclose(dL_dInput, conv.input_gradient)

    # # Test backprop 