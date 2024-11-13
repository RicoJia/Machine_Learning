import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data.my_dataset import MyDataset
from torch.utils.data import DataLoader


def test_dog_classifier_conv():
    from src.models import Synth_Classifier

    kernel_size = [(3, 3), (3, 3), (3, 3)]
    stride = [(1, 1), (1, 1), (1, 1)]

    model = Synth_Classifier(kernel_size, stride)

    model_params = model.state_dict()

    model_weight_shapes = []
    model_bias_shapes = []

    for i in model_params.keys():
        if "weight" in i:
            weight_shape = model_params[i].detach().numpy().shape
            model_weight_shapes.append(weight_shape)

            if "0" in i:
                getattr(model, i.split(".")[0])[0].weight.data.fill_(0.1)
            else:
                getattr(model, i.split(".")[0]).weight.data.fill_(0.1)

        elif "bias" in i:
            bias_shape = model_params[i].detach().numpy().shape
            model_bias_shapes.append(bias_shape)

            if "0" in i:
                getattr(model, i.split(".")[0])[0].bias.data.fill_(0)
            else:
                getattr(model, i.split(".")[0]).bias.data.fill_(0)

    true_weight_shapes = [(2, 1, 3, 3), (4, 2, 3, 3), (8, 4, 3, 3), (2, 8)]
    true_bias_shapes = [(2,), (4,), (8,), (2,)]

    input = torch.Tensor(np.ones((1, 28, 28, 1)))

    _est = model.forward(input)
    _est_mean = np.mean(_est.detach().numpy())

    _true_mean = 4.6656017
    print("est_mean: ", _est_mean)
    print("true mean: ", _true_mean)
    #
    # assert np.all(model_weight_shapes == true_weight_shapes)
    # assert np.all(model_bias_shapes == true_bias_shapes)
    # assert np.allclose(_est_mean, _true_mean)


test_dog_classifier_conv()

# class ExampleNet(torch.nn):
#     """
#     This Neural Network does nothing! Woohoo!!!!
#     """
#     def __init__(self):
#         super(ExampleNet, self).__init__()
#
#     def forward(self, x):
#         return x
