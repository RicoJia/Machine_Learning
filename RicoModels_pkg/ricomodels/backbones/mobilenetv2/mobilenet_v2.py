#!/usr/bin/env python3

import numpy as np
import torch
import torch.backends.cudnn as cudnn  # CUDA Deep Neural Network Library
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchsummary import summary


class ConvBNReLu6(nn.Sequential):
    """Conv + Batch Normalization + ReLu6
    - groups will separate the kernels into multiple groups, each group sees a portion of the input channels. Outputs are finally
        concatenated together
    - dilation convolution is also used. Dilation means "neighboring kerenl cells' index difference in input",
        so the default value is 1
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        have_relu=True,
    ):
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding=0,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if have_relu:
            layers.append(nn.ReLU6(inplace=True))
        super().__init__(*layers)


def get_dilation_padding(dilation, kernel_size):
    """
    Same convolution for dilated convolution.
    see https://ricojia.github.io/2017/01/07/transpose-convolution/
    """

    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    return (pad_beg, pad_end, pad_beg, pad_end)


class InvertedBottleneckResidualBlock(nn.Module):
    """This is a residual block, but we increase the hidden dimension instead of lowering it.

    We are looking to use this for the first block as well. The first block is a simple residual block
    with expansion_factor (t) = 1. So it's

    (hxwxk --3x3 conv--> h/s x w/s x k --linear 1x1 conv--> h/s x w/s x k' --> batch norm)

    In subesequent layers, we have regular inverted bottole neck residual block, with an extra 1x1 conv2d layer

    (hxwxk --1x1 conv--> hxwxtk --3x3 conv--> h/s x w/s x tk --linear 1x1 conv--> h/s x w/s x k' --> batch norm)

    Notes about MobileNetV2:
    - Depthwise Convolution IS used in the expanded conv-batch-relu part.
    - Skip connection is element-wise addition in skip + output in ResNet and MobileNetV2.
        - Concatenation is used in UNet and Segnet, because their input and output layers have different dimensions.
    """

    def __init__(
        self, in_channels, out_channels, stride, expansion_factor, dilation, have_relu
    ) -> None:
        # first conv layer is in charge of the actual stride
        super().__init__()

        # hidden_dim is the same across the bottleneck
        hidden_dim = int(expansion_factor * in_channels)
        layers = []
        self.use_skip_connection = (in_channels == out_channels) and (stride == 1)
        if expansion_factor != 1:
            layers.append(ConvBNReLu6(in_channels, hidden_dim, kernel_size=1))

        # TODO: why groups is hidden_dim here? THAT'S DEPTH WISE convolution?
        layers += [
            ConvBNReLu6(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=stride,
                dilation=dilation,
                groups=hidden_dim,
                have_relu=have_relu,
            ),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]

        # why *layers, not **layers? **is to unpack dictionary, * is to unpack a sequence?
        self.layers = nn.Sequential(*layers)
        # Have relu?
        self.padding = get_dilation_padding(dilation=dilation, kernel_size=3)

    def forward(self, x):
        """
        Applying shortcut between blocks, NOT between expansions. This is from the MobileNetV2 paper.
        """
        x_original = x
        x = nn.functional.pad(x, self.padding)
        out = self.layers(x)

        if self.use_skip_connection:
            return x_original + out
        else:
            return out


class MobileNetV2(nn.Module):
    """
    The architecture is:
    ConvBNReLu6 -> InvertedBottleneckResidualBlocks -> classifier

    Note:
    - The classifier does NOT have sigmoid at the end. This is trained for multi-label classification,
    abd the sigmoid etc. is handled in the loss.
    """

    def __init__(
        self,
        num_classes=1000,
        output_stride=8,
    ):
        super().__init__()
        input_channel = 32
        self.output_dim = 1280

        # t: expansion factor, c (num of channels), n (repeated times), s (stride) in dilation convolution
        specs = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        self.n_channels = 3
        # stride = 2 is from mobilenet v2
        layers = [
            ConvBNReLu6(
                self.n_channels, input_channel, kernel_size=3, stride=2, have_relu=True
            )
        ]
        dilation = 1
        current_stride = 2
        # TODO: Make sure the ouptuts are the same. What are the input dims here?
        for expansion_factor, output_channel, repeated_times, s in specs:
            previous_dilation = dilation
            if current_stride == output_stride:
                stride = 1
                dilation *= s
            else:
                stride = s
                current_stride *= s
            for i in range(repeated_times):
                if i == 0:
                    layers.append(
                        InvertedBottleneckResidualBlock(
                            in_channels=input_channel,
                            out_channels=output_channel,
                            stride=stride,
                            dilation=previous_dilation,
                            expansion_factor=expansion_factor,
                            have_relu=True,
                        )
                    )
                else:
                    layers.append(
                        InvertedBottleneckResidualBlock(
                            in_channels=input_channel,
                            out_channels=output_channel,
                            stride=1,
                            dilation=dilation,
                            expansion_factor=expansion_factor,
                            have_relu=True,
                        )
                    )
                input_channel = output_channel
        layers.append(
            ConvBNReLu6(
                in_channels=input_channel, out_channels=self.output_dim, kernel_size=1
            )
        )
        self.layers = nn.Sequential(*layers)

        # dropout2D vs dropout?
        self.classifier = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(self.output_dim, num_classes)
        )

        self._init_weight()

    def _init_weight(self):
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                # TODO: why batch norm weight is 1?
                elif isinstance(m, (nn.BatchNorm2d,)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                # TODO: why linear init is 1.0?
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 1.0)
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.layers(x)
        # TODO: ?
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    output_stride = 4
    mobilenet_v2 = MobileNetV2(output_stride=output_stride)
    # this is from torchsummary
    print(mobilenet_v2)
