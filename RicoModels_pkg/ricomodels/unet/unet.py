#! /usr/bin/env python3

# This is a regular conv block
from torch import nn as nn
from collections import deque
import torch
from torchvision.transforms import v2, CenterCrop

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding = 1,
                bias = False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding = 1,
                bias = False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv_block(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, intermediate_before_max_pool: bool):
        super().__init__()
        # This should include the bottleneck.
        self._layers = nn.ModuleList([ConvBlock(in_channels[i], in_channels[i+1]) for i in range(len(in_channels) - 1)])
        self._maxpool = nn.MaxPool2d(2, stride=2)
        self._intermediate_before_max_pool = intermediate_before_max_pool
    def forward(self, x):
        # returns unpooled output from each block:
        # [intermediate results ... ], but we don't want to return
        intermediate_outputs = deque([])
        for i in range(0, len(self._layers)):
            x1 = self._layers[i](x)
            x = self._maxpool(x1)
            # Add the intermediate result if not the bottom
            if i != len(self._layers)-1:
                intermediate_outputs.appendleft(x1) \
                    if self._intermediate_before_max_pool else intermediate_outputs.appendleft(x)
        return x, intermediate_outputs

class Decoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self._upward_conv_blocks = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels = channels[i], out_channels = channels[i+1],
                kernel_size=2, stride=2
            ) for i in range(len(channels) - 1)
        ])
        # Then, there's a concat step in between
        self._conv_blocks = nn.ModuleList([
            ConvBlock(in_channels= channels[i], out_channels=channels[i+1])
            for i in range(len(channels) - 1)
        ])

    def forward(self, x, skip_inputs):
        if len(skip_inputs) != len(self._conv_blocks) or len(skip_inputs) != len(self._upward_conv_blocks):
            raise ValueError("Please check implementation. Block lengths should be the same!",
                             f"skip inputs, blocks inputs, up blocks: {len(skip_inputs), len(self._conv_blocks), len(self._upward_conv_blocks)}")
        # x is smaller than skip inputs, because there's no padding in the conv layers
        for skip_input, up_block, conv_block in zip(skip_inputs, self._upward_conv_blocks, self._conv_blocks):
            x = up_block(x)
            # data is [CHW]
            # TODO: here's a small detail. The paper didn't specify if we want to append or prepend. This might cause trouble
            # We need to crop because of the accumulated information loss from the pooling
            skip_input = self.crop(skip_input=skip_input, x=x)
            x = torch.cat((skip_input, x), 1)
            x = conv_block(x)
        return x
    def crop(self, skip_input, x):
        _, _, H, W = x.shape
        return v2.CenterCrop((H,W))(skip_input)

class UNet(nn.Module):
    def __init__(self, class_num, intermediate_before_max_pool: bool):
        """
        intermediate_before_max_pool: get intermediate before max pooling in encoder. 
        Otherwise, intermdiate results come after each max pooling
        """
        super().__init__()
        encoder_in_channels= [3, 64, 128, 256, 512, 1024]    # bottleneck is 128
        decoder_channels = [1024, 512, 256, 128, 64] #?
        self._encoder = Encoder(in_channels=encoder_in_channels, intermediate_before_max_pool = intermediate_before_max_pool)
        self._decoder = Decoder(channels=decoder_channels)
        # 1x1
        self._head = nn.Conv2d(in_channels=decoder_channels[-1], out_channels=class_num, kernel_size=1)
        self._init_weight()
        self.n_channels = encoder_in_channels[0]

    def forward(self, x):
        _, _, H, W = x.shape
        x, intermediate = self._encoder(x)
        output = self._decoder(x, intermediate)
        # raw logits
        output = self._head(output)
        output = torch.nn.functional.interpolate(output, size=(H,W),  mode='nearest')
        return output

    def _init_weight(self):
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)
    def use_checkpointing(self):
        self._encoder = torch.utils.checkpoint(self._encoder)
        self._decoder = torch.utils.checkpoint(self._decoder)
        self._head = torch.utils.checkpoint(self._head)
        
if __name__ == "__main__":
    # forward_pass_poc()
    image, target = train_dataset[0]
    class_num = len(train_dataset.classes)
    image = image.unsqueeze(0)
    _, _, H, W = image.shape
    enc = Encoder([3, 16, 32, 64], False)
    x, intermediate_outputs = enc.forward(image)
    dec = Decoder(channels=[64, 32, 16])
    # torch.Size([1, 16, 216, 216])
    output = dec(x, intermediate_outputs)
    # 1x1
    head = nn.Conv2d(
        in_channels=16,
        out_channels=class_num,
        kernel_size=1,
    )
    output = head(output)
    output = torch.nn.functional.interpolate(output, size=(H,W),  mode='nearest')
    print(output.shape)