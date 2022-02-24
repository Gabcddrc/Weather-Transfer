# -*- coding: utf-8 -*-
import copy

from network.network_utils import *
from functools import partial


# Many of the following building blocks are taken from:
# https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278
# https://github.com/FrancescoSaverioZuppichini/ResNet/blob/master/ResNet.ipynb

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity()
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut:
            residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, kernel_size, expansion=1, downsampling=(1, 1),
                 *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.expansion, self.downsampling, self.kernel_size = expansion, downsampling, kernel_size
        self.shortcut = nn.Sequential(
            Conv2dAuto(self.in_channels, self.expanded_channels,
                       kernel_size=kernel_size,
                       stride=self.downsampling, bias=False),
            nn.BatchNorm2d(self.expanded_channels)) if self.should_apply_shortcut else None

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


class ResNetBasicBlock(ResNetResidualBlock):
    """
    Basic ResNet block composed by two layers of 3x3conv/batchnorm/activation
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, kernel_size=self.kernel_size, bias=False,
                    stride=self.downsampling, affine_bn=True),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.expanded_channels, kernel_size=self.kernel_size,
                    bias=False, affine_bn=True)
        )


class ResNetBottleNeckBlock(ResNetResidualBlock):
    """
    The three layers are 1x1, 3x3, and 1x1 convolutions, where the 1×1 layers are responsible for reducing and then
    increasing (restoring) dimensions, leaving the 3×3 layer a bottleneck with smaller input/output dimensions.
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, kernel_size=self.kernel_size),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.out_channels, kernel_size=3, stride=self.downsampling),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.expanded_channels, kernel_size=1),
        )


class ResNetLayer(nn.Module):
    """
    A ResNet layer composed by `n` blocks stacked one after the other
    """

    def __init__(self, in_channels, out_channels, n, block=ResNetBasicBlock, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1
        self.blocks = nn.Sequential(
            block(in_channels, out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion,
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by layers with increasing features.
    """

    def __init__(self, in_channels=3, blocks_sizes=[64, 128, 256, 512], deepths=[2, 2, 2, 2],
                 activation='relu', block=ResNetBasicBlock, *args, **kwargs):
        super().__init__()
        self.blocks_sizes = blocks_sizes

        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=(7, 7), stride=(2, 2),
                      padding=3, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation_func(activation),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=deepths[0], activation=activation,
                        block=block, *args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion,
                          out_channels, n=n, activation=activation,
                          block=block, *args, **kwargs)
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]
        ])

    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x


# class ResnetDecoder(nn.Module):
#     """
#     This class represents the tail of ResNet. It performs a global pooling and maps the output to the
#     correct class by using a fully connected layer.
#     """
#
#     def __init__(self, in_features, n_classes):
#         super().__init__()
#         self.avg = nn.AdaptiveAvgPool2d((1, 1))
#         self.decoder = nn.Linear(in_features, n_classes)
#
#     def forward(self, x):
#         x = self.avg(x)
#         x = x.view(x.size(0), -1)
#         x = self.decoder(x)
#         return x


class ResNet(nn.Module):

    def __init__(self, in_channels, blocks_sizes, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, blocks_sizes=blocks_sizes, *args, **kwargs)

    def forward(self, x):
        x = self.encoder(x)
        return x


class InitialTranslationNetwork(nn.Module):
    """
    InitialTranslationNetwork is composed by a downsampling module, followed by a ResNet, followed by an upsampling
    module.
    """

    def __init__(self, in_channels, blocks_sizes: list, kernel_size_upsampling, *args, **kwargs):
        super(InitialTranslationNetwork, self).__init__()
        # Downscale units   # TODO: check if we need a downsampling module other than the one included in ResNet
        # # Same as in the Attention module
        # self.downscale_units = nn.ModuleList()
        # downscale_units_channels = [32 + i * 16 for i in range(downscale_units + 1)]
        # downscale_units_channels[0] = in_channels[1]
        # for i in range(downscale_units):
        #     self.downscale_units.append(DownscaleUnit(downscale_units_channels[i], downscale_units_channels[i + 1], 3))
        OUT_CHANNELS = 3
        # ResNet
        self.resnet = ResNet(in_channels=in_channels, blocks_sizes=blocks_sizes, *args, **kwargs)

        # Upsample units
        reversed_block_sizes = copy.deepcopy(blocks_sizes)
        reversed_block_sizes.reverse()
        reversed_block_sizes.append(32)
        reversed_block_sizes.append(OUT_CHANNELS)

        usample_blocks = []
        for i in range(len(reversed_block_sizes) - 1):
            usample_blocks.append(UpscaleUnit(in_channels=reversed_block_sizes[i], out_channels=reversed_block_sizes[i + 1], kernel_size=kernel_size_upsampling))

        self.upsample_units = nn.Sequential(*usample_blocks)

        # Complete initial translation network
        self.model = nn.Sequential(
            # self.downscale_units,
            self.resnet,
            self.upsample_units
        )

    def forward(self, x):
        x = self.model(x)
        return x
