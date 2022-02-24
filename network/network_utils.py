import torch
import torch.nn as nn
from collections import OrderedDict


class UpscaleUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 2, padding: int = 1,
                 output_padding: int = 1, activation=torch.nn.LeakyReLU()):
        """
        upscales the input by factor 2 in resolution, from in_dim, to out_dim. It also applies leaky relu
        # ATTENTION THIS IS NOT GUARANTEED TO UPSCALE BY FACTOR OF TWO
        KERNEL SIZE 3 WITH DEFAULT ARGS WORKS
        """
        super(UpscaleUnit, self).__init__()
        self.conv2dTranspose = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                                                        stride=stride,
                                                        padding=padding, output_padding=output_padding)
        self.activation = activation
        self.norm = torch.nn.BatchNorm2d(out_channels)

    def forward(self, X):
        out = self.conv2dTranspose(X)
        out = self.activation(out)
        out = self.norm(out)

        return out


class DownscaleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        """
        convolution, 2x2 pool, batchnorm

        should in theory scale down b x in_c x w x h to b x out_c x w/2 x h/2
        """
        super(DownscaleUnit, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding="same")
        self.activation = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2)
        self.batch_norm = torch.nn.BatchNorm2d(out_channels)

        # self.conv = Conv2dAuto(in_channels, out_channels, kernel_size, stride=2)
        # self.activation = torch.nn.ReLU()
        # # self.pool = torch.nn.MaxPool2d(2)
        # self.batch_norm = torch.nn.BatchNorm2d(out_channels)

    def forward(self, X):
        out = self.conv(X)
        out = self.activation(out)
        out = self.pool(out)
        out = self.batch_norm(out)

        return out


# Most of the following building blocks are taken from:
# https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278

class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (
            self.kernel_size[0] // 2, self.kernel_size[1] // 2)  # dynamic add padding based on the kernel_size


def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]


# def conv_bn(in_channels, out_channels, kernel_size, bias=False, stride=(1, 1), affine_bn=True):
#     return nn.Sequential(
#         Conv2dAuto(in_channels, out_channels, kernel_size, bias=bias, stride=stride),
#         nn.BatchNorm2d(out_channels, affine=affine_bn)
#     )

def conv_bn(in_channels, out_channels, kernel_size, bias=False, stride=(1, 1), affine_bn=True, *args, **kwargs):
    return nn.Sequential(OrderedDict(
        {'conv': Conv2dAuto(in_channels, out_channels, kernel_size, bias=bias, stride=stride, *args, **kwargs),
         'bn': nn.BatchNorm2d(out_channels, affine=affine_bn)}))
