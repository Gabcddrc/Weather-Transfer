import torch
from generator_attention_module import *
from generator_translation_module import *
from typing import Tuple
import torch.nn as nn
import torch.nn.functional as F


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)



class Generator(nn.Module):
    """
    The Generator class puts together all the building blocks that make it.
    In particular, it combines an attention module and an initial translation module.
    """

    def __init__(self,
                 in_channels_transl: int,
                 block_sizes_transl: list,
                 kernel_size_transl: int,
                 kernel_size_upsampl_transl: int,
                 input_shape_attention: Tuple[int, int, int, int] = (1, 4, 640, 360),
                 attention_module_output_channels: int = 1,
                 kernel_size_blending: Tuple[int, int] = (3, 3),
                 kernel_size_final_filter: Tuple[int, int] = (3, 3),
                 downscale_units_attention: int = 4,
                 attention_units_sequence: int = 3,
                 attention_units_width: int = 2,
                 multi_output = False):
        super().__init__()
        self.multi_output = multi_output
        self.attention_module = AttentionNetwork(in_dim=input_shape_attention,  # (1, 4, 640, 360),
                                                 downscale_units=downscale_units_attention,
                                                 attention_units_sequence=attention_units_sequence,
                                                 attention_units_width=attention_units_width,
                                                 output_channels=attention_module_output_channels)

        self.translation_module = InitialTranslationNetwork(in_channels=in_channels_transl,
                                                            blocks_sizes=block_sizes_transl,
                                                            kernel_size_upsampling=kernel_size_upsampl_transl,
                                                            kernel_size=kernel_size_transl)

        blend_output_channels = 3
        # blends the channels
        self.blend_block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3 + 3 + attention_module_output_channels,
                            out_channels=3,
                            kernel_size=kernel_size_blending,
                            padding="same"),
            torch.nn.BatchNorm2d(blend_output_channels))

        # self.final_filter = torch.nn.Conv2d(in_channels=blend_output_channels,
        #                                     out_channels=3,
        #                                     kernel_size=kernel_size_final_filter,
        #                                     padding="same")
        attention_module_params = sum(p.numel() for p in self.attention_module.parameters())
        translation_module_params = sum(p.numel() for p in self.translation_module.parameters())
        print(f"The attention network has: {attention_module_params:11d} parameters")
        print(f"The translation network has: {translation_module_params:9d} parameters")


        out_features = 64
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(9): # increase from 4
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(3), nn.Conv2d(out_features, 3, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)
    
    
    
    
    def forward(self, X, additional_output=False):
        # pass input through both networks
        attention_map = self.attention_module(X)
        #translation_image = self.translation_module(X)
        translation_image = F.sigmoid(self.translation_module(X))
        # TODO: the two output have different shapes and cannot do pointwise mult (check with tests)
        # TODO: input=(4, 320, 180), attention=(1,1,320,176), transl=(1,4,160,96)

        # do point wise mul and stack with previous output
        point_wise_mul = attention_map * translation_image
        # stacked_output = torch.concat([attention_map, translation_image, point_wise_mul], dim=1)

        # blend that information
        # blend_output = self.blend_block(stacked_output)

        # add the input
        add_output = point_wise_mul + X[:,:3,:,:]

        #out = self.model(add_output)
        out = F.sigmoid(self.model(add_output))

        if self.multi_output:
            return out, attention_map.detach(), translation_image.detach(), point_wise_mul.detach()
        else:
            # default output
            return out
