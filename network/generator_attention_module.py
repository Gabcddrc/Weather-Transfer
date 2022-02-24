from network.network_utils import *
from typing import Tuple

class Self_Attention_2d(nn.Module):
    """ Self attention Layer.
    Taken from https://github.com/heykeetae/Self-Attention-GAN/

    only use it with small resolutions since it creates a width x height x widht x height x batch size matrix
    """

    def __init__(self, in_dim):
        super(Self_Attention_2d, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature # B x
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention


class AttentionNetwork(nn.Module):
    def __init__(self, in_dim: Tuple[int, int, int, int], downscale_units: int, attention_units_sequence: int,
                 attention_units_width: int, attention_channel_inputs: int = 0, output_channels: int = 1):
        """
        Input: B x 4 x width x height
        The idea is to downscale.
        then you do attention
        then you upscale back to
        B x A x W x H

        Params:
            in_dim: (B x C x W x H) -- input shape
            downscale_units: int -- number of downscale units will downscale by 2^downscale_units
            attention_channel_inputs: int -- not used
            attention_units_sequence: int -- how many attention units should be stacked after each other (depth)
            attention_units_width: int -- how many attention units should exist in parallel
            output_channels: int -- how many output channels should the output have B x output_channels x W x H
        """
        super(AttentionNetwork, self).__init__()
        self.downscale_units = nn.ModuleList()
        downscale_units_channels = [32 + i * 16 for i in range(downscale_units + 1)]
        downscale_units_channels[0] = in_dim[1]
        for i in range(downscale_units):
            self.downscale_units.append(DownscaleUnit(downscale_units_channels[i], downscale_units_channels[i + 1], 3))

        # 2d array of the attention units
        self.attention_units = nn.ModuleList()
        in_channels = 0
        for i in range(attention_units_sequence):
            self.attention_units.append(nn.ModuleList())
            for j in range(attention_units_width):
                if i == 0:
                    in_channels = downscale_units_channels[-1]
                else:
                    in_channels = in_channels
                self.attention_units[i].append(Self_Attention_2d(in_channels))
            in_channels *= attention_units_width

        # create upsampler
        last_channels = in_channels
        self.upscale_units = nn.ModuleList()
        upscale_units_channels = [downscale_units_channels[-1] + output_channels - i * 16 for i in
                                  range(downscale_units + 1)]
        upscale_units_channels[0] = last_channels
        upscale_units_channels[-1] = output_channels
        for i in range(downscale_units):
            if i == downscale_units - 1:
                self.upscale_units.append(UpscaleUnit(upscale_units_channels[i], upscale_units_channels[i + 1], 3,
                                                      activation=torch.nn.Tanh()))
            else:
                self.upscale_units.append(UpscaleUnit(upscale_units_channels[i], upscale_units_channels[i + 1], 3))

        return

    def forward(self, X):
        out = X
        # downscaling
        for downscale_unit in self.downscale_units:
            out = downscale_unit(out)

        # attention stuff
        for i in range(len(self.attention_units)):
            concat_output = []
            for j in range(len(self.attention_units[i])):
                concat_output.append(self.attention_units[i][j](out)[0])
            out = torch.concat(concat_output, dim=1)

        # upscaling
        for upscale_unit in self.upscale_units:
            out = upscale_unit(out)

        return out

