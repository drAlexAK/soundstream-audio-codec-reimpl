from torch import nn

from .utils import Conv1D, ResidualUnit


class EncoderBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        S: int,
        kernel_sizes: list[int],
        dilations: list[int],
    ):
        super().__init__()
        self.model = nn.Sequential(
            ResidualUnit(
                input_channels, kernel_sizes=kernel_sizes, dilation=dilations[0]
            ),
            ResidualUnit(
                input_channels, kernel_sizes=kernel_sizes, dilation=dilations[1]
            ),
            ResidualUnit(
                input_channels, kernel_sizes=kernel_sizes, dilation=dilations[2]
            ),
            Conv1D(input_channels, output_channels, kernel_size=2 * S, stride=S),
        )

    def forward(self, x):
        return self.model(x)


class Encoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        channels: list[int],
        strides: list[int],
        kernel_sizes: list[int],
        dilations: list[int],
    ):
        super().__init__()
        self.model = nn.Sequential(
            Conv1D(input_channels, channels[0], kernel_size=kernel_sizes[0]),
            EncoderBlock(
                channels[0],
                channels[1],
                S=strides[0],
                kernel_sizes=kernel_sizes[1:3],
                dilations=dilations,
            ),
            EncoderBlock(
                channels[1],
                channels[2],
                S=strides[1],
                kernel_sizes=kernel_sizes[1:3],
                dilations=dilations,
            ),
            EncoderBlock(
                channels[2],
                channels[3],
                S=strides[2],
                kernel_sizes=kernel_sizes[1:3],
                dilations=dilations,
            ),
            EncoderBlock(
                channels[3],
                channels[4],
                S=strides[3],
                kernel_sizes=kernel_sizes[1:3],
                dilations=dilations,
            ),
            Conv1D(channels[4], output_channels, kernel_size=kernel_sizes[3]),
        )

    def forward(self, x):
        return self.model(x)
