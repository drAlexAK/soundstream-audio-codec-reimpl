from torch import nn

class Conv1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation)
        self.to_pad = (kernel_size - 1) * dilation

    def forward(self, x):
        x_ = nn.functional.pad(x, (self.to_pad, 0), mode="constant", value=0)
        return self.conv(x_)

class Conv1DTranspose(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation)
        self.to_pad = (kernel_size - 1) * dilation

    def forward(self, x):
        x_ = nn.functional.pad(x, (self.to_pad, 0), mode="constant", value=0)
        return self.conv(x_)

class ResidualUnit(nn.Module):
    def __init__(self, N: int, kernel_sizes: list[int], dilation: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.ELU(),
            Conv1D(N, N, kernel_size=kernel_sizes[0], dilation=dilation),
            nn.ELU(),
            Conv1D(N, N, kernel_size=kernel_sizes[1])
        )

    def forward(self, x):
        return x + self.model(x)