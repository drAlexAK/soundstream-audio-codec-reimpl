from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm


class DiscriminatorBlock(nn.Module):
    def __init__(self, channels, kernel_sizes, strides, groups, neg_slope):
        super().__init__()
        self.pads = []
        self.convs = nn.ModuleList()
        self.relus = nn.ModuleList()
        for i in range(len(kernel_sizes)):
            self.pads.append(kernel_sizes[i] // 2)
            self.convs.append(
                weight_norm(
                    nn.Conv1d(
                        channels[i],
                        channels[i + 1],
                        kernel_sizes[i],
                        strides[i],
                        groups=groups[i],
                    )
                )
            )
            if i != len(kernel_sizes) - 1:
                self.relus.append(nn.LeakyReLU(negative_slope=neg_slope))

    def forward(self, x):
        features = []
        for i in range(len(self.pads)):
            x = F.pad(x, (self.pads[i], self.pads[i]))
            x = self.convs[i](x)
            if i != len(self.pads) - 1:
                x = self.relus[i](x)
                features.append(x)
        return x, features


class Discriminator(nn.Module):
    def __init__(
        self,
        downsamples,
        avg_ker_sizes,
        avg_strides,
        avg_paddings,
        channels,
        kernel_sizes,
        strides,
        groups,
        neg_slope,
    ):
        super().__init__()
        self.dis_blocks = nn.ModuleList()
        self.avg_ker_sizes = avg_ker_sizes
        self.avg_strides = avg_strides
        self.avg_paddings = avg_paddings
        for ds in downsamples:
            self.dis_blocks.append(
                DiscriminatorBlock(channels, kernel_sizes, strides, groups, neg_slope)
            )

    def forward(self, x):
        features = []
        xs = []
        for i in range(len(self.dis_blocks)):
            t, f = self.dis_blocks[i](x)
            xs.append(t)
            features.append(f)
            if i != len(self.dis_blocks) - 1:
                x = F.avg_pool1d(
                    x,
                    kernel_size=self.avg_ker_sizes[i],
                    stride=self.avg_strides[i],
                    padding=self.avg_paddings[i],
                )
        return xs, features
