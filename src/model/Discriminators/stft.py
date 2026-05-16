import torch
from torch import nn

class ResidualUnit(nn.Module):
    def __init__(self, N, m, s, kernel_size, neg_slope=0.2):
        super().__init__()
        self.model = nn.Sequential(
            nn.LeakyReLU(negative_slope=neg_slope),
            nn.Conv2d(kernel_size=kernel_size, in_channels=N, out_channels=N, padding=(kernel_size[0] // 2, kernel_size[1] // 2)),
            nn.LeakyReLU(negative_slope=neg_slope),
            nn.Conv2d(kernel_size=(s[0] + 2, s[1] + 2), in_channels=N, out_channels=m * N, stride=s, padding=(s[0] // 2 + 1, s[1] // 2 + 1)),
        )
        self.skip_con = nn.Conv2d(kernel_size=(1, 1), in_channels=N, out_channels=m * N, stride=s)

    def forward(self, x):
        skip = self.skip_con(x)
        x = self.model(x)
        t = min(x.shape[-2], skip.shape[-2])
        f = min(x.shape[-1], skip.shape[-1])

        return x[..., :t, :f] + skip[..., :t, :f]

class STFTDiscriminator(nn.Module):
    def __init__(self, w, h, channels, ms, strides, kernel_pre_size, kernel_in_size, neg_slope=0.2):
        super().__init__()
        self.w = w
        self.h = h
        self.act = nn.LeakyReLU(neg_slope)
        self.conv1 = nn.Conv2d(2, channels[0], kernel_pre_size, padding=(kernel_pre_size[0] // 2, kernel_pre_size[1] // 2))
        self.units = nn.ModuleList()
        for i in range(len(ms)):
            self.units.append(ResidualUnit(channels[i], ms[i], strides[i], kernel_in_size, neg_slope))
        self.conv2 = nn.Conv2d(channels[len(ms)], 1, (1, (w // 2) // (2 ** len(ms))))

    def stft(self, x):
        x = x.squeeze(1)
        window = torch.hann_window(self.w).to(x.device)
        x = torch.stft(
            x,
            n_fft=self.w,
            hop_length=self.h,
            win_length=self.w,
            window=window,
            return_complex=True,
        )

        x = x.transpose(1, 2)
        x = torch.view_as_real(x)
        x = x.permute(0, 3, 1, 2)

        return x[:, :, :, : self.w // 2]

    def forward(self, x):
        features = []
        x = self.act(self.conv1(self.stft(x)))
        features.append(x)
        for unit in self.units:
            x = unit(x)
            features.append(x)
        x = self.conv2(x).squeeze(-1)
        return x, features