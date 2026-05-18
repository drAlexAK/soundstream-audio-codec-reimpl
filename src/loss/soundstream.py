import torch
import torch.nn.functional as F
import torchaudio
from torch import nn


class AdversarialGeneratorLoss(nn.Module):
    def forward(self, fake_logits, **batch):
        return torch.stack([F.relu(1 - d).mean() for d in fake_logits]).mean()


class AdversarialDiscriminatorLoss(nn.Module):
    def forward(self, real_logits, fake_logits, **batch):
        real_loss = torch.stack([F.relu(1 - r).mean() for r in real_logits]).mean()
        fake_loss = torch.stack([F.relu(1 + f).mean() for f in fake_logits]).mean()
        return real_loss + fake_loss, real_loss, fake_loss


class FeatureMatchingLoss(nn.Module):
    def forward(self, real_features, fake_features, **batch):
        loss = []
        for real_disc_features, fake_disc_features in zip(real_features, fake_features):
            for r, f in zip(real_disc_features, fake_disc_features):
                loss.append((r.detach() - f).abs().mean())
        return torch.stack(loss).mean()


class MultiScaleSpectralLoss(nn.Module):
    def __init__(self, scales, n_mels):
        super().__init__()
        self.eps = 1e-9
        self.alphas = [(s / 2) ** 0.5 for s in scales]
        self.mels = nn.ModuleList(
            [
                torchaudio.transforms.MelSpectrogram(
                    n_fft=s, win_length=s, hop_length=s // 4, n_mels=n_mels, power=1.0
                )
                for s in scales
            ]
        )

    def forward(self, x, x_hat, **batch):
        x = x.squeeze(1)
        x_hat = x_hat.squeeze(1)
        loss = x.new_tensor(0.0)
        for alpha, mel in zip(self.alphas, self.mels):
            real = mel(x)
            fake = mel(x_hat)
            loss = loss + (real - fake).abs().mean()
            loss = (
                loss
                + alpha
                * torch.sqrt(
                    (
                        (torch.log(real + self.eps) - torch.log(fake + self.eps)) ** 2
                    ).sum(dim=1)
                ).mean()
            )
        return loss


class SoundStreamGeneratorLoss(nn.Module):
    def __init__(
        self,
        lambda_adv,
        lambda_feat,
        lambda_rec,
        lambda_commit,
        scales,
        n_mels,
    ):
        super().__init__()
        self.adv = AdversarialGeneratorLoss()
        self.feat = FeatureMatchingLoss()
        self.rec = MultiScaleSpectralLoss(scales, n_mels)
        self.ladv = lambda_adv
        self.lfeat = lambda_feat
        self.lrec = lambda_rec
        self.lcommit = lambda_commit

    def forward(self, x, x_hat, fake_logits, real_features, fake_features, **batch):
        crop_len = min(x.shape[-1], x_hat.shape[-1])
        x = x[..., :crop_len]
        x_hat = x_hat[..., :crop_len]

        cropped_real_features = []
        cropped_fake_features = []

        for real_disc_features, fake_disc_features in zip(real_features, fake_features):
            cur_real_features = []
            cur_fake_features = []
            for r, f in zip(real_disc_features, fake_disc_features):
                slices = tuple(slice(0, min(a, b)) for a, b in zip(r.shape, f.shape))
                cur_real_features.append(r[slices])
                cur_fake_features.append(f[slices])
            cropped_real_features.append(cur_real_features)
            cropped_fake_features.append(cur_fake_features)

        adv = self.adv(fake_logits)
        feat = self.feat(cropped_real_features, cropped_fake_features)
        rec = self.rec(x, x_hat)
        commit = F.mse_loss(batch["encoded"], batch["quantized"].detach())

        return {
            "loss": (
                self.ladv * adv
                + self.lfeat * feat
                + self.lrec * rec
                + self.lcommit * commit
            ),
            "adv_loss": adv,
            "feat_loss": feat,
            "rec_loss": rec,
            "commitment_loss": commit,
        }


class SoundStreamDiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.adv = AdversarialDiscriminatorLoss()

    def forward(self, real_logits, fake_logits, **batch):
        cropped_real_logits = []
        cropped_fake_logits = []

        for r, f in zip(real_logits, fake_logits):
            crop_len = min(r.shape[-1], f.shape[-1])
            cropped_real_logits.append(r[..., :crop_len])
            cropped_fake_logits.append(f[..., :crop_len])

        loss, real_loss, fake_loss = self.adv(cropped_real_logits, cropped_fake_logits)

        return {
            "discriminator_loss": loss,
            "real_loss": real_loss,
            "fake_loss": fake_loss,
        }
