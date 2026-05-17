import torch
import torch.nn.functional as F
from torch import nn


class VectorQuantizer(nn.Module):
    def __init__(self, embedding_dim, codebook_size, decay, dead_code_threshold):
        super().__init__()
        self.ndim = embedding_dim
        self.codebook_size = codebook_size
        self.decay = decay
        self.eps = 1e-9
        self.dead_code_threshold = dead_code_threshold

        emb = torch.zeros(codebook_size, embedding_dim)
        self.register_buffer("emb", emb)
        self.register_buffer("frequency", torch.zeros(codebook_size))
        self.register_buffer("embeddings_avg", emb.clone())

    def sample_vectors(self, flat_x, n):
        ids = torch.randint(0, flat_x.shape[0], (n,), device=flat_x.device)
        return flat_x[ids]

    def forward(self, x):
        x_perm = x.permute([0, 2, 1]).contiguous()
        x_t = x_perm.view(-1, self.ndim)

        distances = (
            (x_t**2).sum(dim=1, keepdim=True)
            - 2 * x_t @ self.emb.T
            + (self.emb**2).sum(dim=1)
        )

        codes = distances.argmin(dim=1)
        one_hot = F.one_hot(codes, self.codebook_size).type(x_t.dtype)
        quantized = F.embedding(codes, self.emb)

        if self.training:
            with torch.no_grad():
                frequency = one_hot.sum(dim=0)
                emb_sum = one_hot.T @ x_t

                self.frequency *= self.decay
                self.frequency += frequency * (1 - self.decay)
                self.embeddings_avg *= self.decay
                self.embeddings_avg += emb_sum * (1 - self.decay)

                self.emb = self.embeddings_avg / (
                    self.frequency.unsqueeze(1) + self.eps
                )

                dead_codes = self.frequency < self.dead_code_threshold
                if dead_codes.sum() > 0:
                    new_emb = self.sample_vectors(x_t, dead_codes.sum().item())
                    self.emb[dead_codes] = new_emb.clone()
                    self.embeddings_avg[dead_codes] = new_emb.clone()
                    self.frequency[dead_codes] = self.dead_code_threshold

        quantized = quantized.view_as(x_perm)
        quantized = quantized.permute([0, 2, 1]).contiguous()

        quantized = x + (quantized - x).detach()
        codes = codes.view(x.shape[0], x.shape[2])

        return quantized, codes


class RVQ(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_quantizers,
        codebook_size,
        decay,
        dead_code_threshold,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.dead_code_threshold = dead_code_threshold

        self.quantizers = nn.ModuleList(
            [
                VectorQuantizer(
                    embedding_dim=embedding_dim,
                    codebook_size=codebook_size,
                    decay=decay,
                    dead_code_threshold=dead_code_threshold,
                )
                for i in range(num_quantizers)
            ]
        )

    def forward(self, x):
        quantized = torch.zeros_like(x)
        residual = x
        codes = []

        for quantizer in self.quantizers:
            cur_quantized, cur_codes = quantizer(residual)
            quantized = quantized + cur_quantized
            residual = residual - cur_quantized.detach()
            codes.append(cur_codes)

        codes = torch.stack(codes, dim=1)
        return quantized, codes
