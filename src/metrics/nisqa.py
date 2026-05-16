import torch
from src.metrics.base_metric import BaseMetric


class NISQAMetric(BaseMetric):
    def __init__(self, metric, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = metric.to(device)

    def __call__(self, result, length, **kwargs):
        loss = []
        for i in range(result.shape[0]):
            l = length[i]
            loss.append(self.metric(result[i, 0, :l])[0])
        return torch.stack(loss).mean()