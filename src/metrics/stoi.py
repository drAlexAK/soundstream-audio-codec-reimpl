import torch

from src.metrics.base_metric import BaseMetric


class STOIMetric(BaseMetric):
    def __init__(self, metric, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = metric.to(device)

    def __call__(self, result, data_object, length, **kwargs):
        loss = []
        for i in range(result.shape[0]):
            crop_len = min(length[i], result.shape[-1])
            loss.append(
                -self.metric(data_object[i, 0, :crop_len], result[i, 0, :crop_len])
            )
        return torch.stack(loss).mean()
