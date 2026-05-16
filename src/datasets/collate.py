import torch
import torch.nn.functional as F

class TrainCollate:
    def __init__(self, crop_time, sample_rate):
        self.crop_len = int(crop_time * sample_rate)

    def crop_and_pad(self, data):
        if data.shape[-1] < self.crop_len:
            data = data.repeat(1, self.crop_len // data.shape[-1] + 1)

        start = torch.randint(0, data.shape[-1] - self.crop_len + 1, (1,)).item()
        return data[:, start:start + self.crop_len]

    def __call__(self, dataset_items):
        result_batch = dict()
        result_batch["data_object"] = torch.stack([self.crop_and_pad(elem["data_object"]) for elem in dataset_items])
        result_batch["length"] = torch.tensor([elem["length"] for elem in dataset_items])
        result_batch["original_sample_rate"] = torch.tensor([elem["original_sample_rate"] for elem in dataset_items])
        return result_batch

class InferenceCollate:
    def __init__(self):
        pass

    def pad(self, data):
        lengths = torch.tensor([x.shape[-1] for x in data])
        max_len = lengths.max().item()
        return [F.pad(x, (0, max_len - x.shape[-1])) for x in data]

    def __call__(self, dataset_items):
        result_batch = dict()
        result_batch["data_object"] = torch.stack(self.pad([elem["data_object"] for elem in dataset_items]))
        result_batch["length"] = torch.tensor([elem["length"] for elem in dataset_items])
        result_batch["original_sample_rate"] = torch.tensor([elem["original_sample_rate"] for elem in dataset_items])
        return result_batch