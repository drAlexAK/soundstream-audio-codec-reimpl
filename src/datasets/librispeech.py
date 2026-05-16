import numpy as np
import torch
from tqdm.auto import tqdm

import torchaudio

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class LibriSpeechDataset(BaseDataset):
    def __init__(
            self, split, sample_rate, name, *args, **kwargs
    ):
        self.split = split
        self.sample_rate = sample_rate

        index_path = ROOT_PATH / "data" / "LibriSpeech" / name / "index.json"
        self.index_path = index_path
        self.data_path = ROOT_PATH / "data"
        self.data_path.mkdir(exist_ok=True, parents=True)

        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(split, name)

            index_path.parent.mkdir(exist_ok=True, parents=True)
            write_json(index, str(index_path))

        super().__init__(index, *args, **kwargs)

    def __getitem__(self, index):
        data_dict = self._index[index]
        data_object, sample_rate = torchaudio.load(data_dict["path"])

        if sample_rate != self.sample_rate:
            data_object = torchaudio.functional.resample(
                data_object,
                sample_rate,
                self.sample_rate,
            )

        length = data_object.shape[-1]

        instance_data = {"data_object": data_object, "length": length, "original_sample_rate": data_dict["sample_rate"]}
        instance_data = self.preprocess_data(instance_data)

        return instance_data

    def _create_index(self, split, name):
        """
        Create index for the dataset. The function processes dataset metadata
        and utilizes it to get information dict for each element of
        the dataset.

        Args:
            split (str): split name
            name (str): partition name
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        dataset = torchaudio.datasets.LIBRISPEECH(
            root=self.data_path,
            url=self.split,
            download=True,
        )

        index = []

        for i in tqdm(range(len(dataset))):
            path, sample_rate, text, speaker_id, chapter_id, utterance_id = dataset.get_metadata(i)
            path = self.data_path / "LibriSpeech" / path
            audio_id = f"{speaker_id}-{chapter_id}-{utterance_id}"

            index.append(
                {
                    "path": str(path),
                    "sample_rate": sample_rate,
                    "audio_id": audio_id,
                    "text": text,
                }
            )

        return index

    @staticmethod
    def _assert_index_is_valid(index):
        """
        Check the structure of the index and ensure it satisfies the desired
        conditions.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        for entry in index:
            assert "path" in entry, "Each dataset item should include field 'path'" " - path to audio file."
