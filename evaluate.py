import os
import warnings

import comet_ml
import hydra
import torch
import torchaudio
from hydra.utils import instantiate

from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed

warnings.filterwarnings("ignore", category=UserWarning)

EXPERIMENT_NAME = "test_clean"


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    set_random_seed(config.inferencer.seed)

    config.inferencer.from_pretrained = "saved/best.pth"
    config.inferencer.save_path = None

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture
    model = instantiate(config.model).to(device)

    # get metrics
    metrics = instantiate(config.metrics)

    inferencer = Inferencer(
        model=model,
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        save_path=None,
        metrics=metrics,
        skip_model_load=False,
    )

    logs = inferencer.run_inference()

    experiment = comet_ml.Experiment(
        project_name=os.environ["COMET_PROJECT_NAME"],
        workspace=os.environ.get("COMET_WORKSPACE"),
    )
    experiment.set_name(EXPERIMENT_NAME)

    for part in logs.keys():
        experiment.log_metrics(
            {"{}: {}".format(part, key): value for key, value in logs[part].items()}
        )

    with torch.no_grad():
        batch = inferencer.process_batch(
            0,
            next(iter(dataloaders["test"])),
            metrics=None,
            part="test",
        )

    mel = torchaudio.transforms.MelSpectrogram(
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        n_mels=64,
        power=1.0,
    ).to(device)

    for i in range(batch["data_object"].shape[0]):
        length = batch["length"][i].item()
        x = batch["data_object"][i, :, :length]
        x_hat = batch["result"][i, :, :length]
        experiment.log_audio(
            audio_data=x.detach().cpu().numpy().T,
            file_name=f"test_source_{i}",
            sample_rate=config.sample_rate,
        )
        experiment.log_audio(
            audio_data=x_hat.detach().cpu().numpy().T,
            file_name=f"test_reconstructed_{i}",
            sample_rate=config.sample_rate,
        )
        experiment.log_image(
            image_data=mel(x).squeeze(0).detach().cpu().numpy(),
            name=f"test_source_mel_{i}",
        )
        experiment.log_image(
            image_data=mel(x_hat).squeeze(0).detach().cpu().numpy(),
            name=f"test_reconstructed_mel_{i}",
        )

    experiment.end()


if __name__ == "__main__":
    main()
