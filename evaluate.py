import os
import warnings

import comet_ml
import hydra
import torch
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

    experiment.end()


if __name__ == "__main__":
    main()
