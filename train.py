import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="soundstream")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    logger.info(model)

    discriminators = nn.ModuleDict({name: instantiate(disc_config) for name, disc_config in config.discriminator.items()}).to(device)

    logger.info(discriminators)

    # get function handles of losses and metrics
    generator_loss_function = instantiate(config.generator_loss_function).to(device)
    discriminator_loss_function = instantiate(config.discriminator_loss_function).to(device)
    metrics = instantiate(config.metrics)

    # build optimizers, learning rate schedulers
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.generator_optimizer, params=trainable_params)
    lr_scheduler = instantiate(config.generator_lr_scheduler, optimizer=optimizer)

    discriminator_params = filter(lambda p: p.requires_grad, discriminators.parameters())
    discriminator_optimizer = instantiate(config.discriminator_optimizer, params=discriminator_params)
    discriminator_lr_scheduler = instantiate(config.discriminator_lr_scheduler, optimizer=discriminator_optimizer)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    trainer = Trainer(
        model=model,
        criterion=generator_loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
        discriminators=discriminators,
        discriminator_criterion=discriminator_loss_function,
        discriminator_optimizer=discriminator_optimizer,
        discriminator_lr_scheduler=discriminator_lr_scheduler,
    )

    trainer.train()


if __name__ == "__main__":
    main()
