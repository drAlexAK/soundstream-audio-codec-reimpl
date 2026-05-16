import torch

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def __init__(
        self,
        model,
        criterion,
        metrics,
        optimizer,
        lr_scheduler,
        config,
        device,
        dataloaders,
        logger,
        writer,
        discriminators,
        discriminator_criterion,
        discriminator_optimizer,
        discriminator_lr_scheduler,
        epoch_len=None,
        skip_oom=True,
        batch_transforms=None,
    ):
        super().__init__(
            model=model,
            criterion=criterion,
            metrics=metrics,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config=config,
            device=device,
            dataloaders=dataloaders,
            logger=logger,
            writer=writer,
            epoch_len=epoch_len,
            skip_oom=skip_oom,
            batch_transforms=batch_transforms,
        )
        self.discriminators = discriminators
        self.discriminator_criterion = discriminator_criterion
        self.discriminator_optimizer = discriminator_optimizer
        self.discriminator_lr_scheduler = discriminator_lr_scheduler
        self.sample_rate = config.dataloader.train.collate_fn.sample_rate

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()
            self._set_discriminators_requires_grad(True)

        outputs = self.model(**batch)
        batch.update(outputs)
        batch["x"] = batch["data_object"]
        batch["x_hat"] = batch["result"]

        real_logits, real_features = self._run_discriminators(batch["x"])
        fake_logits, fake_features = self._run_discriminators(batch["x_hat"].detach())

        discriminator_losses = self.discriminator_criterion(
            real_logits=real_logits,
            fake_logits=fake_logits,
        )
        batch.update(discriminator_losses)

        if self.is_train:
            batch["discriminator_loss"].backward()
            self._clip_grad_norm()
            self.discriminator_optimizer.step()
            if self.discriminator_lr_scheduler is not None:
                self.discriminator_lr_scheduler.step()
            self._set_discriminators_requires_grad(False)

        real_logits, real_features = self._run_discriminators(batch["x"])
        fake_logits, fake_features = self._run_discriminators(batch["x_hat"])

        batch.update(
            {
                "real_logits": real_logits,
                "fake_logits": fake_logits,
                "real_features": real_features,
                "fake_features": fake_features,
            }
        )

        generator_losses = self.criterion(**batch)
        batch.update(generator_losses)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _run_discriminators(self, audio):
        waveform_logits, waveform_features = self.discriminators["waveform"](audio)
        stft_logits, stft_features = self.discriminators["stft"](audio)

        logits = waveform_logits + [stft_logits]
        features = waveform_features + [stft_features]
        return logits, features

    def _set_discriminators_requires_grad(self, requires_grad):
        for parameter in self.discriminators.parameters():
            parameter.requires_grad_(requires_grad)

    @torch.no_grad()
    def _get_discriminator_grad_norm(self, norm_type=2):
        parameters = [p for p in self.discriminators.parameters() if p.grad is not None]
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]), norm_type)
        return total_norm.item()

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch after batch training. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example
        for i in range(min(self.config.trainer.samples, batch["data_object"].shape[0])):
            self.writer.add_audio(
                f"{mode}_epoch_{batch_idx}/source_{i}",
                batch["data_object"][i],
                sample_rate=self.sample_rate,
            )
            self.writer.add_audio(
                f"{mode}_epoch_{batch_idx}/reconstructed_{i}",
                batch["result"][i],
                sample_rate=self.sample_rate,
            )
