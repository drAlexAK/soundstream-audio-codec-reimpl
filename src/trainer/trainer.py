import torch
from tqdm.auto import tqdm

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
        self.sample_rate = config.datasets.train.sample_rate

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
            value = met(**batch)
            if torch.is_tensor(value):
                value = value.item()
            metrics.update(met.name, value)
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
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
            norm_type,
        )
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

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        """
        Save the checkpoints.

        Args:
            epoch (int): current epoch number.
            save_best (bool): if True, rename the saved checkpoint to 'model_best.pth'.
            only_best (bool): if True and the checkpoint is the best, save it only as
                'model_best.pth'(do not duplicate the checkpoint as
                checkpoint-epochEpochNumber.pth)
        """
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "discriminators": self.discriminators.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "discriminator_lr_scheduler": (
                self.discriminator_lr_scheduler.state_dict()
            ),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        filename = str(self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth")
        if not (only_best and save_best):
            torch.save(state, filename)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(filename, str(self.checkpoint_dir.parent))
            self.logger.info(f"Saving checkpoint: {filename} ...")
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(best_path, str(self.checkpoint_dir.parent))
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from a saved checkpoint (in case of server crash, etc.).
        The function loads state dicts for everything, including model,
        optimizers, etc.

        Notice that the checkpoint should be located in the current experiment
        saved directory (where all checkpoints are saved in '_save_checkpoint').

        Args:
            resume_path (str): Path to the checkpoint to be resumed.
        """
        resume_path = str(resume_path)
        self.logger.info(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path, self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if checkpoint["config"]["model"] != self.config["model"]:
            self.logger.warning(
                "Warning: Architecture configuration given in the config file is different from that "
                "of the checkpoint. This may yield an exception when state_dict is loaded."
            )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.discriminators.load_state_dict(checkpoint["discriminators"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if (
            checkpoint["config"]["generator_optimizer"]
            != self.config["generator_optimizer"]
            or checkpoint["config"]["generator_lr_scheduler"]
            != self.config["generator_lr_scheduler"]
        ):
            self.logger.warning(
                "Warning: Generator optimizer or lr_scheduler given in the config file is different "
                "from that of the checkpoint. Optimizer and scheduler parameters "
                "are not resumed."
            )
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        if (
            checkpoint["config"]["discriminator_optimizer"]
            != self.config["discriminator_optimizer"]
            or checkpoint["config"]["discriminator_lr_scheduler"]
            != self.config["discriminator_lr_scheduler"]
        ):
            self.logger.warning(
                "Warning: Discriminator optimizer or lr_scheduler given in the config file is different "
                "from that of the checkpoint. Discriminator optimizer and scheduler parameters "
                "are not resumed."
            )
        else:
            self.discriminator_optimizer.load_state_dict(
                checkpoint["discriminator_optimizer"]
            )
            self.discriminator_lr_scheduler.load_state_dict(
                checkpoint["discriminator_lr_scheduler"]
            )

        self.logger.info(
            f"Checkpoint loaded. Resume training from epoch {self.start_epoch}"
        )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch, including logging and evaluation on
        non-train partitions.

        Args:
            epoch (int): current training epoch.
        Returns:
            logs (dict): logs that contain the average loss and metric in
                this epoch.
        """
        self.is_train = True
        self.model.train()
        self.discriminators.train()
        self.train_metrics.reset()
        self.writer.set_step((epoch - 1) * self.epoch_len)
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
            tqdm(self.train_dataloader, desc="train", total=self.epoch_len)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    metrics=self.train_metrics,
                )
            except torch.cuda.OutOfMemoryError as e:
                if self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    torch.cuda.empty_cache()  # free some memory
                    continue
                else:
                    raise e

            self.train_metrics.update("grad_norm", self._get_grad_norm())

            # log current results
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.epoch_len + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                if self.lr_scheduler is not None:
                    self.writer.add_scalar(
                        "learning rate", self.lr_scheduler.get_last_lr()[0]
                    )
                self._log_scalars(self.train_metrics)
                self._log_batch(batch_idx, batch)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx + 1 >= self.epoch_len:
                break

        logs = last_train_metrics

        # Run val/test
        for part, dataloader in self.evaluation_dataloaders.items():
            val_logs = self._evaluation_epoch(epoch, part, dataloader)
            logs.update(**{f"{part}_{name}": value for name, value in val_logs.items()})

        return logs

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Evaluate model on the partition after training for an epoch.

        Args:
            epoch (int): current training epoch.
            part (str): partition to evaluate on
            dataloader (DataLoader): dataloader for the partition.
        Returns:
            logs (dict): logs that contain the information about evaluation.
        """
        self.is_train = False
        self.model.eval()
        self.discriminators.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    metrics=self.evaluation_metrics,
                )
            self.writer.set_step(epoch * self.epoch_len, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_batch(
                batch_idx, batch, part
            )  # log only the last batch during inference

        return self.evaluation_metrics.result()
