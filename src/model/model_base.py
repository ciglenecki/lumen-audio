from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.callbacks import BaseFinetuning

import src.config.config_defaults as config_defaults
from src.utils.utils_train import get_all_modules_after


class ModelBase(pl.LightningModule, ABC):
    """Model hooks:

    def fit(self):
        if global_rank == 0:
            # prepare data is called on GLOBAL_ZERO only
            prepare_data()

        configure_callbacks()

        with parallel(devices):
            # devices can be GPUs, TPUs, ...
            train_on_device(model)


    def train_on_device(model):
        # called PER DEVICE
        setup("fit")
        configure_optimizers()
        on_fit_start()

        # the sanity check runs here

        on_train_start()
        for epoch in epochs:
            fit_loop()
        on_train_end()

        on_fit_end()
        teardown("fit")


    def fit_loop():
        on_train_epoch_start()

        for batch in train_dataloader():
            on_train_batch_start()

            on_before_batch_transfer()
            transfer_batch_to_device()
            on_after_batch_transfer()

            training_step()

            on_before_zero_grad()
            optimizer_zero_grad()

            on_before_backward()
            backward()
            on_after_backward()

            on_before_optimizer_step()
            configure_gradient_clipping()
            optimizer_step()

            on_train_batch_end()

            if should_check_val:
                val_loop()
        # end training epoch
        training_epoch_end()

        on_train_epoch_end()


    def val_loop():
        on_validation_model_eval()  # calls `model.eval()`
        torch.set_grad_enabled(False)

        on_validation_start()
        on_validation_epoch_start()

        val_outs = []
        for batch_idx, batch in enumerate(val_dataloader()):
            on_validation_batch_start(batch, batch_idx)

            batch = on_before_batch_transfer(batch)
            batch = transfer_batch_to_device(batch)
            batch = on_after_batch_transfer(batch)

            out = validation_step(batch, batch_idx)

            on_validation_batch_end(batch, batch_idx)
            val_outs.append(out)

        validation_epoch_end(val_outs)

        on_validation_epoch_end()
        on_validation_end()

        # set up for train
        on_validation_model_train()  # calls `model.train()`
        torch.set_grad_enabled(True)
    """

    def __init__(
        self,
        lr: float,
        warmup_lr: Optional[float],
        unfreeze_at_epoch: Optional[int],
        backbone_after: str | None,
        head_after: str | None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.backbone_lr = lr  # lr once backbone gets unfrozen
        self.warmup_lr = warmup_lr  # starting warmup lr
        self.backbone_after = backbone_after
        self.head_after = head_after
        self.has_finetuning = unfreeze_at_epoch is not None
        if self.warmup_lr:
            self.lr = self.warmup_lr
        else:
            self.lr = lr

        assert int(bool(self.unfreeze_at_epoch)) + int(bool(self.warmup_lr)) in [
            0,
            2,
        ], "Both should exist or both shouldn't exist!"

        assert int(bool(self.head_after)) + int(bool(self.backbone_after)) in [
            0,
            2,
        ], "Both should exist or both shouldn't exist!"

        # save in case indices change with config changes
        self.backup_instruments = config_defaults.INSTRUMENT_TO_IDX

    def setup(self, stage: str) -> None:
        """Freezes everything except trainable backbone and head."""
        out = super().setup(stage)
        self.num_of_steps_in_epoch = int(
            self.trainer.estimated_stepping_batches / self.trainer.max_epochs
        )

        if self.head() is not None and self.trainable_backbone() is not None:
            BaseFinetuning.freeze(self, train_bn=False)
            BaseFinetuning.make_trainable(self.trainable_backbone())
            BaseFinetuning.make_trainable(self.head())

        if self.has_finetuning:
            self._set_finetune_until_step()
        return out

    def head(self) -> Union[nn.ModuleList, nn.Module] | None:
        """Returns "head" part of the model. That's usually whatever's after the large feature
        extractor.

        Returns:
            Union[nn.ModuleList, nn.Module]: modules which are considered a head
        """
        if self.head_after:
            return get_all_modules_after(self, self.head_after)
        else:
            return None

    def trainable_backbone(self) -> Union[nn.ModuleList, nn.Module] | None:
        """Returns "backbone" part of the model. That's usually the large feature extractor.

        Returns:
            Union[nn.ModuleList, nn.Module]: modules which are considered a backbone
        """
        if self.backbone_after:
            return get_all_modules_after(self, self.backbone_after)
        else:
            return None

    def _set_lr(self, lr: float):
        if self.trainer is not None:
            for optim in self.trainer.optimizers:
                for param_group in optim.param_groups:
                    param_group["lr"] = lr
        self.lr = lr
        self.hparams.update({"lr": lr})

    def count_trainable_params(self):
        """Returns number of total, trainable and non trainable parameters."""
        return {
            "total": int(sum(p.numel() for p in self.parameters())),
            "trainable": int(
                sum(p.numel() for p in self.parameters() if p.requires_grad)
            ),
            "non_trainable": int(
                sum(p.numel() for p in self.parameters() if not p.requires_grad)
            ),
        }

    def is_finetuning_phase(self):
        if not self.has_finetuning:
            return False

        return (self.finetune_until_step is not None) and (
            self.global_step <= self.finetune_until_step
        )

    def print_params(self):
        """Print module's parameters."""
        for _, module in self.named_modules():
            for param_name, param in module.named_parameters():
                print(param_name, "requires_grad:", param.requires_grad)

    def print_modules(self):
        """Print module's parameters."""
        for module_name, module in self.named_modules():
            module_req_grad = all(
                [x[1].requires_grad for x in module.named_parameters()]
            )
            print(module_name, "requires_grad:", module_req_grad)

    def _set_finetune_until_step(self):
        """We have to caculate what's the step number after which the fine tuning phase is over. We
        also dynamically set the finetune lr nominator, which will ensure that warmup learning rate
        starts at `warmup_lr` and ends with `backbone_lr`. Once the trainer reaches the step
        `finetune_until_step` and learning rate becomes `backbone_lr`, the finetuning phase is
        over.

        lr = ((warmup_lr * numerator) * numerator) ... * numerator))  =  warmup_lr * (numerator)^unfreeze_backbone_at_epoch
                                                    ^ multiplying unfreeze_backbone_at_epoch times
        """
        assert self.unfreeze_at_epoch is not None

        self.finetune_until_step = self.num_of_steps_in_epoch * self.unfreeze_at_epoch

        _a = self.backbone_lr / self.warmup_lr
        _b = self.finetune_until_step - 1
        self.finetune_lr_nominator = np.exp(np.log(_a) / (_b))

        assert np.isclose(
            np.log(self.backbone_lr),
            np.log(self.warmup_lr)
            + (self.finetune_until_step - 1) * np.log(self.finetune_lr_nominator),
        ), "should be: lr = warmup_lr * (numerator)^unfreeze_backbone_at_epoch"

    @abstractmethod
    def _lr_finetuning_step(self, optimizer_idx):
        """Manually updates learning rate in the finetuning phase.

        Args:
            optimizer_idx
        """
        return

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        """We ignore the lr scheduler in the fine tuning phase and update lr maunally.

        Once the finetuning phase is over we start using the lr scheduler
        """

        if self.is_finetuning_phase():
            self._lr_finetuning_step(optimizer_idx)
            return

        if metric is None:
            scheduler.step()
        else:
            scheduler.step(metric)

    def on_fit_start(self) -> None:
        super().on_fit_start()
        if self.warmup_lr:
            self._set_lr(self.warmup_lr)
