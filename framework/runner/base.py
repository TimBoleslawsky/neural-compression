from functools import partial
from typing import Union, Any, Iterable, Optional

import lightning.pytorch as pl
import torch as th

from framework.training import TrainingModule

class BaseRunner(pl.LightningModule):
    def __init__(
        self,
        *,
        training_module: TrainingModule,
        optimizer: partial[th.optim.Optimizer],
        scheduler: Optional[partial[th.optim.lr_scheduler.LRScheduler]] = None
    ):
        super().__init__()
        self.training_module: TrainingModule = training_module
        self.optimizer_init_fn: partial[th.optim.Optimizer] = optimizer
        self.scheduler_init_fn: partial[th.optim.lr_scheduler.LRScheduler] = scheduler

    @property
    def model(self) -> th.nn.Module:
        return self.training_module.model

    @model.setter
    def model(self, model: th.nn.Module) -> None:
        self.training_module.model = model

    def configure_optimizers(self):
        optimizer = self.optimizer_init_fn(
            params=filter(
                lambda p: p.requires_grad,
                self.training_module.model.parameters(),
            )
        )
        optimization_dict = {"optimizer": optimizer}

        if self.scheduler_init_fn is not None:
            scheduler_dict = self.scheduler_init_fn(optimizer=optimizer)
            optimization_dict.update({"lr_scheduler": scheduler_dict})

        return optimization_dict
    

    def _base_predict(
        self, batch: dict[str, th.Tensor]
    ) -> tuple[th.Tensor, th.Tensor, dict[str, th.Tensor], dict[str, th.Tensor]]:
        return self.training_module(batch, self.trainer.state.stage)

    def training_step(
        self, batch: Union[Iterable[th.Tensor], Any]
    ) -> Union[tuple[dict[str, th.Tensor], th.Tensor], Any]:
        p_blob, loss, precessed_batch, metrics = self._base_predict(batch)

        # Log loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Log metrics if any
        if metrics:
            self.log_dict(metrics, on_step=True, on_epoch=True, logger=True)

        return {
            "predicted": p_blob,
            "loss": loss,
            "processed_batch": precessed_batch,
        }
    
    def validation_step(
            self, batch: Union[Iterable[th.Tensor], Any]
        ) -> Union[tuple[dict[str, th.Tensor], th.Tensor], Any]:
            p_blob, loss, precessed_batch, metrics = self._base_predict(batch)

            # Log loss (sync_dist=True averages across all DDP processes)
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

            # Log metrics if any
            if metrics:
                self.log_dict(metrics, on_step=False, on_epoch=True, logger=True, sync_dist=True)

            return {
                "predicted": p_blob,
                "loss": loss,
                "processed_batch": precessed_batch,
            }
    
    def test_step(
        self, batch: Union[Iterable[th.Tensor], Any]
    ) -> Union[tuple[dict[str, th.Tensor], th.Tensor], Any]:
        p_blob, loss, precessed_batch, metrics = self._base_predict(batch)

        # Log loss (sync_dist=True averages across all DDP processes)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # Log metrics if any
        if metrics:
            self.log_dict(metrics, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        return {
            "predicted": p_blob,
            "loss": loss,
            "processed_batch": precessed_batch,
        }
