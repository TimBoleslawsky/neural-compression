from abc import abstractmethod, ABC
from typing import Optional, Union, Literal

import torch as th
from lightning.pytorch.trainer.states import RunningStage

class TrainingModule(th.nn.Module, ABC):
    """
    Base class for all training modules.

    A training module orchestrates the full training pipeline:
    - Preprocessing: Apply dynamic transformations before model forward pass
    - Model forward pass
    - Postprocessing: Apply inverse transformations after model output
    - Loss computation
    - Optional metrics computation

    The preprocessing/postprocessing logic handles stage-aware (train/val/test)
    and direction-aware (forward/inverse) dynamic transformations.
    """

    def __init__(
        self,
        model: th.nn.Module,
        discriminator: Optional[th.nn.Module] = None,
        target_key: Literal["encode", "decode"] = "encode",
    ):
        super().__init__()
        self._model = model
        self._discriminator = discriminator
        self.target_key: str = target_key
        
    # Public interface for runners
    @property
    def model(self) -> th.nn.Module:
        """Get the underlying model."""
        return self._model

    @model.setter
    def model(self, value: th.nn.Module) -> None:
        """Set the underlying model."""
        self._model = value

    @abstractmethod
    def _calculate_loss(
        self, pred: th.Tensor, batch: dict[str, th.Tensor]
    ) -> th.Tensor:
        """Compute loss between prediction and target."""
        pass

    def _compute_metrics(
        self,
        _pred: th.Tensor,
        _batch: dict[str, th.Tensor],
        _runner_state: RunningStage,
    ) -> dict[str, th.Tensor]:
        """
        Compute evaluation metrics for the task.

        Default implementation returns no metrics. Specialized training modules
        should implement this method to compute task-specific metrics
        (e.g., accuracy for classification, MAE for regression).

        Args:
            _p_blob: Predicted signal blob (unused in base implementation)
            _batch: Batch containing target signals (unused in base implementation)
            _runner_state: Current training stage (unused in base implementation)

        Returns:
            Dictionary of metric names to values (empty dict by default)
        """
        return {}

    def _forward(
        self,
        batch: dict[str, th.Tensor],
    ) -> th.Tensor:
        return self._model(batch)

    def forward(
        self,
        batch: dict[str, th.Tensor],
        runner_state: Optional[RunningStage] = RunningStage.PREDICTING,
    ) -> tuple[th.Tensor, Union[float, th.Tensor], dict[str, th.Tensor], dict[str, th.Tensor]]:
        """
        Forward pass through the training module.

        Returns:
            Tuple of (predicted, loss, processed_batch, metrics_dict)
        """
        pred = self._forward(batch)

        if runner_state != RunningStage.PREDICTING:
            loss = self._calculate_loss(pred, batch)
        else:
            loss = th.tensor(0.0, device=pred.device)

        # Compute metrics if configured
        metrics = self._compute_metrics(pred, batch, runner_state)

        return pred, loss, batch, metrics
