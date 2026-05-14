import torch as th
from torch import nn
from typing import Literal

from .base import TrainingModule


class CompressionTrainingModule(TrainingModule):
    """
    Training module for compression models with MSE reconstruction loss.

    Args:
        model: The neural compression model
        target_key: Which tensor to treat as target ("encode" or "decode")
    """

    def __init__(
        self,
        model: nn.Module,
        target_key: Literal["encode", "decode"] = "encode",
    ):
        super().__init__(model, target_key=target_key)

    def _calculate_loss(
        self,
        pred: th.Tensor,
        batch: dict[str, th.Tensor]
    ) -> th.Tensor:
        """
        Calculate MSE reconstruction loss.

        Implementation:
        - Sum over variables/channels (D)
        - Mean over time (T)
        - Mean over batch (B)

        This differs from PyTorch's default MSELoss which divides by all dimensions (B×D×T).
        The paper sums over variables and averages over time only.

        Args:
            p_blob: Predicted signal blob from model output
            batch: Batch containing target signals in "decode"

        Returns:
            MSE loss tensor (scalar)
        """
        pred_signals = pred
        target_signals = batch["decode"]

        # Compute MSE using paper's formula
        # Shape progression: (B, D, T) -> (B,) -> scalar
        squared_errors = (pred_signals - target_signals) ** 2  # (B, D, T)

        # Sum over channels/variables (dim=1), mean over time (dim=2)
        # This gives per-sample MSE: (1/T) × Σ_d Σ_t (x - x̂)²
        mse_per_sample = squared_errors.sum(dim=1).mean(dim=1)  # (B,)

        # Average over batch
        mse = mse_per_sample.mean()  # scalar

        return mse
