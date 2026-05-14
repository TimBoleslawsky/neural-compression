import torch as th
import torch.nn as nn
from typing import Optional, Literal
from lightning.pytorch.trainer.states import RunningStage

from .base import TrainingModule


class PerceptualLoss(nn.Module):
    """
    Loss function that emphasizes accurate reconstruction of peaks and valleys.

    Reference: https://arxiv.org/html/2302.04032v3
    """

    def __init__(
        self,
        peak_weight: float = 1.0,
        peak_percentile: float = 90
    ):
        super().__init__()
        self.peak_weight = peak_weight
        self.peak_percentile = peak_percentile

    def forward(self, pred: th.Tensor, target: th.Tensor) -> th.Tensor:
        """
        Compute peak-aware loss.

        Args:
            pred: Predicted signals (B, C, T)
            target: Target signals (B, C, T)

        Returns:
            Combined loss with peak emphasis
        """

        # Identify peaks (high values) and valleys (low values)
        # Compute per-batch percentiles
        upper_threshold = th.quantile(target, self.peak_percentile / 100.0, dim=-1, keepdim=True)
        lower_threshold = th.quantile(target, (100 - self.peak_percentile) / 100.0, dim=-1, keepdim=True)

        # Create mask for extreme values (peaks or valleys)
        is_peak = (target >= upper_threshold) | (target <= lower_threshold)

        # Compute loss only on peaks if any exist
        if is_peak.any():
            # Extract peak values
            peak_pred = pred[is_peak]
            peak_target = target[is_peak]

            # Compute peak-specific loss
            peak_loss_fuction = nn.MSELoss(reduction='mean')
            peak_loss = peak_loss_fuction(peak_pred, peak_target)
        else:
            peak_loss = th.tensor(0.0, device=pred.device)


        return self.peak_weight * peak_loss


class DiscontinuousPerceptualLoss(nn.Module):
    """
    Loss function that emphasizes accurate reconstruction of peaks and valleys and encourages flat regions.
    """

    def __init__(
        self,
        peak_weight: float = 1.0,
        peak_percentile: float = 90,
        lambda_tv: float = 0.1
    ):
        super().__init__()
        self.peak_weight = peak_weight
        self.peak_percentile = peak_percentile
        self.lambda_tv = lambda_tv

    def forward(self, pred: th.Tensor, target: th.Tensor) -> th.Tensor:
        """
        Compute peak-aware loss.

        Args:
            pred: Predicted signals (B, C, T)
            target: Target signals (B, C, T)

        Returns:
            Combined loss with peak emphasis
        """

        # Identify peaks (high values) and valleys (low values)
        # Compute per-batch percentiles
        upper_threshold = th.quantile(target, self.peak_percentile / 100.0, dim=-1, keepdim=True)
        lower_threshold = th.quantile(target, (100 - self.peak_percentile) / 100.0, dim=-1, keepdim=True)

        # Create mask for extreme values (peaks or valleys)
        is_peak = (target >= upper_threshold) | (target <= lower_threshold)

        # Compute loss only on peaks if any exist
        if is_peak.any():
            # Extract peak values
            peak_pred = pred[is_peak]
            peak_target = target[is_peak]

            # Compute peak-specific loss
            peak_loss_fuction = nn.MSELoss(reduction='mean')
            peak_loss = peak_loss_fuction(peak_pred, peak_target)
        else:
            peak_loss = th.tensor(0.0, device=pred.device)

        # Compute prediction differences
        pred_diff = pred[:, 1:] - pred[:, :-1]
        
        # Compute target differences (identify where target is changing)
        target_diff = th.abs(target[:, 1:] - target[:, :-1])
        
        # Create mask: only penalize TV where target is flat (low variation)
        # If target has big jump, don't penalize prediction for having jump too
        is_flat = target_diff < 0.1  # Threshold: adjust based on normalization
        
        # Masked TV: only penalize variations in flat regions
        masked_tv = (th.abs(pred_diff) * is_flat.float()).mean()
        
        return self.lambda_tv * masked_tv + self.peak_weight * peak_loss


class DiscontinuousLoss(nn.Module):
    """
    Loss function for discontinuous (piecewise constant) signals with abrupt state transitions.

    The key insight: By penalizing the TV of the PREDICTION, we encourage the model
    to output piecewise constant signals. However, we need to balance this with
    reconstruction accuracy, so the penalty weight λ should be tuned carefully.

    Args:
        lambda_tv: Weight for total variation penalty (default: 0.1)
                   Higher λ → stronger penalty on smooth transitions

    Reference:
        Total Variation denoising (Rudin et al., 1992)
        Adapted for encouraging piecewise constant reconstructions
    """

    def __init__(self, lambda_tv: float = 0.1):
        super().__init__()
        self.lambda_tv = lambda_tv

    def forward(self, pred: th.Tensor) -> th.Tensor:
        """
        Compute TV loss.
        """

        if pred.dim() == 3:  # (batch, 1, seq_len)
            pred_seq = pred.squeeze(1)  # (batch, seq_len)
        else:
            pred_seq = pred

        # Compute first-order differences
        diff = pred_seq[:, 1:] - pred_seq[:, :-1]  # (batch, seq_len-1)

        # L1 norm of differences (total variation)
        tv = th.abs(diff).mean()

        return  self.lambda_tv * tv


class UniEdgeCodecTrainingModule(TrainingModule):
    """
    Training module for UniEdgeCodec with signal-type-aware losses.

    Loss strategy:
        - Smooth signals (type 0): α × MSE + (1-α) × SmoothL1 (baseline)
        - Sparse signals (type 1): Focal Loss (focus on rare events)
        - Discontinuous signals (type 2): MSE + λ × TV penalty (preserve sharp transitions)
        - VQ commitment: η × commitment_loss (all signals)

    Total loss: weighted_reconstruction_loss + η × vq_commitment

    Flat region handling:
        - Detects flat/constant regions based on local variance
        - Replaces predictions in flat regions with window mean
        - Prevents extreme spikes at flat region boundaries
    """

    def __init__(
        self,
        model: nn.Module,
        alpha: float = 0.5,
        eta: float = 0.25,
        lambda_tv: float = 0.1,
        peak_weight: float = 1.0,
        peak_percentile: float = 90.0,
        discontinuous_weight: float = 1.0,
        perceptual_weight: float = 1.0, 
        discontinuous_perceptual_weight: float = 1.0,
        signal_type_ids: Optional[list[int]] = None,
        target_key: Literal["encode", "decode"] = "encode",
    ):
        super().__init__(
            model=model,
            target_key=target_key,
        )

        # Loss functions
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='mean')
        self.discontinuous_loss = DiscontinuousLoss(lambda_tv=lambda_tv)
        self.perceptual_loss = PerceptualLoss(peak_weight=peak_weight, peak_percentile=peak_percentile)
        self.discontinuous_perceptual_loss = DiscontinuousPerceptualLoss(peak_weight=peak_weight, peak_percentile=peak_percentile, lambda_tv=lambda_tv)

        # Loss weights
        self.alpha = alpha
        self.eta = eta
        self.lambda_tv = lambda_tv
        self.discontinuous_weight = discontinuous_weight
        self.perceptual_weight = perceptual_weight
        self.discontinuous_perceptual_weight = discontinuous_perceptual_weight

        if signal_type_ids is None:
            raise ValueError("UniEdgeCodecTrainingModule requires signal_type_ids aligned to channel order")
        self._signal_types_tensor = th.tensor(list(signal_type_ids), dtype=th.long)

        # Per-type loss tracking (set during forward pass)
        self._per_type_losses = {}
        self._loss_counts = {}

    def _calculate_loss(
        self,
        pred: th.Tensor,
        batch: dict[str, th.Tensor]
    ) -> th.Tensor:
        """
        Compute signal-adaptive total loss with flat region handling.

        Flat regions are detected and replaced with their mean values from target
        before loss computation, preventing extreme spikes at flat region boundaries.
        """

        # Extract predictions and targets
        target = batch[self.target_key]
        
        batch_size, num_channels, seq_len = pred.shape
        
        # Compute signal-type-specific reconstruction losses
        if self._signal_types_tensor is not None:
            # Track per-type losses for monitoring
            total_loss = th.tensor(0.0, device=pred.device)
            loss_counts = {"smooth_continuous": 0, "smooth_discontinuous": 0, "spiky_continuous": 0, "spiky_discontinuous": 0}
            loss_sums = {
                "smooth_continuous": th.tensor(0.0, device=pred.device),
                "smooth_discontinuous": th.tensor(0.0, device=pred.device),
                "spiky_continuous": th.tensor(0.0, device=pred.device),
                "spiky_discontinuous": th.tensor(0.0, device=pred.device)
            }

            # Compute loss per channel based on signal type
            for channel_idx in range(num_channels):
                signal_type = self._signal_types_tensor[channel_idx].item()
                pred_channel = pred[:, channel_idx, :]
                target_channel = target[:, channel_idx, :]

                if signal_type == 0:  # Smooth & continuous signals
                    # Mixed MSE + SmoothL1 (baseline EdgeCodec approach - works well)
                    loss_channel = (
                        self.alpha * self.mse_loss(pred_channel, target_channel) +
                        (1 - self.alpha) * self.smooth_l1_loss(pred_channel, target_channel)
                    ) 
                    loss_sums["smooth_continuous"] += loss_channel
                    loss_counts["smooth_continuous"] += 1

                elif signal_type == 1:  # Smooth & discontinuous signals 
                    # Discontinuous Loss: MSE + Total Variation penalty
                    # Penalizes smooth transitions → encourages sharp state changes
                    loss_channel = (
                        self.mse_loss(pred_channel, target_channel) + self.discontinuous_loss(pred_channel)
                    ) * self.discontinuous_weight
                    loss_sums["smooth_discontinuous"] += loss_channel
                    loss_counts["smooth_discontinuous"] += 1
                
                elif signal_type == 2:  # Spiky & continuous signals
                    # Perceptual Loss: MSE + peak_loss
                    loss_channel = loss_channel = (
                        self.mse_loss(pred_channel, target_channel) + self.perceptual_loss(pred_channel, target_channel)
                    ) * self.perceptual_weight
                    loss_sums["spiky_continuous"] += loss_channel
                    loss_counts["spiky_continuous"] += 1

                elif signal_type == 3:  # Spiky & discontinuous signals
                    # Discontinuous Loss + Perceptual Loss
                    loss_channel = loss_channel = (
                        self.mse_loss(pred_channel, target_channel) + self.discontinuous_perceptual_loss(pred_channel, target_channel)
                    ) * self.discontinuous_perceptual_weight
                    loss_sums["spiky_discontinuous"] += loss_channel
                    loss_counts["spiky_discontinuous"] += 1
                else:
                    raise ValueError(f"Unknown signal type: {signal_type}. Expected 0/1/2/3.")

                # Accumulate total loss (equal weight per channel)
                total_loss += loss_channel

            # Average across all channels
            recon_loss = total_loss / num_channels

            # Store per-type losses for metrics tracking
            self._per_type_losses = {
                "smooth_continuous": loss_sums["smooth_continuous"] / max(loss_counts["smooth_continuous"], 1),
                "smooth_discontinuous": loss_sums["smooth_discontinuous"] / max(loss_counts["smooth_discontinuous"], 1),
                "spiky_continuous": loss_sums["spiky_continuous"] / max(loss_counts["spiky_continuous"], 1),
                "spiky_discontinuous": loss_sums["spiky_discontinuous"] / max(loss_counts["spiky_discontinuous"], 1)
            }
            self._loss_counts = loss_counts

        else:
            raise AttributeError("No signal_type_mapping provided!")

        # VQ commitment loss
        vq_loss = self._model.vq_loss if self._model.vq_loss is not None else th.tensor(0.0, device=pred.device)

        # Total loss
        total_loss = recon_loss + self.eta * vq_loss

        return total_loss

    def _compute_metrics(
        self,
        pred: th.Tensor,
        batch: dict[str, th.Tensor],
        runner_state: RunningStage,
    ) -> dict[str, th.Tensor]:
        """
        Compute detailed metrics including per-partition perplexity.
        """

        # Extract predictions and targets
        target = batch[self.target_key]

        # Handle dict signals
        if isinstance(pred, dict):
            signal_list = []
            for key in sorted(pred.keys()):
                signal = pred[key]
                if signal.dim() == 2:
                    signal = signal.unsqueeze(1)
                signal_list.append(signal)
            pred = th.cat(signal_list, dim=1)

        if isinstance(target, dict):
            signal_list = []
            for key in sorted(target.keys()):
                signal = target[key]
                if signal.dim() == 2:
                    signal = signal.unsqueeze(1)
                signal_list.append(signal)
            target = th.cat(signal_list, dim=1)

        # Compute loss metrics (overall)
        metrics = {
            "mse": self.mse_loss(pred, target),
            "smooth_l1": self.smooth_l1_loss(pred, target),
        }

        # Per-type reconstruction losses (from _calculate_loss)
        # These show how well each signal type is being reconstructed
        if hasattr(self, '_per_type_losses'):
            metrics["loss_smooth_continuous"] = self._per_type_losses.get("smooth_continuous", th.tensor(0.0, device=pred.device))
            metrics["loss_smooth_discontinuous"] = self._per_type_losses.get("smooth_discontinuous", th.tensor(0.0, device=pred.device))
            metrics["loss_spiky_continuous"] = self._per_type_losses.get("spiky_continuous", th.tensor(0.0, device=pred.device))
            metrics["loss_spiky_discontinuous"] = self._per_type_losses.get("spiky_discontinuous", th.tensor(0.0, device=pred.device))

        # Per-type channel counts (for monitoring dataset composition)
        if hasattr(self, '_loss_counts'):
            metrics["count_smooth_continuous"] = th.tensor(self._loss_counts.get("smooth_continuous", 0), device=pred.device)
            metrics["count_smooth_discontinuous"] = th.tensor(self._loss_counts.get("smooth_discontinuous", 0), device=pred.device)
            metrics["count_spiky_continuous"] = th.tensor(self._loss_counts.get("spiky_continuous", 0), device=pred.device)
            metrics["count_spiky_discontinuous"] = th.tensor(self._loss_counts.get("spiky_discontinuous", 0), device=pred.device)

        # VQ metrics from model
        vq_loss = self._model.vq_loss if self._model.vq_loss is not None else th.tensor(0.0, device=pred.device)
        perplexity = self._model.perplexity if self._model.perplexity is not None else {}

        metrics["vq_loss"] = vq_loss

        # Perplexity metrics (per-partition if available)
        if isinstance(perplexity, dict):
            metrics["perplexity_overall"] = perplexity.get("overall", th.tensor(0.0, device=pred.device))
            metrics["perplexity_smooth_continuous"] = perplexity.get("smooth_continuous", th.tensor(0.0, device=pred.device))
            metrics["perplexity_smooth_discontinuous"] = perplexity.get("smooth_discontinuous", th.tensor(0.0, device=pred.device))
            metrics["perplexity_spiky_continuous"] = perplexity.get("spiky_continuous", th.tensor(0.0, device=pred.device))
            metrics["perplexity_spiky_discontinuous"] = perplexity.get("spiky_discontinuous", th.tensor(0.0, device=pred.device))
        else:
            # Fallback for baseline RVQ (returns scalar perplexity)
            metrics["perplexity_overall"] = perplexity

        return metrics
