import torch as th
import torch.nn as nn
from typing import Optional, Literal
from lightning.pytorch.trainer.states import RunningStage

from .base import TrainingModule


class EdgeCodecTrainingModule(TrainingModule):
    """
    Training module for EdgeCodec with mixed reconstruction loss and optional adversarial training.

    Loss components:
        1. Mixed reconstruction: α × MSE + (1-α) × SmoothL1
        2. VQ commitment: η × commitment_loss (from quantizer)
        3. Adversarial (optional): γ × adversarial_loss (if discriminator provided)

    Total loss: mixed_reconstruction + η × vq_commitment + γ × adversarial

    Args:
        model: EdgeCodec model instance
        discriminator: Optional discriminator for adversarial training
        alpha: Weight for MSE vs SmoothL1 (default: 0.5)
                alpha=1.0 → pure MSE
                alpha=0.0 → pure SmoothL1
        eta: Weight for VQ commitment loss (default: 0.25)
        gamma: Weight for adversarial loss (default: 0.1, only used if discriminator provided)
        discriminator_lr: Learning rate for discriminator optimizer (default: 1e-4)
        target_key: Which tensor to treat as target ("encode" or "decode")

    Attributes:
        mse_loss: MSE loss function
        smooth_l1_loss: SmoothL1 loss function
        adversarial_loss: BCE loss for discriminator
        alpha: MSE weight
        eta: VQ commitment weight
        gamma: Adversarial loss weight
        use_discriminator: Whether discriminator training is enabled
    """

    def __init__(
        self,
        model: nn.Module,
        discriminator: Optional[nn.Module] = None,
        alpha: float = 0.4,
        eta: float = 1,
        gamma: float = 0.1,
        discriminator_lr: float = 0.0001,
        target_key: Literal["encode", "decode"] = "encode",
    ):
        super().__init__(
            model=model,
            discriminator=discriminator,
            target_key=target_key,
        )

        # Loss functions (using mean reduction for DDP compatibility)
        # Note: EdgeCodec GitHub used 'sum', but 'mean' is safer with PyTorch Lightning DDP
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='mean')

        # Adversarial loss (for discriminator training)
        self.adversarial_loss = nn.BCEWithLogitsLoss()

        # Loss weights
        self.alpha = alpha
        self.eta = eta
        self.gamma = gamma

        # Validate parameters
        assert 0.0 <= alpha <= 1.0, f"alpha must be in [0, 1], got {alpha}"
        assert eta >= 0.0, f"eta must be non-negative, got {eta}"
        assert gamma >= 0.0, f"gamma must be non-negative, got {gamma}"

        # Discriminator setup
        self.use_discriminator = discriminator is not None
        if self.use_discriminator:
            # Discriminator optimizer (not managed by Lightning - we'll update manually)
            self.disc_optimizer = th.optim.Adam(
                self._discriminator.parameters(),
                lr=discriminator_lr,
                betas=(0.9, 0.98),
                eps=1e-8
            )

    def train_discriminator(
        self,
        real_signals: th.Tensor,
        fake_signals: th.Tensor
    ) -> th.Tensor:
        """
        Train discriminator to distinguish real from fake signals.

        Following EdgeCodec's approach:
        - Real signals get label 1
        - Fake (generated) signals get label 0
        - Fake signals are detached to prevent backprop to generator

        Args:
            real_signals: Ground truth signals (from input batch)
            fake_signals: Generated/reconstructed signals (detached from generator)

        Returns:
            Discriminator loss (scalar tensor)
        """
        # Predict on real signals (label = 1)
        real_pred = self._discriminator(real_signals)
        real_labels = th.ones_like(real_pred)
        real_loss = self.adversarial_loss(real_pred, real_labels)

        # Predict on fake signals (label = 0)
        # CRITICAL: Detach to avoid backprop to generator
        fake_pred = self._discriminator(fake_signals.detach())
        fake_labels = th.zeros_like(fake_pred)
        fake_loss = self.adversarial_loss(fake_pred, fake_labels)

        # Total discriminator loss
        disc_loss = real_loss + fake_loss

        # Backprop and optimize discriminator
        self.disc_optimizer.zero_grad()
        disc_loss.backward()
        self.disc_optimizer.step()

        return disc_loss

    def _forward(
        self,
        batch: dict[str, th.Tensor],
    ) -> th.Tensor:
        # Initialize counter
        if not hasattr(self, '_step_count'):
            self._step_count = 0
        
        # Step 1: Train discriminator only every few steps
        if self.use_discriminator and self.training:
            self._step_count += 1
            
            # Train discriminator only every few generator updates
            if self._step_count % 9 == 0:
                # Get real signals from batch
                real_signals = batch[self.target_key]
                
                self._model.eval()
                self._discriminator.train()
                
                with th.no_grad():
                    fake_signals = self._model(batch)
                
                self.disc_loss = self.train_discriminator(real_signals, fake_signals)
                
                self._model.train()
                self._discriminator.eval()
            else:
                # Don't train discriminator this step
                self.disc_loss = None
        else:
            self.disc_loss = None

        # Step 2: Forward pass for generator (model)
        return self._model(batch)

    def _calculate_loss(
        self,
        pred: th.Tensor,
        batch: dict[str, th.Tensor]
    ) -> th.Tensor:
        """
        Compute total EdgeCodec loss with optional adversarial component.

        Loss = α × MSE + (1-α) × SmoothL1 + η × VQ_commitment + γ × adversarial

        Args:
            p_blob: Predicted signals blob
            batch: Input batch with target signals

        Returns:
            Total loss tensor (scalar)
        """
        # Extract predictions and targets
        target = batch[self.target_key]

        # 1. Mixed reconstruction loss
        mse = self.mse_loss(pred, target)
        smooth_l1 = self.smooth_l1_loss(pred, target)
        mixed_recon_loss = self.alpha * mse + (1 - self.alpha) * smooth_l1

        # 2. VQ commitment loss (extracted from model during forward pass)
        vq_loss = self._model.vq_loss if self._model.vq_loss is not None else th.tensor(0.0, device=pred.device)

        # 3. Adversarial loss (if discriminator enabled)
        if self.use_discriminator and self.training:
            # Fool the discriminator: predict fake signals and label them as real
            fake_pred = self._discriminator(pred)
            real_labels = th.ones_like(fake_pred)
            adv_loss = self.adversarial_loss(fake_pred, real_labels)
        else:
            adv_loss = th.tensor(0.0, device=pred.device)

        # 4. Total loss
        total_loss = mixed_recon_loss + self.eta * vq_loss + self.gamma * adv_loss

        return total_loss

    def _compute_metrics(
        self,
        pred: th.Tensor,
        batch: dict[str, th.Tensor],
        runner_state: RunningStage,
    ) -> dict[str, th.Tensor]:
        """
        Compute detailed metrics for monitoring.

        Returns separate MSE, SmoothL1, VQ loss, perplexity, and discriminator metrics.

        Args:
            p_blob: Predicted signals blob
            batch: Input batch with target signals
            runner_state: Current training stage (unused)

        Returns:
            Dictionary with keys:
                - "mse": Mean squared error
                - "smooth_l1": Smooth L1 loss
                - "vq_loss": VQ commitment loss
                - "perplexity": Codebook usage metric
                - "mixed_recon": Weighted reconstruction loss
                - "disc_loss": Discriminator loss (if enabled)
                - "adv_loss": Adversarial loss for generator (if enabled)
        """
        # Extract predictions and targets
        target = batch[self.target_key]

        # Compute individual loss components
        mse = self.mse_loss(pred, target)
        smooth_l1 = self.smooth_l1_loss(pred, target)
        mixed_recon = self.alpha * mse + (1 - self.alpha) * smooth_l1

        # Get VQ metrics from model
        vq_loss = self._model.vq_loss if self._model.vq_loss is not None else th.tensor(0.0, device=pred.device)
        perplexity = self._model.perplexity if self._model.perplexity is not None else th.tensor(0.0, device=pred.device)

        metrics = {
            "mse": mse,
            "smooth_l1": smooth_l1,
            "vq_loss": vq_loss,
            "perplexity": perplexity,
            "mixed_recon": mixed_recon,
        }

        # Add discriminator metrics if enabled
        if self.use_discriminator:
            # Discriminator loss (already computed in _forward)
            if self.disc_loss is not None:
                metrics["disc_loss"] = self.disc_loss

            # Adversarial loss for generator
            with th.no_grad():
                fake_pred = self._discriminator(pred)
                real_labels = th.ones_like(fake_pred)
                adv_loss = self.adversarial_loss(fake_pred, real_labels)
                metrics["adv_loss_times_gamma"] = adv_loss * self.gamma

        return metrics
