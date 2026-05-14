import torch as th
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Literal, Optional
from lightning.pytorch.trainer.states import RunningStage

from .base import TrainingModule


class SimpleTaskHead(nn.Module):
    """
    Lightweight task prediction head for regression or classification.

    Args:
        input_channels: Number of input signal channels
        sequence_length: Input sequence length
        output_dim: Task output dimension
        task_type: "regression" or "classification"
        hidden_dim: Hidden layer size
    """

    def __init__(
        self,
        input_channels: int,
        sequence_length: int,
        output_dim: int,
        task_type: Literal["regression", "classification"],
        hidden_dim: int = 128
    ):
        super().__init__()

        input_dim = input_channels * sequence_length
        self.task_type = task_type

        self.flatten = nn.Flatten(start_dim=1)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, signals: th.Tensor) -> th.Tensor:
        """
        Args:
            signals: Compressed signals

        Returns:
            Task predictions
        """
        x = self.flatten(signals)  # (B, C*T)
        return self.net(x)  # (B, output_dim)


class TaskAwareEdgeCodecTrainingModule(TrainingModule):
    """
    EdgeCodec with task-aware training (simplest implementation).

    Loss = α × MSE + (1-α) × SmoothL1 + η × vq + β × task_loss

    Automatically detects dataset type and loads appropriate targets:
    - Smart Home (energydata) → Regression (appliance energy sequence)
    - E-Mobility (emob) → Classification (battery anomaly detection)

    Args:
        model: EdgeCodec model
        config: Full training configuration (for loading targets)
        alpha: Weight for MSE vs SmoothL1 (default: 0.4)
        eta: VQ commitment weight (default: 1.0)
        beta: Task loss weight (default: 0.1) - TUNE THIS!
        task_hidden_dim: Task head hidden dimension (default: 128)
        target_key: Which tensor to treat as target ("encode" or "decode")

    Usage:
        training_module = TaskAwareEdgeCodecTrainingModule(
            model=edgecodec,
            config=config,  # Automatically detects dataset and loads targets
            beta=0.1        # Start here, tune between 0.1-0.5
        )

    NO Data Loader Modification Required!
    Targets are loaded automatically based on dataset configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[dict] = None,
        alpha: float = 0.4,
        eta: float = 1.0,
        beta: float = 0.1,
        task_hidden_dim: int = 128,
        target_key: Literal["encode", "decode"] = "encode",
    ):
        super().__init__(
            model=model,
            target_key=target_key,
        )

        # Loss functions
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='mean')

        # Loss weights
        self.alpha = alpha
        self.eta = eta
        self.beta = beta

        # Task loss clipping threshold (prevent extreme spikes)
        self.task_loss_clip = 100.0  # Clip task loss to reasonable range

        # Validate parameters
        assert 0.0 <= alpha <= 1.0, f"alpha must be in [0, 1], got {alpha}"
        assert eta >= 0.0, f"eta must be non-negative, got {eta}"
        assert beta >= 0.0, f"beta must be non-negative, got {beta}"

        # Load task targets and setup task head
        if config is not None:
            self._setup_task_from_config(config, task_hidden_dim)
        else:
            # Fallback: no task supervision if config not provided
            self.task_head = None
            self.task_targets = None
            self.task_type = None
            print("Warning: No config provided, task-aware training disabled (beta will be ignored)")

    def _setup_task_from_config(self, config, task_hidden_dim: int):
        """
        Load targets and setup task head based on dataset configuration.

        Follows the same logic as evaluate_model.py's load_downstream_targets().
        """
        dataset_name = config.dataset.name
        dataset_args = config.dataset.arguments

        split_ratios = dataset_args.split_ratios
        train_ratio = split_ratios[0]

        window_size = dataset_args.window_size
        stride = dataset_args.stride

        # Determine input channels from encode_signals
        input_channels = len(dataset_args.encode_signals)

        # ====================================================================
        # EMOB DATASET - CLASSIFICATION TASK
        # ====================================================================
        if dataset_name == 'emob' or 'emob' in str(dataset_args.get('file_path', '')).lower():
            self.task_type = 'classification'
            task_name = 'Battery Anomaly Detection'

            # Load labels
            label_file = dataset_args.get(
                'label_file',
                "datasets/emob_cycle_labels.csv"
            )

            print(f"[Task-Aware] Dataset: emob | Task: {task_name} ({self.task_type})")
            print(f"[Task-Aware] Loading labels from: {label_file}")

            labels_df = pd.read_csv(label_file)
            total_windows = len(labels_df)

            # Extract training set labels only
            train_end = int(total_windows * train_ratio)
            train_labels = labels_df.iloc[:train_end]['anomaly_label'].values

            # Convert to tensor
            self.task_targets = th.tensor(train_labels, dtype=th.long)

            # Task head for binary classification
            num_classes = 2
            self.task_head = SimpleTaskHead(
                input_channels=input_channels,
                sequence_length=window_size,
                output_dim=num_classes,
                task_type='classification',
                hidden_dim=task_hidden_dim
            )

            self.task_loss_fn = nn.CrossEntropyLoss(reduction='mean')

            print(f"[Task-Aware] Loaded {len(self.task_targets)} training labels")
            print(f"[Task-Aware] Task head: {input_channels} channels × {window_size} steps → {num_classes} classes")

        # ====================================================================
        # SMART HOME DATASET - REGRESSION TASK
        # ====================================================================
        elif dataset_name == 'timeseries_text' or 'energydata' in str(dataset_args.get('file_path', '')).lower():
            self.task_type = 'regression'
            task_name = 'Appliance Energy Sequence Prediction'
            target_column = 'Appliances'

            file_path = dataset_args.file_path

            print(f"[Task-Aware] Dataset: smart home | Task: {task_name} ({self.task_type})")
            print(f"[Task-Aware] Loading target column '{target_column}' from: {file_path}")

            df = pd.read_csv(file_path)
            appliances = df[target_column].values

            num_timesteps = len(df)
            num_windows = (num_timesteps - window_size) // stride + 1

            # Extract training set windows
            train_end = int(num_windows * train_ratio)

            train_window_targets = []
            for i in range(train_end):
                start_idx = i * stride
                end_idx = start_idx + window_size
                window_appliances = appliances[start_idx:end_idx]
                train_window_targets.append(window_appliances)

            # Convert to tensor: (num_train_windows, window_size)
            self.task_targets = th.tensor(np.array(train_window_targets), dtype=th.float32)

            # Task head for regression
            output_dim = window_size  # Predict full sequence
            self.task_head = SimpleTaskHead(
                input_channels=input_channels,
                sequence_length=window_size,
                output_dim=output_dim,
                task_type='regression',
                hidden_dim=task_hidden_dim
            )

            # Use SmoothL1Loss (Huber) instead of MSE for robustness to outliers
            self.task_loss_fn = nn.SmoothL1Loss(reduction='mean', beta=1.0)

            print(f"[Task-Aware] Loaded {len(self.task_targets)} training windows")
            print(f"[Task-Aware] Task head: {input_channels} channels × {window_size} steps → {output_dim} predictions")

        else:
            raise ValueError(
                f"Unknown dataset: {dataset_name}\n"
                f"Supported datasets:\n"
                f"  - 'emob' → Binary classification (anomaly detection)\n"
                f"  - 'timeseries_text' with 'energydata' file → Regression (energy prediction)"
            )

        print(f"[Task-Aware] Task-aware training enabled with β={self.beta}")

    def _forward(
        self,
        batch: dict[str, th.Tensor],
    ) -> th.Tensor:
        return self._model(batch)

    def _calculate_loss(
        self,
        pred: th.Tensor,
        batch: dict[str, th.Tensor]
    ) -> th.Tensor:
        """
        Compute total task-aware loss.

        Loss = α × MSE + (1-α) × SmoothL1 + η × vq + β × task_loss
        """
        # Extract predictions and targets
        target = batch[self.target_key]

        # 1. Mixed reconstruction loss (EdgeCodec style)
        mse = self.mse_loss(pred, target)
        smooth_l1 = self.smooth_l1_loss(pred, target)
        mixed_recon_loss = self.alpha * mse + (1 - self.alpha) * smooth_l1

        # 2. VQ commitment loss
        vq_loss = self._model.vq_loss if self._model.vq_loss is not None else th.tensor(0.0, device=pred.device)

        # 3. Task loss (NEW!)
        task_loss = th.tensor(0.0, device=pred.device)
        if self.beta > 0.0 and self.task_head is not None and self.task_targets is not None:
            # Get batch indices (assumes sequential batching during training)
            batch_size = pred.shape[0]

            # Get batch start index from the batch metadata if available
            # Otherwise assume sequential indexing
            if "batch_idx" in batch:
                batch_indices = batch["batch_idx"]
            else:
                # Fallback: use internal counter with modulo wrapping
                if not hasattr(self, '_batch_counter'):
                    self._batch_counter = 0

                # Create indices with modulo to handle wraparound
                dataset_size = len(self.task_targets)
                batch_indices = [(self._batch_counter + i) % dataset_size for i in range(batch_size)]
                self._batch_counter = (self._batch_counter + batch_size) % dataset_size

            # Extract task labels for this batch
            batch_targets = self.task_targets[batch_indices].to(pred.device)

            # Predict from compressed signals
            task_predictions = self.task_head(pred)

            # Compute task loss
            task_loss = self.task_loss_fn(task_predictions, batch_targets)

            # Clip task loss to prevent extreme spikes
            task_loss = th.clamp(task_loss, max=self.task_loss_clip)

        # 4. Total loss
        total_loss = mixed_recon_loss + self.eta * vq_loss + self.beta * task_loss

        # Cache for metrics
        self._task_loss = task_loss

        return total_loss

    def _compute_metrics(
        self,
        pred: th.Tensor,
        batch: dict[str, th.Tensor],
        runner_state: RunningStage,
    ) -> dict[str, th.Tensor]:
        """Compute detailed metrics for monitoring."""
        # Extract predictions and targets
        target = batch[self.target_key]

        # Compute individual loss components
        mse = self.mse_loss(pred, target)
        smooth_l1 = self.smooth_l1_loss(pred, target)
        mixed_recon = self.alpha * mse + (1 - self.alpha) * smooth_l1

        # Get VQ metrics
        vq_loss = self._model.vq_loss if self._model.vq_loss is not None else th.tensor(0.0, device=pred.device)
        perplexity = self._model.perplexity if self._model.perplexity is not None else th.tensor(0.0, device=pred.device)

        metrics = {
            "mse": mse,
            "smooth_l1": smooth_l1,
            "vq_loss": vq_loss,
            "perplexity": perplexity,
            "mixed_recon": mixed_recon,
        }

        # Add task metrics
        if hasattr(self, '_task_loss'):
            metrics["task_loss"] = self._task_loss
            metrics["task_loss_times_beta"] = self._task_loss * self.beta

        return metrics

    def on_train_epoch_start(self):
        """Reset batch counter at epoch start."""
        if hasattr(self, '_batch_counter'):
            self._batch_counter = 0
