import torch as th
import torch.nn as nn
from typing import Optional, Dict, Tuple
from vector_quantize_pytorch import ResidualVQ

from .base import BaseQuantizer


class SignalAdaptiveVQ(BaseQuantizer):
    """
    Partitioned Vector Quantizer for signal-adaptive compression.
    """

    def __init__(
        self,
        num_quantizers: int = 2,
        codebook_size: int = 8192,
        embedding_dim: int = 18,
        commitment_cost: float = 0.25,
        partition_config: Optional[Dict[str, list]] = None,
        epsilon: float = 1e-5,
        **kwargs
    ):
        super().__init__(
            codebook_size=codebook_size,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            epsilon=epsilon
        )

        self.num_quantizers = num_quantizers

        self.partition_config = partition_config
        self.partition_names = ["smooth_continuous", "smooth_discontinuous", "spiky_continuous", "spiky_discontinuous"]

        # Create separate VectorQuantize instances for each partition
        # Uses vector-quantize-pytorch library (same as EdgeCodec)
        # Benefits: EMA-based updates, k-means init, dead code replacement
        self.vq_modules = nn.ModuleDict()

        for partition_name in self.partition_names:
            start_idx, end_idx = partition_config[partition_name]
            partition_size = end_idx - start_idx

            # Each partition gets its own VQ with EMA updates
            self.vq_modules[partition_name] = ResidualVQ(
                dim=embedding_dim,
                num_quantizers=num_quantizers,
                codebook_size=partition_size,
                threshold_ema_dead_code=2,  # Replace dead codes via EMA (EdgeCodec setting)
                kmeans_init=True,           # K-means initialization (EdgeCodec setting)
                kmeans_iters=100,           # K-means iterations (EdgeCodec setting)
                commitment_weight=commitment_cost,
                accept_image_fmap=False,    # We handle shape ourselves
            )

    def forward(
        self,
        z: th.Tensor,
        signal_types: Optional[th.Tensor] = None
    ) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        """
        Quantize continuous embeddings using signal-type-specific partitions.
        """

        batch_size, num_channels, emb_dim = z.shape

        # Flatten: (batch, num_channels, emb_dim) → (batch * num_channels, emb_dim)
        z_flat = z.reshape(-1, emb_dim)

        # Partition-aware quantization with EMA-based VQ
        signal_types_flat = signal_types.reshape(-1)  # (batch * num_channels,)
        
        quantized_flat, indices_flat, commit_losses = self._quantize_with_partitions(
            z_flat, signal_types_flat
        )

        # Reshape back: (batch * num_channels, emb_dim) → (batch, num_channels, emb_dim)
        quantized = quantized_flat.reshape(batch_size, num_channels, emb_dim)

        vq_loss = commit_losses.mean()

        # Compute perplexity metrics
        perplexity = self._compute_perplexity(indices_flat, signal_types_flat if signal_types is not None else None)

        return quantized, vq_loss, perplexity

    def _quantize_with_partitions(
        self,
        z: th.Tensor,
        signal_types: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Quantize using partition-specific VQ modules with EMA updates.
        """
        
        N, emb_dim = z.shape
        device = z.device

        quantized = th.zeros_like(z)
        indices = th.zeros((N, self.num_quantizers), dtype=th.long, device=device)
        all_commit_losses = []

        # Process each partition separately with its VQ module
        for partition_id, partition_name in enumerate(self.partition_names):
            # Find vectors belonging to this partition
            mask = (signal_types == partition_id)

            if not mask.any():
                continue  # Skip if no vectors for this partition

            # Get vectors for this partition
            z_partition = z[mask]  # (num_partition_vectors, emb_dim)

            # Apply partition-specific VQ 
            quantized_partition, local_indices, commit_loss_partition = self.vq_modules[partition_name](z_partition)

            # Convert local partition indices to global codebook indices
            start_idx, _ = self.partition_config[partition_name]
            global_indices = local_indices + start_idx

            # Store results for this partition
            quantized[mask] = quantized_partition.to(z.dtype)
            indices[mask] = global_indices
            all_commit_losses.append(commit_loss_partition)

        commit_losses = th.cat(all_commit_losses, dim=0)  # Concatenate along batch dim

        return quantized, indices, commit_losses

    def _compute_perplexity(
        self,
        indices: th.Tensor,
        signal_types: Optional[th.Tensor] = None
    ) -> Dict[str, th.Tensor]:
        """
        Compute codebook usage perplexity.
        """
        
        indices_flat = indices.flatten()

        # Compute perplexity on flattened indices
        counts = th.bincount(indices_flat, minlength=self._codebook_size).float()
        probs = counts / counts.sum()
        
        # Avoid log(0)
        probs = probs[probs > 0]
        
        # Perplexity = exp(entropy)
        entropy = -(probs * th.log(probs)).sum()
        perplexity = th.exp(entropy)
        
        metrics = {
            'overall': perplexity,
            'avg_usage': (counts > 0).float().mean(),
        }
        
        if signal_types is not None:
            for partition_id, partition_name in enumerate(self.partition_names):
                mask = (signal_types == partition_id)

                if not mask.any():
                    metrics[partition_name] = th.tensor(1.0, device=indices.device)
                    continue

                if indices.ndim == 2:
                    partition_indices = indices[mask].flatten()
                else:
                    partition_indices = indices[mask]
                
                start_idx, codebook_size = self.partition_config[partition_name]
                local_indices = partition_indices - start_idx
                
                partition_counts = th.bincount(local_indices, minlength=codebook_size).float()
                
                if partition_counts.sum() == 0:
                    metrics[partition_name] = th.tensor(1.0, device=indices.device)
                    continue
                
                partition_probs = partition_counts / partition_counts.sum()
                partition_probs = partition_probs[partition_probs > 0]
                
                partition_entropy = -(partition_probs * th.log(partition_probs)).sum()
                partition_perplexity = th.exp(partition_entropy)
                
                metrics[partition_name] = partition_perplexity
        
        return metrics
