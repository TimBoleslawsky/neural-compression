import torch as th
from vector_quantize_pytorch import ResidualVQ

from .base import BaseQuantizer


class ResidualVectorQuantizer(BaseQuantizer):
    """
    Residual Vector Quantizer using lucidrains/vector-quantize-pytorch library.

    Architecture:
        z → VQ1 → residual → VQ2 → residual → VQ3 → residual → VQ4 → final

    Args:
        num_quantizers: Number of RVQ stages
        codebook_size: Number of codes per quantizer
        embedding_dim: Dimension of input embeddings
        commitment_cost: Weight for commitment loss
        epsilon: Numerical stability constant
    """

    def __init__(
        self,
        num_quantizers: int = 4,
        codebook_size: int = 768,
        embedding_dim: int = 9,
        commitment_cost: float = 0.25,
        epsilon: float = 1e-5,
        **kwargs
    ):
        # Initialize base class
        super().__init__(
            codebook_size=codebook_size,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            epsilon=epsilon
        )

        self.num_quantizers = num_quantizers

        self.rvq = ResidualVQ(
            dim=embedding_dim,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            threshold_ema_dead_code=2,
            kmeans_init=True,
            kmeans_iters=100,

        )

    def forward(
        self,
        z: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Perform residual vector quantization.

        Args:
            z: Input embeddings of shape (batch, seq_len, embedding_dim)

        Returns:
            Tuple containing:
            - quantized: Quantized embeddings (batch, seq_len, embedding_dim)
            - vq_loss: Total VQ loss across all quantizers (scalar)
            - perplexity: Average codebook usage metric (scalar)
        """
        # Apply RVQ
        quantized, indices, commit_loss = self.rvq(z)

        # VQ loss = commitment loss
        # ResidualVQ returns commit_loss with shape (batch, num_quantizers)
        # We need to average across both dimensions to get a scalar (matching EdgeCodec GitHub)
        vq_loss = commit_loss.mean()

        # Compute perplexity for codebook usage monitoring
        perplexities = []

        for q_idx in range(self.num_quantizers):
            q_indices = indices[:, :, q_idx].reshape(-1)
            counts = th.bincount(q_indices, minlength=self.codebook_size).float()

            probs = counts / (counts.sum() + self._epsilon)
            log_probs = th.log(probs + self._epsilon)
            entropy = -th.sum(probs * log_probs)
            perp = th.exp(entropy)
            perplexities.append(perp)

        perplexity = th.stack(perplexities).mean()

        return quantized, vq_loss, perplexity
