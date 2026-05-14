
import torch as th
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseQuantizer(nn.Module, ABC):
    """
    Abstract base class for quantizer modules.

    Args:
        codebook_size: Number of discrete codes in the codebook
        embedding_dim: Dimension of each code vector
        commitment_cost: Weight for commitment loss term
        epsilon: Small constant for numerical stability
    """

    def __init__(
        self,
        codebook_size: int,
        embedding_dim: int,
        commitment_cost: float,
        epsilon: float,
        **kwargs
    ):
        super().__init__()

        # Store parameters as instance attributes
        self._codebook_size = codebook_size
        self._embedding_dim = embedding_dim
        self._commitment_cost = commitment_cost
        self._epsilon = epsilon

    @abstractmethod
    def forward(self, z: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Quantize continuous embeddings to discrete codes.

        Args:
            z: Input embeddings

        Returns:
            Tuple: (quantized, vq_loss, perplexity)
        """
        pass

    @property
    def codebook_size(self) -> int:
        """Number of discrete codes in the codebook."""
        return self._codebook_size

    @property
    def embedding_dim(self) -> int:
        """Dimension of each code vector."""
        return self._embedding_dim

    @property
    def commitment_cost(self) -> float:
        """Weight for commitment loss term."""
        return self._commitment_cost

    @property
    def epsilon(self) -> float:
        """Small constant for numerical stability."""
        return self._epsilon