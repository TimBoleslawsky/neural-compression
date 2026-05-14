import torch as th
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseDecoder(nn.Module, ABC):
    """
    Abstract base class for decoder modules.

    Args:
        embedding_dim: Number of input embedding channels
        out_channels: Number of output signal channels
        num_layers: Number of decoder layers
        kernel_size: Convolution kernel size
        dropout: Dropout probability
    """

    def __init__(
        self,
        embedding_dim: int,
        out_channels: int,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()

        # Store parameters as instance attributes
        self._embedding_dim = embedding_dim
        self._out_channels = out_channels
        self._num_layers = num_layers
        self._kernel_size = kernel_size
        self._dropout = dropout

    @abstractmethod
    def forward(self, embeddings: th.Tensor) -> th.Tensor:
        """
        Decode embeddings back to signal space.

        Args:
            embeddings: Embedding tensor

        Returns:
            Reconstructed signals
        """
        pass

    @property
    def embedding_dim(self) -> int:
        """Number of input embedding channels."""
        return self._embedding_dim

    @property
    def out_channels(self) -> int:
        """Number of output signal channels."""
        return self._out_channels

    @property
    def num_layers(self) -> int:
        """Number of decoder layers."""
        return self._num_layers

    @property
    def kernel_size(self) -> int:
        """Convolution kernel size."""
        return self._kernel_size

    @property
    def dropout(self) -> float:
        """Dropout probability."""
        return self._dropout
