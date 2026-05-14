
import torch as th
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseEncoder(nn.Module, ABC):
    """
    Abstract base class for encoder modules.

    Args:
        in_channels: Number of input signal channels
        hidden_channels: Number of output feature channels
        num_layers: Number of encoder layers
        kernel_size: Convolution kernel size
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()

        # Store parameters as instance attributes
        self._in_channels = in_channels
        self._hidden_channels = hidden_channels
        self._num_layers = num_layers
        self._kernel_size = kernel_size
        self._dropout = dropout

    @abstractmethod
    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Encode input signals to feature representations.

        Args:
            x: Input tensor

        Returns:
            Encoded features
        """
        pass

    @property
    def in_channels(self) -> int:
        """Number of input channels."""
        return self._in_channels

    @property
    def hidden_channels(self) -> int:
        """Number of output feature channels."""
        return self._hidden_channels

    @property
    def num_layers(self) -> int:
        """Number of encoder layers."""
        return self._num_layers

    @property
    def kernel_size(self) -> int:
        """Convolution kernel size."""
        return self._kernel_size

    @property
    def dropout(self) -> float:
        """Dropout probability."""
        return self._dropout