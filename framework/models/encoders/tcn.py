import torch as th
import torch.nn as nn
from .base import BaseEncoder


class CausalConv1d(nn.Module):
    """
    1D Causal Convolution with left-padding.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolution kernel
        dilation: Dilation factor
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        **kwargs
    ):
        super().__init__()

        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=0,
            **kwargs
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Forward pass with causal padding.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        x = nn.functional.pad(x, (self.padding, 0))
        x = self.conv(x)
        return x


class TCNResidualBlock(nn.Module):
    """
    TCN Residual Block with two causal convolutions and residual connection.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolution kernels
        dilation: Dilation factor
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv1.conv = nn.utils.weight_norm(self.conv1.conv)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.conv2.conv = nn.utils.weight_norm(self.conv2.conv)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.residual = nn.utils.weight_norm(
                nn.Conv1d(in_channels, out_channels, kernel_size=1)
            )
        else:
            self.residual = nn.Identity()

        self.relu_out = nn.ReLU()

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Forward pass through the residual block.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.relu_out(out + residual)
        return out


class TCNEncoder(BaseEncoder):
    """
    Stack of TCN Residual Blocks with exponentially increasing dilation factors.

    Args:
        in_channels: Number of input signal channels
        hidden_channels: Number of channels in TCN layers
        num_layers: Number of TCN residual blocks
        kernel_size: Size of convolution kernels
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout
        )

        # Build stack of TCN blocks with exponentially increasing dilation
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i  # Exponential dilation: 1, 2, 4, 8, ...

            # First layer maps from in_channels to hidden_channels
            # Subsequent layers maintain hidden_channels
            in_ch = in_channels if i == 0 else hidden_channels

            layers.append(
                TCNResidualBlock(
                    in_channels=in_ch,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Encode input signals through the TCN stack.

        Args:
            x: Input tensor of shape (batch, in_channels, seq_len)

        Returns:
            Encoded tensor of shape (batch, hidden_channels, seq_len)
        """
        return self.network(x)

    @property
    def receptive_field(self) -> int:
        """
        Calculate the theoretical receptive field of the TCN encoder.

        Each residual block has 2 causal convolutions with the same dilation.
        The receptive field grows with dilation pattern 2^i for layer i.

        Returns:
            Number of timesteps in the receptive field
        """
        # Calculate exact receptive field
        # Each TCNResidualBlock has 2 causal convs with same dilation
        # RF contribution per block = 2 * (kernel_size - 1) * dilation
        # Total RF = 1 + sum of all blocks' contributions

        rf = 1
        for i in range(self.num_layers):
            dilation = 2 ** i
            rf += 2 * (self.kernel_size - 1) * dilation

        return rf
