import torch as th
import torch.nn as nn

from .base import BaseEncoder


class Conv1dPadded(nn.Module):
    """
    Conv1d with custom symmetric padding (matches EdgeCodec GitHub).

    Applies manual padding before convolution to handle dilation properly.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, dilation: int = 1, bias: bool = True):
        super().__init__()

        self.causal_padding = dilation * (kernel_size - 1)
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=0,
            bias=bias,
            padding_mode='zeros'
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        # Apply symmetric padding
        x = nn.functional.pad(
            x,
            (self.causal_padding // 2, self.causal_padding - self.causal_padding // 2),
            'constant'
        )
        return self.conv1d(x)


class EdgeCodecResidualUnit(nn.Module):
    """
    Residual unit from EdgeCodec GitHub (exact reproduction).

    Architecture:
        x → Conv1d(k=5, d=dilation) → PReLU → Conv1d(k=1) → (+) x → out

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (should equal in_channels)
        dilation: Dilation factor (3 for encoder)
    """

    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__()

        self.dilation = dilation

        self.layers = nn.Sequential(
            Conv1dPadded(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=5,
                dilation=dilation
            ),
            nn.PReLU(),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1
            ),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Forward pass: x + layers(x)

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        return x + self.layers(x)


class EdgeCodecEncoderBlock(nn.Module):
    """
    Encoder block from EdgeCodec GitHub (exact reproduction).

    Architecture:
        x → ResidualUnit(d=3) → PReLU → Conv1d(k=2*stride, stride) → out

    Note: Only ONE residual unit per block, not three!

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Downsampling stride
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()

        self.layers = nn.Sequential(
            EdgeCodecResidualUnit(
                in_channels=int(in_channels),
                out_channels=int(in_channels),
                dilation=3
            ),
            nn.PReLU(),
            Conv1dPadded(
                in_channels=int(in_channels),
                out_channels=int(out_channels),
                kernel_size=2 * stride,
                stride=stride
            ),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Forward pass through encoder block.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        return self.layers(x)


class EdgeCodecEncoder(BaseEncoder):
    """
    EdgeCodec encoder with progressive downsampling.

    Architecture:
        Input → Conv1d → EncoderBlocks with residuals → Conv1d → Output

    Args:
        in_channels: Number of input channels
        C: Initial channel expansion
        D: Bottleneck dimension
        channel_progression: Channel dimensions across encoder blocks
    """

    def __init__(
        self,
        in_channels: int = 36,
        C: int = 36,
        D: int = 9,
        channel_progression: list[int] = None, 
        **kwargs
    ):
        # Initialize base class
        super().__init__(
            in_channels=in_channels,
            hidden_channels=D,
            num_layers=4,  # 4 encoder blocks
            kernel_size=7,
            dropout=0.0,
        )

        self.C = C
        self.D = D

        self.layers = nn.Sequential(
            Conv1dPadded(in_channels=in_channels, out_channels=C, kernel_size=7),
            nn.PReLU(),

            EdgeCodecEncoderBlock(in_channels=C, out_channels=channel_progression[0], stride=1),
            nn.PReLU(),

            EdgeCodecEncoderBlock(in_channels=channel_progression[0], out_channels=channel_progression[1], stride=1),
            nn.PReLU(),

            EdgeCodecEncoderBlock(in_channels=channel_progression[1], out_channels=channel_progression[2], stride=1),
            nn.PReLU(),

            EdgeCodecEncoderBlock(in_channels=channel_progression[2], out_channels=channel_progression[3], stride=2),
            nn.PReLU(),

            Conv1dPadded(in_channels=channel_progression[3], out_channels=D, kernel_size=1),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Encode input signals with progressive downsampling.

        Args:
            x: Input tensor

        Returns:
            Encoded features
        """
        return self.layers(x)
