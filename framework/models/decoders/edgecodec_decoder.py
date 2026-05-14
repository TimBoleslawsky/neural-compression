import torch as th
import torch.nn as nn

from .base import BaseDecoder


class Conv1dPadded(nn.Module):
    """
    Conv1d with custom symmetric padding (matches EdgeCodec GitHub).
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
        x = nn.functional.pad(
            x,
            (self.causal_padding // 2, self.causal_padding - self.causal_padding // 2),
            'constant'
        )
        return self.conv1d(x)


class ConvTranspose1dPadded(nn.Module):
    """
    ConvTranspose1d with custom padding (matches EdgeCodec GitHub).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, output_padding: int = 0, bias: bool = True):
        super().__init__()

        self.causal_padding = (kernel_size - 1) + output_padding - stride * 2

        self.convtranspose1d = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            output_padding=output_padding,
            padding=0,
            bias=bias
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = nn.functional.pad(
            x,
            (self.causal_padding // 2, self.causal_padding - self.causal_padding // 2)
        )
        return self.convtranspose1d(x)


class ChannelWiseLinear(nn.Module):
    """
    Channel-wise independent linear transformations (exact reproduction from GitHub).

    Applies a separate Linear(out_features, out_features) to each channel independently.

    Args:
        in_channels: Number of input channels
        out_features: Output temporal dimension per channel
    """

    def __init__(self, in_channels: int, out_features: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_features = out_features

        self.linear_layers = nn.ModuleList([
            nn.Linear(out_features, out_features)
            for _ in range(in_channels)
        ])

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Apply independent linear transformation to each channel.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        outputs = [self.linear_layers[i](x[:, i, :]) for i in range(self.in_channels)]
        return th.stack(outputs, dim=1)


class EdgeCodecResidualUnit(nn.Module):
    """
    Residual unit from EdgeCodec GitHub (exact reproduction).

    Architecture:
        x → Conv1d(k=5, d=dilation) → PReLU → Conv1d(k=1) → (+) x → out

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (should equal in_channels)
        dilation: Dilation factor
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
        return x + self.layers(x)


class EdgeCodecDecoderBlock(nn.Module):
    """
    Decoder block from EdgeCodec GitHub (exact reproduction).

    Architecture:
        x → ConvTranspose1d(k=2*stride, stride) → ELU →
        ResidualUnit(d=1) → ELU →
        ResidualUnit(d=3) → ELU →
        ResidualUnit(d=9) → out

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Upsampling stride
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()

        self.layers = nn.Sequential(
            ConvTranspose1dPadded(
                in_channels=int(in_channels),
                out_channels=int(out_channels),
                kernel_size=2 * stride,
                stride=stride
            ),
            nn.ELU(),

            EdgeCodecResidualUnit(
                in_channels=int(out_channels),
                out_channels=int(out_channels),
                dilation=1
            ),
            nn.ELU(),

            EdgeCodecResidualUnit(
                in_channels=int(out_channels),
                out_channels=int(out_channels),
                dilation=3
            ),
            nn.ELU(),

            EdgeCodecResidualUnit(
                in_channels=int(out_channels),
                out_channels=int(out_channels),
                dilation=9
            ),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.layers(x)


class EdgeCodecDecoder(BaseDecoder):
    """
    EdgeCodec decoder with progressive upsampling.

    Architecture:
        Input → ChannelWiseLinear → Conv1d → DecoderBlocks with residuals
        → ChannelWiseLinear → DecoderBlocks → Conv1d → Output

    Args:
        C: Output channel dimension
        D: Input bottleneck dimension
        window_size: Input sequence length before encoding
        channel_progression: Channel dimensions across decoder blocks
    """

    def __init__(
        self, 
        C: int = 36, 
        D: int = 9, 
        window_size: int = 800, 
        channel_progression: list[int] = None, 
        **kwargs
    ):
        # Initialize base class
        super().__init__(
            embedding_dim=D,
            out_channels=C,
            num_layers=4,  # 4 decoder blocks
            kernel_size=3,
            dropout=0.5,
        )

        self.C = C
        self.D = D
        self.window_size = window_size

        # Calculate sequence lengths at different decoder stages
        encoded_len = window_size // 2
        mid_len = window_size       

        self.layers = nn.Sequential(
            ChannelWiseLinear(D, encoded_len),  # Adapts to encoded sequence length
            nn.ELU(),

            Conv1dPadded(in_channels=D, out_channels=channel_progression[0], kernel_size=1),
            nn.ELU(),

            EdgeCodecDecoderBlock(in_channels=channel_progression[0], out_channels=channel_progression[1], stride=2),  # encoded_len → encoded_len*2
            nn.ELU(),

            EdgeCodecDecoderBlock(in_channels=channel_progression[1], out_channels=channel_progression[2], stride=1),  # encoded_len*2 → encoded_len*4
            ChannelWiseLinear(channel_progression[2], mid_len),     # Adapts to mid-upsampling sequence length
            nn.ELU(),
            nn.Dropout(0.5),

            EdgeCodecDecoderBlock(in_channels=channel_progression[2], out_channels=channel_progression[3], stride=1),  # encoded_len*4 → encoded_len*8
            nn.ELU(),

            EdgeCodecDecoderBlock(in_channels=channel_progression[3], out_channels=C, stride=1),   # Keep encoded_len*8
            nn.ELU(),

            Conv1dPadded(in_channels=C, out_channels=C, kernel_size=3)
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Decode quantized embeddings with progressive upsampling.

        Args:
            x: Input tensor

        Returns:
            Reconstructed signals
        """
        return self.layers(x)