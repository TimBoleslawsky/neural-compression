import torch as th
import torch.nn as nn
from typing import Dict, Any, Optional, List

from .encoders import BaseEncoder
from .quantizers import BaseQuantizer
from .decoders import BaseDecoder


class UniEdgeCodecModel(nn.Module):
    """
    Signal-adaptive EdgeCodec with partitioned VQ codebook and dual encoders.

    Args:
        encoder_module: EdgeCodec encoder for smooth signals
        quantizer_module: SignalAdaptiveVQ with partitioned codebook
        decoder_module: EdgeCodec decoder
        signal_type_map: Dict mapping signal names to types

    Attributes:
        vq_loss: VQ commitment loss (set during forward pass)
        perplexity: Dict with per-partition perplexity metrics
    """

    def __init__(
        self,
        encoder_module: BaseEncoder,
        quantizer_module: BaseQuantizer,
        decoder_module: BaseDecoder,
        signal_type_ids: Optional[List[int]] = None,
    ):
        super().__init__()

        self.encoder_module = encoder_module
        self.quantizer_module = quantizer_module
        self.decoder_module = decoder_module

        if signal_type_ids is None:
            raise ValueError("UniEdgeCodecModel requires signal_type_ids aligned to channel order")
        signal_types_tensor = th.tensor(list(signal_type_ids), dtype=th.long)
        self.register_buffer('_signal_types_base', signal_types_tensor)

        self.vq_loss = None
        self.perplexity = None

    def encode(
        self,
        x: th.Tensor,
        signal_types: Optional[th.Tensor] = None
    ) -> th.Tensor:
        """
        Encode input signals with signal-type-aware architecture and quantization.

        Args:
            x: Input signals
            signal_types: Signal type IDs or None

        Returns:
            Quantized embeddings
        """
        z = self.encoder_module(x)
        quantized, vq_loss, perplexity = self.quantizer_module(z, signal_types)
        self.vq_loss = vq_loss
        self.perplexity = perplexity

        return quantized

    def decode(self, embeddings: th.Tensor) -> th.Tensor:
        """
        Decode quantized embeddings to reconstructed signals.

        Args:
            embeddings: Quantized embeddings

        Returns:
            Reconstructed signals
        """
        return self.decoder_module(embeddings)

    def forward(self, batch: Dict[str, Any]) -> th.Tensor:
        """
        Full forward pass with signal-type-aware quantization.

        Args:
            batch: Dict containing at least "encode": Tensor (B, C, T).

        Returns:
            Reconstructed signals tensor (batch, channels, timesteps)
        """
        x = batch["encode"]

        batch_size = x.shape[0]

        # Expand signal types from (num_channels,) to (batch, num_channels)
        signal_types = self._signal_types_base.unsqueeze(0).expand(batch_size, -1)

        # Encode with signal-type-aware quantization
        quantized = self.encode(x, signal_types)

        # Decode to reconstruct
        reconstructed = self.decode(quantized)

        return reconstructed
