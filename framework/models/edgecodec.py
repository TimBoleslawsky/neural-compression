import torch as th
import torch.nn as nn
from typing import Dict, Any

from .encoders import BaseEncoder
from .quantizers import BaseQuantizer
from .decoders import BaseDecoder


class EdgeCodecModel(nn.Module):
    """
    Complete EdgeCodec model with encoder-quantizer-decoder pipeline.

    Architecture:
        Input → Encoder: progressive downsampling
        → RVQ: residual quantization
        → Decoder: progressive upsampling → Output

    Args:
        encoder_module: EdgeCodec encoder instance
        quantizer_module: Residual vector quantizer instance
        decoder_module: EdgeCodec decoder instance

    Attributes:
        vq_loss: VQ commitment loss (set during forward pass)
        perplexity: Codebook usage metric (set during forward pass)
    """

    def __init__(
        self,
        encoder_module: BaseEncoder,
        quantizer_module: BaseQuantizer,
        decoder_module: BaseDecoder,
        use_discriminator: bool
    ):
        super().__init__()

        self.encoder_module = encoder_module
        self.quantizer_module = quantizer_module
        self.decoder_module = decoder_module

        # Attributes set during forward pass (for training module access)
        self.vq_loss = None
        self.perplexity = None

    def encode(self, x: th.Tensor) -> th.Tensor:
        """
        Encode input signals to quantized embeddings.

        Args:
            x: Input signals (batch, channels, timesteps)

        Returns:
            Quantized embeddings (batch, embedding_dim, compressed_timesteps)
        """
        z = self.encoder_module(x)

        # Channel-wise quantization (following paper's approach):
        # Keep as (B, C, T) so RVQ treats each channel as a vector of timesteps
        # Each of C channels becomes a vector of dimension T
        # With num_quantizers=4, total codes = C × 4
        quantized, vq_loss, perplexity = self.quantizer_module(z)

        # Store VQ metrics for training
        self.vq_loss = vq_loss
        self.perplexity = perplexity

        return quantized

    def decode(self, embeddings: th.Tensor) -> th.Tensor:
        """
        Decode quantized embeddings to reconstructed signals.

        Args:
            embeddings: Quantized embeddings (batch, embedding_dim, compressed_timesteps)

        Returns:
            Reconstructed signals (batch, channels, timesteps)
        """
        return self.decoder_module(embeddings)

    def forward(self, batch: Dict[str, Any]) -> th.Tensor:
        """
        Full forward pass through EdgeCodec pipeline.

        Args:
            batch: Dict containing at least "encode": Tensor (B, C, T).

        Returns:
            Reconstructed signals tensor of shape (batch, channels, timesteps)
        """
        x = batch["encode"]

        # Encode and quantize
        quantized = self.encode(x)

        # Decode to reconstruct
        reconstructed = self.decode(quantized)

        return reconstructed
    
    
class EdgeCodecDiscriminator(nn.Module): 

    def __init__(self, input_channels = 36):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=15, stride=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=7, stride=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=7, stride=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=7, stride=2),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool1d(1),   
            nn.Flatten(),

            nn.Linear(512,1),
            
        )

    def forward(self,x):
        return self.layers(x)
