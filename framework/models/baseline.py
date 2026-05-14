import torch as th
from torch import nn
from .encoders import BaseEncoder
from .decoders import BaseDecoder


class BaselineCompressionModel(nn.Module):
    """
    One-stage compression with continuous latent codes.

    Architecture:
    1. Encoder: Pluggable encoder module → Linear projection
    2. Decoder: Pluggable decoder module

    Args:
        encoder_module: Encoder module instance (e.g., TCNEncoder, SimpleCNNEncoder)
        decoder_module: Decoder module instance (e.g., GRUDecoder)
        embedding_dim: Embedding dimension for latents
    """
    def __init__(
        self,
        encoder_module: BaseEncoder = None,
        decoder_module: BaseDecoder = None,
        embedding_dim: int = 1,
    ):
        super(BaselineCompressionModel, self).__init__()

        self.in_channels = encoder_module.in_channels
        self.hidden_channels = encoder_module.hidden_channels
        self.embedding_dim = embedding_dim

        # 1. Encoder module
        self.encoder_module = encoder_module

        # 2. Embedding projection: bottleneck compression layer
        self.fc_embed = nn.Linear(encoder_module.hidden_channels, embedding_dim)
        self.activation = nn.Tanh()

        # 3. Decoder module
        self.decoder_module = decoder_module

    def encode(self, x: th.Tensor) -> th.Tensor:
        """
        Encode input signals to compressed embeddings.

        Args:
            x: Input signals (batch, in_channels, seq_len)

        Returns:
            Embeddings (batch, seq_len, embedding_dim)
        """
        encoded = self.encoder_module(x)
        encoded = encoded.transpose(1, 2)
        embedded = self.fc_embed(encoded)
        embedded = self.activation(embedded)
        return embedded

    def decode(self, embeddings: th.Tensor) -> th.Tensor:
        """
        Decode embeddings back to signal space.

        Args:
            embeddings: Compressed embeddings (batch, seq_len, embedding_dim)

        Returns:
            Reconstructed signals (batch, in_channels, seq_len)
        """
        return self.decoder_module(embeddings)

    def forward(self, batch: dict[str, th.Tensor]) -> th.Tensor:
        """
        Forward pass: encode → compress → decode → reconstruct.

        Args:
            batch: Dictionary with "encode" and "decode"

        Returns:
            Reconstructed signals tensor
        """
        x = batch["encode"]

        embedded = self.encode(x)
        reconstructed = self.decode(embedded)

        return reconstructed
