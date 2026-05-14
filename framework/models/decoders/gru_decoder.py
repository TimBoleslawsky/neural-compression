import torch as th
import torch.nn as nn

from .base import BaseDecoder


class GRUDecoder(BaseDecoder):
    """
    GRU-based decoder with reverse-order reconstruction.

    Architecture:
        embeddings → reverse → GRU → reverse → Linear → signals

    Args:
        embedding_dim: Input embedding dimension
        out_channels: Number of output signal channels
        gru_hidden_size: GRU hidden state dimension
        gru_num_layers: Number of stacked GRU layers
        dropout: Dropout probability
        bidirectional: Whether to use bidirectional GRU
    """

    def __init__(
        self,
        embedding_dim: int,
        out_channels: int,
        gru_hidden_size: int = 2,
        gru_num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        # Initialize base class
        super().__init__(
            embedding_dim=embedding_dim,
            out_channels=out_channels,
            num_layers=gru_num_layers,
            kernel_size=0,  # Not applicable for GRU
            dropout=dropout,
        )

        self.gru_hidden_size = gru_hidden_size
        self.bidirectional = bidirectional

        # GRU: processes embeddings in reverse temporal order
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if gru_num_layers > 1 else 0.0
        )

        # Output projection: map GRU output back to signal space
        gru_output_size = gru_hidden_size * 2 if bidirectional else gru_hidden_size
        self.fc_out = nn.Linear(gru_output_size, out_channels)

    def forward(self, embeddings: th.Tensor) -> th.Tensor:
        """
        Decode embeddings back to signal space using reverse-order reconstruction.

        Args:
            embeddings: Compressed embeddings

        Returns:
            Reconstructed signals
        """
        embeddings_reversed = th.flip(embeddings, dims=[1])

        # 2. GRU decoding: processes sequence in reverse temporal order
        # Input: (batch, seq_len, embedding_dim) - reversed time [T → 1]
        # Output: (batch, seq_len, gru_hidden_size) - still in reversed order
        gru_out, _ = self.gru(embeddings_reversed)

        # 3. Reverse back to original temporal order
        # The GRU output is in reverse time [T → 1], flip it back to [1 → T]
        gru_out = th.flip(gru_out, dims=[1])

        # 4. Output projection: map back to original signal dimension
        # Input: (batch, seq_len, gru_output_size)
        # Output: (batch, seq_len, out_channels)
        reconstructed = self.fc_out(gru_out)

        # Transpose to (batch, out_channels, seq_len) to match input format
        reconstructed = reconstructed.transpose(1, 2)

        return reconstructed  # (batch, out_channels, seq_len)
