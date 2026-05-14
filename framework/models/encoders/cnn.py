import torch.nn as nn
from .base import BaseEncoder

class SimpleCNNEncoder(BaseEncoder):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int = 2,
            num_layers: int = 3,
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

        layers = []
        current_channels = in_channels

        for _ in range(num_layers):
            layers.extend([
                nn.Conv1d(
                    current_channels,
                    hidden_channels,
                    kernel_size,
                    padding="same"
                ),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_channels = hidden_channels

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)
