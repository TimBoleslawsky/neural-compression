import torch.nn as nn
from .base import BaseEncoder

class ResidualBlock1d(nn.Module):
    """
    Residual block for 1D signals with skip connection.
    
    Architecture: x → Conv → ReLU → Conv → (+) → ReLU
                  ↓________________________↑
                       (skip connection)
    """
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding='same')
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding='same')
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x  # Skip connection
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + identity  # Add skip connection
        out = self.relu(out)
        
        return out


class ResNetEncoder(BaseEncoder):
    """
    ResNet-style encoder for signal compression.
    
    Advantages over SimpleCNN:
    - Skip connections enable deeper networks (12-20 layers)
    - Better gradient flow during training
    - Preserves low-level features through identity mappings
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        # num_blocks is the ResNet-specific parameter for number of residual blocks
        # We pass it as num_layers to the base class for interface consistency
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,  # num_blocks
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # Initial projection to hidden dimension
        self.input_conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size, padding='same'),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU()
        )
        
        # Stack of residual blocks
        self.res_blocks = nn.Sequential(*[
            ResidualBlock1d(hidden_channels, kernel_size, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        x = self.input_conv(x)
        x = self.res_blocks(x)
        return x
