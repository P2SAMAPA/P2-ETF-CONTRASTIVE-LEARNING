"""
1D‑CNN encoder with projection head for contrastive learning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    """Lightweight 1D‑CNN encoder for time‑series windows."""
    def __init__(self, input_dim: int, window_size: int, embedding_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.window_size = window_size

        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, embedding_dim)

    def forward(self, x):
        # x: [batch, window_size, input_dim]
        x = x.permute(0, 2, 1)  # [batch, input_dim, window_size]

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.pool(x).squeeze(-1)  # [batch, 128]
        x = self.fc(x)                # [batch, embedding_dim]
        return F.normalize(x, dim=-1)


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class ContrastiveModel(nn.Module):
    """Encoder + Projection head for SimCLR‑style training."""
    def __init__(self, input_dim: int, window_size: int, embedding_dim: int, projection_dim: int):
        super().__init__()
        self.encoder = CNNEncoder(input_dim, window_size, embedding_dim)
        self.projector = ProjectionHead(embedding_dim, embedding_dim * 2, projection_dim)

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return h, z
