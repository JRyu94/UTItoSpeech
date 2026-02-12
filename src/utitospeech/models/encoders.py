from __future__ import annotations

import torch
from torch import nn


class FrameCNN(nn.Module):
    """Lightweight ResNet-style frame encoder for U/L images."""

    def __init__(self, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(128, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C, H, W]
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        feat = self.net(x).flatten(1)
        feat = self.proj(feat)
        return feat.reshape(b, t, -1)


class TemporalEncoder(nn.Module):
    def __init__(self, dim: int, heads: int, layers: int, dropout: float) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dropout=dropout,
            dim_feedforward=dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class VideoStreamEncoder(nn.Module):
    """2D CNN + temporal Transformer."""

    def __init__(
        self,
        in_channels: int,
        frame_embed_dim: int,
        temporal_dim: int,
        temporal_heads: int,
        temporal_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.frame = FrameCNN(in_channels=in_channels, embed_dim=frame_embed_dim)
        self.in_proj = nn.Linear(frame_embed_dim, temporal_dim)
        self.temporal = TemporalEncoder(
            dim=temporal_dim,
            heads=temporal_heads,
            layers=temporal_layers,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.frame(x)
        x = self.in_proj(x)
        return self.temporal(x)
