from __future__ import annotations

import torch
from torch import nn


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, channels: int) -> None:
        super().__init__()
        self.scale = nn.Linear(cond_dim, channels)
        self.shift = nn.Linear(cond_dim, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale = self.scale(cond).unsqueeze(-1).unsqueeze(-1)
        shift = self.shift(cond).unsqueeze(-1).unsqueeze(-1)
        return x * (1 + scale) + shift


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act = nn.GELU()
        self.film = FiLM(cond_dim=cond_dim, channels=channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.film(h, cond)
        h = self.act(h)
        h = self.conv2(h)
        h = self.norm2(h)
        return self.act(h + x)


class ConditionProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(feat.transpose(1, 2)).squeeze(-1)
        return self.proj(pooled)


class MelCrossAttention(nn.Module):
    """Injects articulatory temporal features into mel tokens."""

    def __init__(self, model_dim: int, cond_dim: int, heads: int) -> None:
        super().__init__()
        self.query_proj = nn.Linear(model_dim, model_dim)
        self.key_proj = nn.Linear(cond_dim, model_dim)
        self.value_proj = nn.Linear(cond_dim, model_dim)
        self.attn = nn.MultiheadAttention(model_dim, heads, batch_first=True)

    def forward(self, mel_feat: torch.Tensor, artic_feat: torch.Tensor) -> torch.Tensor:
        # mel_feat: [B, C, T, F] -> tokens [B, T*F, C]
        b, c, t, f = mel_feat.shape
        tokens = mel_feat.permute(0, 2, 3, 1).reshape(b, t * f, c)
        q = self.query_proj(tokens)
        k = self.key_proj(artic_feat)
        v = self.value_proj(artic_feat)
        ctx, _ = self.attn(q, k, v)
        out = (tokens + ctx).reshape(b, t, f, c).permute(0, 3, 1, 2)
        return out


class ConditionalDiffusionDecoder(nn.Module):
    """UNet-like diffusion denoiser head for mel [B, 1, T, F]."""

    def __init__(
        self,
        mel_bins: int,
        model_dim: int,
        depth: int,
        cond_dim: int,
        speaker_dim: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.mel_bins = mel_bins
        self.in_conv = nn.Conv2d(1, model_dim, kernel_size=3, padding=1)
        self.cross_attn = MelCrossAttention(model_dim=model_dim, cond_dim=cond_dim, heads=heads)
        self.blocks = nn.ModuleList([ResidualBlock(model_dim, cond_dim + speaker_dim) for _ in range(depth)])
        self.to_noise = nn.Conv2d(model_dim, 1, kernel_size=3, padding=1)

    def forward(
        self,
        noisy_mel: torch.Tensor,
        artic_seq: torch.Tensor,
        pooled_cond: torch.Tensor,
        spk_embed: torch.Tensor,
    ) -> torch.Tensor:
        fused_cond = torch.cat([pooled_cond, spk_embed], dim=-1)
        x = self.in_conv(noisy_mel)
        x = self.cross_attn(x, artic_seq)
        for blk in self.blocks:
            x = blk(x, fused_cond)
        return self.to_noise(x)
