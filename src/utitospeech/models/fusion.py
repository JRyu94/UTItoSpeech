from __future__ import annotations

import torch
from torch import nn


class CrossModalFusion(nn.Module):
    """Supports concat and bidirectional cross-attention fusion."""

    def __init__(self, dim: int, mode: str = "cross_attention", heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.mode = mode
        self.norm_u = nn.LayerNorm(dim)
        self.norm_l = nn.LayerNorm(dim)

        if mode == "concat":
            self.out = nn.Linear(dim * 2, dim)
        elif mode == "cross_attention":
            self.u_to_l = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
            self.l_to_u = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
            self.ff = nn.Sequential(
                nn.Linear(dim * 2, dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 2, dim),
            )
        else:
            raise ValueError(f"Unsupported fusion mode: {mode}")

    def forward(self, u_feat: torch.Tensor | None, l_feat: torch.Tensor | None) -> torch.Tensor:
        if u_feat is None and l_feat is None:
            raise ValueError("At least one modality must be provided.")
        if u_feat is None:
            return l_feat
        if l_feat is None:
            return u_feat

        if self.mode == "concat":
            return self.out(torch.cat([u_feat, l_feat], dim=-1))

        u_q = self.norm_u(u_feat)
        l_q = self.norm_l(l_feat)

        u_ctx, _ = self.u_to_l(query=u_q, key=l_q, value=l_q)
        l_ctx, _ = self.l_to_u(query=l_q, key=u_q, value=u_q)
        fused = torch.cat([u_ctx + u_feat, l_ctx + l_feat], dim=-1)
        return self.ff(fused)
