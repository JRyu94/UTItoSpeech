from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from utitospeech.configs.model_config import ATSConfig
from utitospeech.models.diffusion_decoder import ConditionProjector, ConditionalDiffusionDecoder
from utitospeech.models.encoders import VideoStreamEncoder
from utitospeech.models.fusion import CrossModalFusion


@dataclass
class ATSBatch:
    ultrasound: torch.Tensor | None  # [B, T, C, H, W]
    lips: torch.Tensor | None  # [B, T, C, H, W]
    noisy_mel: torch.Tensor  # [B, 1, Tm, F]
    speaker_id: torch.Tensor  # [B]


class MultimodalDiffusionATS(nn.Module):
    """Unified U / L / U+L conditional diffusion ATS model."""

    def __init__(self, cfg: ATSConfig, num_speakers: int) -> None:
        super().__init__()
        self.cfg = cfg

        if cfg.ultrasound.temporal_dim != cfg.fusion.hidden_dim or cfg.lips.temporal_dim != cfg.fusion.hidden_dim:
            raise ValueError("Encoder temporal_dim must match fusion.hidden_dim for now.")

        self.ultrasound_encoder = VideoStreamEncoder(**cfg.ultrasound.__dict__)
        self.lips_encoder = VideoStreamEncoder(**cfg.lips.__dict__)

        self.fusion = CrossModalFusion(
            dim=cfg.fusion.hidden_dim,
            mode=cfg.fusion.mode,
            heads=cfg.fusion.heads,
            dropout=cfg.fusion.dropout,
        )
        self.speaker_embed = nn.Embedding(num_speakers, cfg.diffusion.speaker_dim)
        self.cond_projector = ConditionProjector(in_dim=cfg.fusion.hidden_dim, out_dim=cfg.diffusion.model_dim)
        self.decoder = ConditionalDiffusionDecoder(
            mel_bins=cfg.diffusion.mel_bins,
            model_dim=cfg.diffusion.model_dim,
            depth=cfg.diffusion.depth,
            cond_dim=cfg.diffusion.model_dim,
            speaker_dim=cfg.diffusion.speaker_dim,
            heads=cfg.diffusion.heads,
        )

    def encode_modalities(self, ultrasound: torch.Tensor | None, lips: torch.Tensor | None) -> torch.Tensor:
        u_feat = self.ultrasound_encoder(ultrasound) if ultrasound is not None else None
        l_feat = self.lips_encoder(lips) if lips is not None else None
        return self.fusion(u_feat, l_feat)

    def forward(self, batch: ATSBatch) -> torch.Tensor:
        artic_seq = self.encode_modalities(batch.ultrasound, batch.lips)
        pooled_cond = self.cond_projector(artic_seq)
        spk_embed = self.speaker_embed(batch.speaker_id)
        pred_noise = self.decoder(batch.noisy_mel, artic_seq=artic_seq, pooled_cond=pooled_cond, spk_embed=spk_embed)
        return pred_noise
