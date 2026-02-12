from dataclasses import dataclass, field


@dataclass
class EncoderConfig:
    in_channels: int
    frame_embed_dim: int = 128
    temporal_dim: int = 256
    temporal_heads: int = 4
    temporal_layers: int = 3
    dropout: float = 0.1


@dataclass
class FusionConfig:
    mode: str = "cross_attention"  # one of: concat, cross_attention
    hidden_dim: int = 256
    heads: int = 4
    dropout: float = 0.1


@dataclass
class DiffusionConfig:
    mel_bins: int = 80
    model_dim: int = 256
    heads: int = 4
    depth: int = 4
    speaker_dim: int = 128


@dataclass
class ATSConfig:
    ultrasound: EncoderConfig = field(default_factory=lambda: EncoderConfig(in_channels=1))
    lips: EncoderConfig = field(default_factory=lambda: EncoderConfig(in_channels=3))
    fusion: FusionConfig = field(default_factory=FusionConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
