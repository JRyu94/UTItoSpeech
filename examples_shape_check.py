"""Minimal shape check for the proposed multimodal diffusion ATS skeleton."""

import torch

from utitospeech.configs.model_config import ATSConfig
from utitospeech.models.system import ATSBatch, MultimodalDiffusionATS


def main() -> None:
    cfg = ATSConfig()
    model = MultimodalDiffusionATS(cfg, num_speakers=81)

    batch = ATSBatch(
        ultrasound=torch.randn(2, 20, 1, 64, 64),
        lips=torch.randn(2, 20, 3, 64, 64),
        noisy_mel=torch.randn(2, 1, 100, cfg.diffusion.mel_bins),
        speaker_id=torch.randint(0, 81, (2,)),
    )
    out = model(batch)
    print("pred_noise shape:", tuple(out.shape))


if __name__ == "__main__":
    main()
