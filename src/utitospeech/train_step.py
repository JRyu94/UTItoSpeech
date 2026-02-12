from __future__ import annotations

import torch
import torch.nn.functional as F

from utitospeech.models.system import ATSBatch, MultimodalDiffusionATS


def diffusion_train_step(model: MultimodalDiffusionATS, batch: ATSBatch) -> torch.Tensor:
    """Simple epsilon-prediction training objective.

    Notes:
        - Placeholder scheduler: alpha sampled implicitly via noise scale.
        - Replace with DDPM/DDIM scheduler in real experiments.
    """

    clean_mel = batch.noisy_mel
    noise = torch.randn_like(clean_mel)
    sigma = torch.rand(clean_mel.shape[0], 1, 1, 1, device=clean_mel.device)
    noisy = clean_mel + sigma * noise

    noisy_batch = ATSBatch(
        ultrasound=batch.ultrasound,
        lips=batch.lips,
        noisy_mel=noisy,
        speaker_id=batch.speaker_id,
    )
    pred_noise = model(noisy_batch)
    return F.mse_loss(pred_noise, noise)
