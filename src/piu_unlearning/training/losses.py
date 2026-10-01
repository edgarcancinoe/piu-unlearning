from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from diffusers import DDPMScheduler, UNet2DConditionModel

    from piu_unlearning.models.arc2face import Arc2FaceIdentityConditioner


@dataclass(frozen=True)
class LossOutput:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]
    gradients: tuple[torch.Tensor, ...] | None = None


@dataclass(frozen=True)
class NoisePredictionContext:
    model: UNet2DConditionModel
    teacher: UNet2DConditionModel
    scheduler: DDPMScheduler
    conditioner: Arc2FaceIdentityConditioner
    reference_conditioning: torch.Tensor
    device: torch.device


def sample_noisy_latents(batch_size: int, model: UNet2DConditionModel, scheduler: DDPMScheduler, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
    latent_shape = (batch_size, model.config.in_channels, model.config.sample_size, model.config.sample_size)
    latents = torch.randn(latent_shape, device=device, dtype=dtype)
    noise = torch.randn_like(latents)
    return scheduler.add_noise(latents, noise, timesteps), timesteps


class GuidedNoiseLoss:
    """Teacher-guided erasure with optional retain distillation, used by PIU."""

    def __init__(self, preservation_weight: float, negative_guidance_scale: float) -> None:
        self.preservation_weight = preservation_weight
        self.negative_guidance_scale = negative_guidance_scale

    def compute_loss(self, forget_batch: torch.Tensor, retain_batch: torch.Tensor | None, context: NoisePredictionContext) -> LossOutput:
        model, teacher, scheduler = context.model, context.teacher, context.scheduler
        preserve = self.preservation_weight > 0
        with torch.no_grad():
            forget_conditioning = context.conditioner.encode(forget_batch.to(context.device))
            if preserve: retain_conditioning = context.conditioner.encode(retain_batch.to(context.device))
        reference = context.reference_conditioning.expand(len(forget_batch), -1, -1)
        forget_latents, forget_timesteps = sample_noisy_latents(len(forget_batch), model, scheduler, context.device, forget_conditioning.dtype)
        if preserve: retain_latents, retain_timesteps = sample_noisy_latents(len(retain_batch), model, scheduler, context.device, retain_conditioning.dtype)

        # Keep PIU's sampling and forward-pass order for reproducibility.
        with torch.no_grad():
            noise_reference = teacher(forget_latents, forget_timesteps, encoder_hidden_states=reference).sample
            noise_forget = teacher(forget_latents, forget_timesteps, encoder_hidden_states=forget_conditioning).sample
            if preserve: noise_retain = teacher(retain_latents, retain_timesteps, encoder_hidden_states=retain_conditioning).sample
        prediction = model(forget_latents, forget_timesteps, encoder_hidden_states=forget_conditioning).sample
        if preserve: retain_prediction = model(retain_latents, retain_timesteps, encoder_hidden_states=retain_conditioning).sample
        target = noise_reference - self.negative_guidance_scale * (noise_forget - noise_reference)
        forget_loss = F.mse_loss(prediction.float(), target.float())
        preserve_loss = F.mse_loss(retain_prediction.float(), noise_retain.float()) if preserve else forget_loss.new_zeros(())
        return LossOutput(forget_loss + self.preservation_weight * preserve_loss, {"forget_loss": forget_loss, "preserve_loss": preserve_loss})
