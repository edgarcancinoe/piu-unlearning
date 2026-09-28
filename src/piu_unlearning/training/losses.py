from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from diffusers import DDPMScheduler, UNet2DConditionModel

    from piu_unlearning.models.arc2face import Arc2FaceIdentityConditioner


def sample_noisy_latents(batch_size: int, model: UNet2DConditionModel, scheduler: DDPMScheduler, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
    latent_shape = (batch_size, model.config.in_channels, model.config.sample_size, model.config.sample_size)
    latents = torch.randn(latent_shape, device=device, dtype=dtype)
    noise = torch.randn_like(latents)
    return scheduler.add_noise(latents, noise, timesteps), timesteps


class PIU:
    def __init__(self, preservation_weight: float, negative_guidance_scale: float) -> None:
        self.preservation_weight = preservation_weight
        self.negative_guidance_scale = negative_guidance_scale

    def compute_piu_loss(self, forget_batch: torch.Tensor, retain_batch: torch.Tensor, anchor_conditioning: torch.Tensor, model: UNet2DConditionModel, model_frozen: UNet2DConditionModel, scheduler: DDPMScheduler, identity_conditioner: Arc2FaceIdentityConditioner, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        forget_embeddings = forget_batch.to(device)
        retain_embeddings = retain_batch.to(device)
        batch_size = forget_embeddings.shape[0]

        with torch.no_grad():
            forget_conditioning = identity_conditioner.encode(forget_embeddings)
            retain_conditioning = identity_conditioner.encode(retain_embeddings)
        anchor_conditioning_batch = anchor_conditioning.expand(batch_size, -1, -1)

        forget_noisy_latents, forget_timesteps = sample_noisy_latents(batch_size, model, scheduler, device, forget_conditioning.dtype)
        retain_noisy_latents, retain_timesteps = sample_noisy_latents(batch_size, model, scheduler, device, retain_conditioning.dtype)

        with torch.no_grad():
            noise_anchor = model_frozen(forget_noisy_latents, forget_timesteps, encoder_hidden_states=anchor_conditioning_batch).sample
            noise_forget = model_frozen(forget_noisy_latents, forget_timesteps, encoder_hidden_states=forget_conditioning).sample
            noise_retain = model_frozen(retain_noisy_latents, retain_timesteps, encoder_hidden_states=retain_conditioning).sample

        pred_noise_forget = model(forget_noisy_latents, forget_timesteps, encoder_hidden_states=forget_conditioning).sample
        pred_noise_retain = model(retain_noisy_latents, retain_timesteps, encoder_hidden_states=retain_conditioning).sample

        piu_target = noise_anchor - self.negative_guidance_scale * (noise_forget - noise_anchor)
        forget_loss = torch.nn.functional.mse_loss(pred_noise_forget.float(), piu_target.float(), reduction="mean")
        preserve_loss = torch.nn.functional.mse_loss(pred_noise_retain.float(), noise_retain.float(), reduction="mean")
        loss = forget_loss + self.preservation_weight * preserve_loss
        return loss, forget_loss, preserve_loss
