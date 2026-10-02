from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from piu_unlearning.config import WIDConfig
from piu_unlearning.data import ExperimentSplit, PairedImageDataset, load_image_manifest
from piu_unlearning.methods.reference import ReferenceIdentity, proximity_reference
from piu_unlearning.models.identity_encoder import IdentityEncoder, load_identity_encoder, resolve_identity_checkpoint
from piu_unlearning.dataset.hub import sha256
from piu_unlearning.training.losses import LossOutput, NoisePredictionContext, sample_noisy_latents


@dataclass(frozen=True)
class WIDInputs:
    dataset: PairedImageDataset
    reference: ReferenceIdentity
    encoder: IdentityEncoder | None
    identity_target: torch.Tensor | None
    metadata: dict


@dataclass(frozen=True)
class WIDContext:
    noise: NoisePredictionContext
    vae: nn.Module
    identity_encoder: nn.Module | None


def prepare_wid_inputs(split: ExperimentSplit, config: WIDConfig) -> WIDInputs:
    """Validate real images and encode each forget-training image as its own target."""
    # Anchor selection is CPU-only here so preflight does not allocate diffusion weights.
    from dataclasses import replace

    reference = proximity_reference(split, replace(config, device="cpu"))
    paths, manifest = load_image_manifest(config, split.forget_train.indices)
    dataset = PairedImageDataset(split.forget_train, paths)
    metadata = {"manifest": manifest, "target_mode": "original_images" if config.identity_loss_weight else "disabled", "forget_indices": split.forget_train.indices.tolist(), "anchor_id": reference.identity_id}
    if not config.identity_loss_weight: return WIDInputs(dataset, reference, None, None, metadata)
    checkpoint = resolve_identity_checkpoint(config.identity_checkpoint)
    encoder = load_identity_encoder(checkpoint, config.identity_channel_order)
    with torch.no_grad():
        identity_target = torch.cat([encoder(batch["pixel_values"]) for batch in DataLoader(dataset, batch_size=config.batch_size)])
    dataset = PairedImageDataset(split.forget_train, paths, identity_target)
    metadata.update(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint), backbone="ir_se50", channel_order=config.identity_channel_order, preprocessing="resize112_bilinear_full_image_minus1_to1", identity_loss="mean_mse", target_dimension=identity_target.shape[1])
    return WIDInputs(dataset, reference, encoder, identity_target.detach(), metadata)


def reconstruct_clean_latents(noisy: torch.Tensor, prediction: torch.Tensor, timesteps: torch.Tensor, scheduler) -> torch.Tensor:
    alpha = scheduler.alphas_cumprod.to(noisy.device)[timesteps].reshape(-1, 1, 1, 1)
    return (noisy - (1 - alpha).sqrt() * prediction) / alpha.sqrt()


class WID:
    prepare_reference = staticmethod(proximity_reference)

    def __init__(self, model_loss_weight: float, identity_loss_weight: float, preservation_weight: float):
        self.model_loss_weight = model_loss_weight
        self.identity_loss_weight = identity_loss_weight
        self.preservation_weight = preservation_weight

    def compute_loss(self, forget_batch: dict, retain_batch: torch.Tensor | None, context: WIDContext) -> LossOutput:
        noise, vae = context.noise, context.vae
        pixels = forget_batch["pixel_values"].to(noise.device)
        timesteps = torch.randint(0, noise.scheduler.config.num_train_timesteps, (len(pixels),), device=noise.device)
        with torch.no_grad():
            conditioning = noise.conditioner.encode(forget_batch["face_embs"].to(noise.device))
            clean = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
            noisy = noise.scheduler.add_noise(clean, torch.randn_like(clean), timesteps)
            target = noise.teacher(noisy, timesteps, encoder_hidden_states=noise.reference_conditioning.expand(len(pixels), -1, -1)).sample
        prediction = noise.model(noisy, timesteps, encoder_hidden_states=conditioning).sample
        model_loss = F.mse_loss(prediction.float(), target.float())
        identity_loss, preserve_loss = model_loss.new_zeros(()), model_loss.new_zeros(())
        if self.identity_loss_weight:
            reconstruction = reconstruct_clean_latents(noisy, prediction, timesteps, noise.scheduler)
            decoded = vae.decode(reconstruction / vae.config.scaling_factor).sample
            identities = context.identity_encoder(decoded)
            identity_loss = F.mse_loss(identities, forget_batch["identity_target"].to(noise.device))
        if self.preservation_weight:
            with torch.no_grad(): retain_conditioning = noise.conditioner.encode(retain_batch.to(noise.device))
            latents, retain_timesteps = sample_noisy_latents(len(retain_batch), noise.model, noise.scheduler, noise.device, conditioning.dtype)
            with torch.no_grad(): retain_target = noise.teacher(latents, retain_timesteps, encoder_hidden_states=retain_conditioning).sample
            retain_prediction = noise.model(latents, retain_timesteps, encoder_hidden_states=retain_conditioning).sample
            preserve_loss = F.mse_loss(retain_prediction.float(), retain_target.float())
        model_term, identity_term = self.model_loss_weight * model_loss, self.identity_loss_weight * identity_loss
        forget_loss = model_term + identity_term
        total = forget_loss + self.preservation_weight * preserve_loss
        if not torch.isfinite(total): raise FloatingPointError("WID produced a nonfinite loss")
        return LossOutput(total, {"forget_loss": forget_loss, "model_loss": model_loss, "identity_loss": identity_loss, "model_term": model_term, "identity_term": identity_term, "preserve_loss": preserve_loss, "preserve_term": self.preservation_weight * preserve_loss})
