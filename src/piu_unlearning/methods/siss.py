from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from piu_unlearning.config import SISSConfig
from piu_unlearning.data import ExperimentSplit, PairedImageDataset, load_image_manifest
from piu_unlearning.training.losses import LossOutput


@dataclass(frozen=True)
class SISSInputs:
    forget: PairedImageDataset
    retain: PairedImageDataset
    metadata: dict


@dataclass(frozen=True)
class SISSContext:
    model: nn.Module
    vae: nn.Module
    scheduler: object
    conditioner: object
    device: torch.device


def prepare_siss_inputs(split: ExperimentSplit, config: SISSConfig) -> SISSInputs:
    required = torch.cat([split.forget_train.indices, split.retain_train.indices])
    paths, manifest = load_image_manifest(config, required)
    return SISSInputs(PairedImageDataset(split.forget_train, paths), PairedImageDataset(split.retain_train, paths), {"manifest": manifest, "forget_indices": split.forget_train.indices.tolist(), "retain_indices": split.retain_train.indices.tolist()})


def diffusion_loss(model, scheduler, clean, conditioning, noise, timesteps):
    noisy = scheduler.add_noise(clean, noise, timesteps)
    prediction = model(noisy, timesteps, encoder_hidden_states=conditioning).sample
    if scheduler.config.prediction_type == "epsilon": target = noise
    elif scheduler.config.prediction_type == "v_prediction": target = scheduler.get_velocity(clean, noise, timesteps)
    else: raise ValueError(f"Unsupported prediction type: {scheduler.config.prediction_type}")
    return F.mse_loss(prediction.float(), target.float())


def combine_gradients(retain, forget, beta):
    retain_norm = torch.stack([gradient.float().square().sum() for gradient in retain]).sum().sqrt()
    forget_norm = torch.stack([gradient.float().square().sum() for gradient in forget]).sum().sqrt()
    scale = beta * retain_norm / (forget_norm + 1e-8)
    if not torch.isfinite(retain_norm + forget_norm + scale): raise FloatingPointError("SISS produced nonfinite gradients")
    for retained, forgotten in zip(retain, forget, strict=True): retained.add_(forgotten * -scale)
    return retain, scale, retain_norm, forget_norm


class SISS:
    """PIU's SISS adaptation: g_retain - beta * ||g_retain|| / ||g_forget|| * g_forget."""

    def __init__(self, beta: float):
        self.beta = beta

    def compute_loss(self, forget_batch: dict, retain_batch: dict, context: SISSContext) -> LossOutput:
        model, vae, scheduler = context.model, context.vae, context.scheduler
        parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
        with torch.no_grad():
            retain_clean = vae.encode(retain_batch["pixel_values"].to(context.device)).latent_dist.sample() * vae.config.scaling_factor
            forget_clean = vae.encode(forget_batch["pixel_values"].to(context.device)).latent_dist.sample() * vae.config.scaling_factor
            retain_conditioning = context.conditioner.encode(retain_batch["face_embs"].to(context.device))
            forget_conditioning = context.conditioner.encode(forget_batch["face_embs"].to(context.device))
        if retain_clean.shape != forget_clean.shape: raise ValueError("SISS requires matched retain/forget latent shapes")
        noise = torch.randn_like(retain_clean)
        timesteps = torch.randint(scheduler.config.num_train_timesteps, (len(noise),), device=context.device)
        retain_loss = diffusion_loss(model, scheduler, retain_clean, retain_conditioning, noise, timesteps)
        retain_gradients = torch.autograd.grad(retain_loss, parameters)
        forget_loss = diffusion_loss(model, scheduler, forget_clean, forget_conditioning, noise, timesteps)
        forget_gradients = torch.autograd.grad(forget_loss, parameters)
        gradients, scale, retain_norm, forget_norm = combine_gradients(retain_gradients, forget_gradients, self.beta)
        retain_loss, forget_loss = retain_loss.detach(), forget_loss.detach()
        return LossOutput(retain_loss - scale * forget_loss, {"forget_loss": forget_loss, "retain_loss": retain_loss, "scaling_factor": scale, "retain_grad_norm": retain_norm, "forget_grad_norm": forget_norm}, gradients)
