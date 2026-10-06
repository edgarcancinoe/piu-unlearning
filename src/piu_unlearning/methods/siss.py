from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from piu_unlearning.config import SISSConfig
from piu_unlearning.data import EmbeddingPartition, ExperimentSplit, PairedImageDataset, load_image_manifest
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
    paths, manifest = load_image_manifest(config)
    retain = split.retain_train
    retain_ids = torch.unique(retain.labels).tolist()
    if config.num_preserve_ids and len(retain_ids) > config.num_preserve_ids:
        random.Random(42).shuffle(retain_ids)
        selected = torch.isin(retain.labels, torch.tensor(retain_ids[:config.num_preserve_ids], dtype=retain.labels.dtype))
        retain = EmbeddingPartition(retain.embeddings[selected], retain.labels[selected], retain.indices[selected])
    metadata = {"manifest": manifest, "forget_indices": split.forget_train.indices.tolist(), "retain_indices": retain.indices.tolist(), "retain_identity_ids": torch.unique(retain.labels).tolist(), "sampler": "epoch_shuffle"}
    return SISSInputs(PairedImageDataset(split.forget_train, paths), PairedImageDataset(retain, paths), metadata)


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

    def __init__(self, beta: float, batch_size: int):
        self.beta = beta
        self.batch_size = batch_size

    @torch.no_grad()
    def encode_batch(self, batch: dict, context: SISSContext):
        latents, conditioning = [], []
        for start in range(0, len(batch["pixel_values"]), self.batch_size):
            part = slice(start, start + self.batch_size)
            latents.append(context.vae.encode(batch["pixel_values"][part].to(context.device)).latent_dist.sample() * context.vae.config.scaling_factor)
            conditioning.append(context.conditioner.encode(batch["face_embs"][part].to(context.device)))
        return torch.cat(latents), torch.cat(conditioning)

    def branch_gradients(self, clean, conditioning, noise, timesteps, parameters, context):
        gradients, total_loss = None, clean.new_zeros(())
        for start in range(0, len(clean), self.batch_size):
            part = slice(start, start + self.batch_size)
            loss = diffusion_loss(context.model, context.scheduler, clean[part], conditioning[part], noise[part], timesteps[part]) * (len(clean[part]) / len(clean))
            current = torch.autograd.grad(loss, parameters)
            if gradients is None: gradients = current
            else:
                for accumulated, gradient in zip(gradients, current, strict=True): accumulated.add_(gradient)
            total_loss += loss.detach()
            del current, loss
        return total_loss, gradients

    def compute_loss(self, forget_batch: dict, retain_batch: dict, context: SISSContext) -> LossOutput:
        """Normalize once per logical batch, after accumulating raw microbatch gradients."""
        parameters = tuple(parameter for parameter in context.model.parameters() if parameter.requires_grad)
        retain_clean, retain_conditioning = self.encode_batch(retain_batch, context)
        forget_clean, forget_conditioning = self.encode_batch(forget_batch, context)
        if retain_clean.shape != forget_clean.shape: raise ValueError("SISS requires matched retain/forget latent shapes")
        noise = torch.randn_like(retain_clean)
        timesteps = torch.randint(context.scheduler.config.num_train_timesteps, (len(noise),), device=context.device)
        retain_loss, retain_gradients = self.branch_gradients(retain_clean, retain_conditioning, noise, timesteps, parameters, context)
        forget_loss, forget_gradients = self.branch_gradients(forget_clean, forget_conditioning, noise, timesteps, parameters, context)
        gradients, scale, retain_norm, forget_norm = combine_gradients(retain_gradients, forget_gradients, self.beta)
        return LossOutput(retain_loss - scale * forget_loss, {"forget_loss": forget_loss, "retain_loss": retain_loss, "scaling_factor": scale, "retain_grad_norm": retain_norm, "forget_grad_norm": forget_norm}, gradients)
