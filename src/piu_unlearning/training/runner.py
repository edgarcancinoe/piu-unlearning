from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import torch

from piu_unlearning.training.losses import PIU

if TYPE_CHECKING:
    from diffusers import DDPMScheduler, UNet2DConditionModel
    from torch.utils.data import DataLoader

    from piu_unlearning.models.arc2face import Arc2FaceIdentityConditioner


def batches_forever(loader: DataLoader[torch.Tensor]) -> Iterator[torch.Tensor]:
    while True:
        yield from loader


def train_piu(
    piu: PIU,
    forget_loader: DataLoader[torch.Tensor],
    retain_loader: DataLoader[torch.Tensor],
    anchor_conditioning: torch.Tensor,
    trainable_unet: UNet2DConditionModel,
    frozen_unet: UNet2DConditionModel,
    scheduler: DDPMScheduler,
    identity_conditioner: Arc2FaceIdentityConditioner,
    device: torch.device,
    output_dir: Path,
    training_steps: int,
    learning_rate: float,
    weight_decay: float,
    log_every: int,
) -> Path:
    trainable_unet.train()
    optimizer = torch.optim.AdamW((parameter for parameter in trainable_unet.parameters() if parameter.requires_grad), lr=learning_rate, weight_decay=weight_decay)
    forget_batches = batches_forever(forget_loader)
    retain_batches = batches_forever(retain_loader)
    for step in range(1, training_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, forget_loss, preserve_loss = piu.compute_piu_loss(next(forget_batches), next(retain_batches), anchor_conditioning, trainable_unet, frozen_unet, scheduler, identity_conditioner, device)
        loss.backward()
        optimizer.step()

        if step == 1 or step % log_every == 0 or step == training_steps:
            print(f"step={step:04d} loss={loss.item():.6f} forget={forget_loss.item():.6f} preserve={preserve_loss.item():.6f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "piu_unet.pt"
    torch.save({"unet_state_dict": trainable_unet.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "training_steps": training_steps, "preservation_weight": piu.preservation_weight, "negative_guidance_scale": piu.negative_guidance_scale}, checkpoint_path)
    return checkpoint_path
