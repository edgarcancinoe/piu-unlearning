from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator

import torch
from tqdm.auto import tqdm

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
    gradient_accumulation_steps: int,
    evaluate_ism: Callable[[int], tuple[float, float]] | None,
    evaluation_every: int,
    learning_rate: float,
    weight_decay: float,
    log_every: int,
) -> Path:
    trainable_unet.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW((parameter for parameter in trainable_unet.parameters() if parameter.requires_grad), lr=learning_rate, weight_decay=weight_decay)
    ism_history_path = output_dir / "ism_history.jsonl"
    forget_batches = batches_forever(forget_loader)
    retain_batches = batches_forever(retain_loader)
    with tqdm(total=training_steps * gradient_accumulation_steps, desc="Training PIU", unit="microbatch", dynamic_ncols=True) as progress:
        for step in range(1, training_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            total_loss = total_forget_loss = total_preserve_loss = 0.0
            for micro_step in range(1, gradient_accumulation_steps + 1):
                loss, forget_loss, preserve_loss = piu.compute_piu_loss(next(forget_batches), next(retain_batches), anchor_conditioning, trainable_unet, frozen_unet, scheduler, identity_conditioner, device)
                (loss / gradient_accumulation_steps).backward()
                total_loss += loss.item() / gradient_accumulation_steps
                total_forget_loss += forget_loss.item() / gradient_accumulation_steps
                total_preserve_loss += preserve_loss.item() / gradient_accumulation_steps
                progress.set_postfix(step=f"{step}/{training_steps}", micro=f"{micro_step}/{gradient_accumulation_steps}", loss=f"{loss.item():.4f}", forget=f"{forget_loss.item():.4f}", preserve=f"{preserve_loss.item():.4f}")
                progress.update()
            optimizer.step()

            if evaluate_ism is not None and evaluation_every and step % evaluation_every == 0:
                progress.set_description("Evaluating ISM")
                forget_ism, retain_ism = evaluate_ism(step)
                with ism_history_path.open("a", encoding="utf-8") as history: history.write(json.dumps({"step": step, "forget_ism": forget_ism, "retain_ism": retain_ism}) + "\n")
                progress.write(f"step={step:04d}/{training_steps:04d} forget_ism={forget_ism:.6f} retain_ism={retain_ism:.6f}")
                trainable_unet.train()
                progress.set_description("Training PIU")

            if step == 1 or step % log_every == 0 or step == training_steps:
                progress.write(f"step={step:04d}/{training_steps:04d} loss={total_loss:.6f} forget={total_forget_loss:.6f} preserve={total_preserve_loss:.6f}")

    checkpoint_path = output_dir / "piu_unet.pt"
    torch.save({"unet_state_dict": trainable_unet.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "training_steps": training_steps, "gradient_accumulation_steps": gradient_accumulation_steps, "preservation_weight": piu.preservation_weight, "negative_guidance_scale": piu.negative_guidance_scale}, checkpoint_path)
    return checkpoint_path
