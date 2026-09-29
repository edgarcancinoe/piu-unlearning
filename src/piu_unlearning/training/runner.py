from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator

import torch
from tqdm.auto import tqdm

from piu_unlearning.training.losses import LossOutput

if TYPE_CHECKING:
    from diffusers import UNet2DConditionModel
    from torch.utils.data import DataLoader
    from piu_unlearning.config import ESDConfig, PIUConfig


def batches_forever(loader: DataLoader) -> Iterator:
    if not len(loader): raise ValueError("Training loader is empty")
    while True:
        yield from loader


def train_model(model: UNet2DConditionModel, compute_loss: Callable, forget_loader: DataLoader, retain_loader: DataLoader | None, config: PIUConfig | ESDConfig, evaluate_ism: Callable[[int], tuple[float, float]] | None = None) -> Path:
    """Optimize a supplied objective; methods own their conditioning and loss computation."""
    model.train()
    output_dir = config.output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer_type = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[config.optimizer]
    optimizer = optimizer_type((parameter for parameter in model.parameters() if parameter.requires_grad), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_history_path = output_dir / "loss_history.jsonl"
    ism_history_path = output_dir / "ism_history.jsonl"
    loss_history_path.write_text("", encoding="utf-8")
    ism_history_path.write_text("", encoding="utf-8")
    forget_batches = batches_forever(forget_loader)
    retain_batches = batches_forever(retain_loader) if retain_loader is not None else None
    title = f"Training {config.method.upper()}"
    with tqdm(total=config.training_steps * config.gradient_accumulation_steps, desc=title, unit="microbatch", dynamic_ncols=True) as progress:
        for step in range(1, config.training_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            totals = defaultdict(float)
            for micro_step in range(1, config.gradient_accumulation_steps + 1):
                result: LossOutput = compute_loss(next(forget_batches), next(retain_batches) if retain_batches is not None else None)
                (result.loss / config.gradient_accumulation_steps).backward()
                metrics = {"loss": result.loss, **result.metrics}
                for name, value in metrics.items(): totals[name] += value.detach().item() / config.gradient_accumulation_steps
                progress.set_postfix(step=f"{step}/{config.training_steps}", micro=f"{micro_step}/{config.gradient_accumulation_steps}", **{name: f"{value.item():.4f}" for name, value in metrics.items()})
                progress.update()
            optimizer.step()
            with loss_history_path.open("a", encoding="utf-8") as history: history.write(json.dumps({"step": step, **totals}) + "\n")

            if evaluate_ism is not None and config.evaluation_every and step % config.evaluation_every == 0:
                progress.set_description("Evaluating ISM")
                forget_ism, retain_ism = evaluate_ism(step)
                with ism_history_path.open("a", encoding="utf-8") as history: history.write(json.dumps({"step": step, "forget_ism": forget_ism, "retain_ism": retain_ism}) + "\n")
                progress.write(f"step={step:04d}/{config.training_steps:04d} forget_ism={forget_ism:.6f} retain_ism={retain_ism:.6f}")
                model.train()
                progress.set_description(title)

            if step == 1 or step % config.log_every == 0 or step == config.training_steps:
                values = " ".join(f"{name}={value:.6f}" for name, value in totals.items())
                progress.write(f"step={step:04d}/{config.training_steps:04d} {values}")

    checkpoint_path = output_dir / f"{config.method}_unet.pt"
    checkpoint = {
        "method": config.method,
        "config": json.loads(json.dumps(asdict(config), default=str)),
        "unet_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_steps": config.training_steps,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "preservation_weight": config.preservation_weight,
        "negative_guidance_scale": config.negative_guidance_scale,
        "trainable_parameters": [name for name, parameter in model.named_parameters() if parameter.requires_grad],
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path
