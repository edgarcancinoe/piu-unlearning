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
    from piu_unlearning.config import TrainingConfig


def batches_forever(loader: DataLoader) -> Iterator:
    if not len(loader): raise ValueError("Training loader is empty")
    while True:
        yield from loader


def train_model(model: UNet2DConditionModel, compute_loss: Callable, forget_loader: DataLoader, retain_loader: DataLoader | None, config: TrainingConfig, evaluate_ism: Callable[[int], tuple[float, float]] | None = None) -> Path:
    """Optimize a supplied objective; methods own their conditioning and loss computation."""
    model.train()
    output_dir = config.output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer_type = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[config.optimizer]
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    optimizer = optimizer_type(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    ema = None
    if config.method == "siss" and config.use_ema:
        from diffusers.training_utils import EMAModel

        ema = EMAModel(parameters)
    loss_history_path = output_dir / "loss_history.jsonl"
    ism_history_path = output_dir / "ism_history.jsonl"
    loss_history_path.write_text("", encoding="utf-8")
    ism_history_path.write_text("", encoding="utf-8")
    forget_batches = batches_forever(forget_loader)
    retain_batches = batches_forever(retain_loader) if retain_loader is not None else None
    accumulation_steps = config.gradient_accumulation_steps
    if config.method == "siss": accumulation_steps = config.batch_size * accumulation_steps // config.gradient_batch_size
    unit = "gradient batch" if config.method == "siss" else "microbatch"
    title = f"Training {config.method.upper()}"
    with tqdm(total=config.training_steps * accumulation_steps, desc=title, unit=unit, dynamic_ncols=True) as progress:
        for step in range(1, config.training_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            totals = defaultdict(float)
            for micro_step in range(1, accumulation_steps + 1):
                result: LossOutput = compute_loss(next(forget_batches), next(retain_batches) if retain_batches is not None else None)
                if result.gradients is None:
                    (result.loss / accumulation_steps).backward()
                else:
                    for parameter, gradient in zip(parameters, result.gradients, strict=True):
                        gradient.div_(accumulation_steps)
                        if parameter.grad is None: parameter.grad = gradient
                        else: parameter.grad.add_(gradient)
                metrics = {"loss": result.loss, **result.metrics}
                for name, value in metrics.items(): totals[name] += value.detach().item() / accumulation_steps
                progress.set_postfix(step=f"{step}/{config.training_steps}", batch=f"{micro_step}/{accumulation_steps}", **{name: f"{value.item():.4f}" for name, value in metrics.items()})
                progress.update()
            if config.max_grad_norm is not None: torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            if ema is not None: ema.step(parameters)
            with loss_history_path.open("a", encoding="utf-8") as history: history.write(json.dumps({"step": step, **totals}) + "\n")

            if evaluate_ism is not None and config.evaluation_every and step % config.evaluation_every == 0:
                progress.set_description("Evaluating ISM")
                if ema is not None:
                    ema.store(parameters)
                    ema.copy_to(parameters)
                try:
                    forget_ism, retain_ism = evaluate_ism(step)
                finally:
                    if ema is not None: ema.restore(parameters)
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
        "training_steps": config.training_steps,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "trainable_parameters": [name for name, parameter in model.named_parameters() if parameter.requires_grad],
    }
    if config.method != "siss": checkpoint["preservation_weight"] = config.preservation_weight
    if config.method == "piu": checkpoint["negative_guidance_scale"] = config.negative_guidance_scale
    if config.method == "siss": checkpoint["beta"] = config.beta
    if ema is not None: checkpoint["ema_state_dict"] = ema.state_dict()
    torch.save(checkpoint, checkpoint_path)
    if ema is not None: ema.copy_to(parameters)
    return checkpoint_path
