from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from piu_unlearning.config import PIUConfig, parse_config
from piu_unlearning.data import ExperimentSplit, create_embedding_loaders, create_evaluation_conditions, create_experiment_split, load_prepared_data, save_evaluation_conditions, select_anchor_embedding, write_split_manifest
from piu_unlearning.evaluation import EvaluationReport, evaluate_before_after
from piu_unlearning.models.arc2face import Arc2FaceIdentityConditioner, GeneratedSamples, generate_evaluation_samples, load_arc2face
from piu_unlearning.training.losses import PIU
from piu_unlearning.training.runner import train_piu

if TYPE_CHECKING:
    from diffusers import StableDiffusionPipeline


@dataclass(frozen=True)
class PIUResult:
    identity_id: int
    checkpoint_path: Path
    before_images: GeneratedSamples
    after_images: GeneratedSamples
    evaluation: EvaluationReport


def unlearn_identity(model: StableDiffusionPipeline, split: ExperimentSplit, config: PIUConfig) -> Path:
    """Train PIU for one identity and save the resulting U-Net checkpoint."""
    from diffusers import DDPMScheduler

    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    forget_loader, retain_loader = create_embedding_loaders(split, config)
    anchor_embedding, anchor_id, anchor_similarity = select_anchor_embedding(split, config)
    print(f"Selected anchor identity: {anchor_id} (cosine similarity {anchor_similarity:.6f})", flush=True)
    anchor_embedding = anchor_embedding.unsqueeze(0).to(device)
    identity_conditioner = Arc2FaceIdentityConditioner(model.tokenizer, model.text_encoder)
    with torch.no_grad():
        anchor_conditioning = identity_conditioner.encode(anchor_embedding)

    frozen_unet = copy.deepcopy(model.unet).eval().requires_grad_(False)
    scheduler = DDPMScheduler.from_config(model.scheduler.config)
    piu = PIU(preservation_weight=config.preservation_weight, negative_guidance_scale=config.negative_guidance_scale)
    return train_piu(
        piu=piu,
        forget_loader=forget_loader,
        retain_loader=retain_loader,
        anchor_conditioning=anchor_conditioning,
        trainable_unet=model.unet,
        frozen_unet=frozen_unet,
        scheduler=scheduler,
        identity_conditioner=identity_conditioner,
        device=device,
        output_dir=config.output_dir / "checkpoints",
        training_steps=config.training_steps,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        log_every=config.log_every,
    )


def run_demo(config: PIUConfig) -> PIUResult:
    before_dir = config.output_dir / "before"
    after_dir = config.output_dir / "after"
    print(f"PIU demo for identity {config.identity_id}", flush=True)
    print("[1/6] Preparing canonical data splits and evaluation conditions", flush=True)
    embeddings, labels, centroids, centroid_labels = load_prepared_data(config)
    split = create_experiment_split(embeddings, labels, centroids, centroid_labels, config)
    conditions = create_evaluation_conditions(split, config)
    write_split_manifest(split, config.output_dir / "split.json")
    save_evaluation_conditions(conditions, config.output_dir / "evaluation_conditions.npz")
    print(f"Forget train dataset indices ({len(split.forget_train.indices)}): {split.forget_train.indices.tolist()}", flush=True)
    print(f"Forget validation dataset indices ({len(split.forget_validation.indices)}): {split.forget_validation.indices.tolist()}", flush=True)
    print(f"Retain split: {len(split.retain_train.indices)} training rows, {len(split.retain_validation.indices)} validation rows", flush=True)
    print(f"Evaluation forget source indices: {list(conditions.forget_source_indices)}", flush=True)
    print(f"Evaluation retain indices: {conditions.retain_indices.tolist()}", flush=True)
    print(f"Complete split manifest: {config.output_dir / 'split.json'}", flush=True)
    print(f"[2/6] Loading Arc2Face on {config.device}", flush=True)
    model = load_arc2face(config)
    print(f"[3/6] Generating baseline samples ({config.num_samples} forget, {config.num_samples} retain)", flush=True)
    before_images = generate_evaluation_samples(model, conditions, config, before_dir, "Baseline")
    print(f"[4/6] Training PIU ({config.training_steps} optimizer steps, micro-batch {config.batch_size}, accumulation {config.gradient_accumulation_steps}, effective batch {config.batch_size * config.gradient_accumulation_steps})", flush=True)
    checkpoint_path = unlearn_identity(model, split, config)
    print(f"[5/6] Generating post-unlearning samples ({config.num_samples} forget, {config.num_samples} retain)", flush=True)
    after_images = generate_evaluation_samples(model, conditions, config, after_dir, "Post-unlearning")
    print("[6/6] Extracting face embeddings and computing evaluation metrics", flush=True)
    evaluation = evaluate_before_after(before_images, after_images, conditions, split, config)
    result = PIUResult(identity_id=config.identity_id, checkpoint_path=checkpoint_path, before_images=before_images, after_images=after_images, evaluation=evaluation)
    write_summary(result, config.output_dir / "summary.json")
    print_comparison(result)
    print(f"Finished. Results written to {config.output_dir}", flush=True)
    return result


def write_summary(result: PIUResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result), indent=2, default=str) + "\n", encoding="utf-8")


def print_comparison(result: PIUResult) -> None:
    print(f"Identity {result.identity_id}")
    print(f"{'Metric':<12} {'Before':>12} {'After':>12} {'Change':>12}")
    metrics = (
        ("forget ISM", result.evaluation.before.forget.ism, result.evaluation.after.forget.ism),
        ("retain ISM", result.evaluation.before.retain.ism, result.evaluation.after.retain.ism),
        ("AccU", result.evaluation.before.srk.forget_accuracy, result.evaluation.after.srk.forget_accuracy),
        ("AccR", result.evaluation.before.srk.retain_accuracy, result.evaluation.after.srk.retain_accuracy),
        ("SRK", result.evaluation.before.srk.score, result.evaluation.after.srk.score),
    )
    for name, before, after in metrics:
        print(f"{name:<12} {before:>12.4f} {after:>12.4f} {after - before:>+12.4f}")


def main() -> None:
    run_demo(parse_config())


if __name__ == "__main__":
    main()
