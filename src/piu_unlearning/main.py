from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from piu_unlearning.config import RunConfig, SISSConfig, UCEConfig, WIDConfig, parse_config
from piu_unlearning.data import EvaluationConditions, ExperimentSplit, create_embedding_loaders, create_evaluation_conditions, create_experiment_split, create_training_loaders, load_prepared_data, save_evaluation_conditions, write_split_manifest
from piu_unlearning.dataset import prepare_for_demo
from piu_unlearning.evaluation import EvaluationReport, evaluate_before_after, evaluate_training_ism
from piu_unlearning.models.arcface import ArcFaceExtractor
from piu_unlearning.models.arc2face import Arc2FaceIdentityConditioner, GeneratedSamples, generate_evaluation_samples, load_arc2face, load_generated_samples
from piu_unlearning.methods import build_method
from piu_unlearning.methods.siss import SISSContext, SISSInputs, prepare_siss_inputs
from piu_unlearning.methods.wid import WIDContext, WIDInputs, prepare_wid_inputs
from piu_unlearning.training.losses import NoisePredictionContext
from piu_unlearning.training.runner import train_model
from piu_unlearning.visualization import create_summary_visuals

if TYPE_CHECKING:
    from diffusers import StableDiffusionPipeline


@dataclass(frozen=True)
class UnlearningResult:
    identity_id: int
    checkpoint_path: Path
    before_images: GeneratedSamples
    after_images: GeneratedSamples
    evaluation: EvaluationReport
    method: str = "piu"


def unlearn_identity(model: StableDiffusionPipeline, split: ExperimentSplit, conditions: EvaluationConditions, config: RunConfig, wid_inputs: WIDInputs | None = None, siss_inputs: SISSInputs | None = None) -> Path:
    """Dispatch to image-based SISS, reference-guided training, or closed-form editing."""
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    method = build_method(config)
    if isinstance(config, SISSConfig):
        from diffusers import DDPMScheduler

        inputs = siss_inputs or prepare_siss_inputs(split, config)
        forget_loader, retain_loader = create_training_loaders(inputs.forget, inputs.retain, config)
        config.output_dir.mkdir(parents=True, exist_ok=True)
        (config.output_dir / "siss_data.json").write_text(json.dumps(inputs.metadata, indent=2) + "\n", encoding="utf-8")
        context = SISSContext(model.unet, model.vae.eval().requires_grad_(False), DDPMScheduler.from_config(model.scheduler.config), Arc2FaceIdentityConditioner(model.tokenizer, model.text_encoder), device)
        extractor = ArcFaceExtractor(config.embeddings_path.parent / "face_models", config.evaluation_device) if config.evaluation_every else None

        def evaluate(step: int) -> tuple[float, float]: return evaluate_training_ism(model, conditions, split, config, extractor, config.output_dir / "training_evaluation", step)

        return train_model(model.unet, partial(method.compute_loss, context=context), forget_loader, retain_loader, config, evaluate if extractor is not None else None)
    if isinstance(config, WIDConfig) and wid_inputs is None: wid_inputs = prepare_wid_inputs(split, config)
    reference = wid_inputs.reference if wid_inputs is not None else method.prepare_reference(split, config)
    reference_metadata = {"mode": reference.mode, "identity_id": reference.identity_id, "similarity": reference.similarity}
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "reference.json").write_text(json.dumps(reference_metadata, indent=2) + "\n", encoding="utf-8")
    torch.save(reference.embedding.detach().cpu(), config.output_dir / "reference_embedding.pt")
    print(f"Reference identity: {reference_metadata}", flush=True)
    identity_conditioner = Arc2FaceIdentityConditioner(model.tokenizer, model.text_encoder)
    if isinstance(config, UCEConfig): return method.edit(model.unet, split, identity_conditioner, reference, config)

    from diffusers import DDPMScheduler

    if wid_inputs is not None: forget_loader, retain_loader = create_training_loaders(wid_inputs.dataset, split.retain_train.embeddings, config)
    else: forget_loader, retain_loader = create_embedding_loaders(split, config)
    with torch.no_grad():
        reference_conditioning = identity_conditioner.encode(reference.embedding.to(device))

    frozen_unet = copy.deepcopy(model.unet).eval().requires_grad_(False)
    scheduler = DDPMScheduler.from_config(model.scheduler.config)
    context = NoisePredictionContext(model.unet, frozen_unet, scheduler, identity_conditioner, reference_conditioning, device)
    if wid_inputs is not None:
        encoder = wid_inputs.encoder.to(device) if wid_inputs.encoder is not None else None
        target = wid_inputs.identity_target.to(device) if wid_inputs.identity_target is not None else None
        context = WIDContext(context, model.vae.eval().requires_grad_(False), encoder, target)
        (config.output_dir / "wid_identity.json").write_text(json.dumps(wid_inputs.metadata, indent=2) + "\n", encoding="utf-8")
        if target is not None: torch.save(target.detach().cpu(), config.output_dir / "identity_target.pt")
        print(f"WID identity target: {wid_inputs.metadata['target_mode']}; loss: mean MSE", flush=True)
    extractor = ArcFaceExtractor(config.embeddings_path.parent / "face_models", config.evaluation_device) if config.evaluation_every else None

    def evaluate(step: int) -> tuple[float, float]: return evaluate_training_ism(model, conditions, split, config, extractor, config.output_dir / "training_evaluation", step)

    return train_model(model.unet, partial(method.compute_loss, context=context), forget_loader, retain_loader, config, evaluate if extractor is not None else None)


def run_demo(config: RunConfig, split: ExperimentSplit | None = None) -> UnlearningResult:
    before_dir = config.output_dir / "before"
    after_dir = config.output_dir / "after"
    write_config(config, config.output_dir / "config.json")
    print(f"{config.method.upper()} demo for identity {config.identity_id}", flush=True)
    print(f"Resolved configuration: {config.output_dir / 'config.json'}", flush=True)
    print("[1/6] Preparing canonical data splits and evaluation conditions", flush=True)
    if split is None:
        embeddings, labels, centroids, centroid_labels = load_prepared_data(config)
        split = create_experiment_split(embeddings, labels, centroids, centroid_labels, config)
    conditions = create_evaluation_conditions(split, config)
    wid_inputs = prepare_wid_inputs(split, config) if isinstance(config, WIDConfig) else None
    siss_inputs = prepare_siss_inputs(split, config) if isinstance(config, SISSConfig) else None
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
    if config.reuse_baseline:
        print(f"[3/6] Reusing baseline samples from {before_dir}", flush=True)
        before_images = load_generated_samples(before_dir, config.num_samples)
    else:
        print(f"[3/6] Generating baseline samples ({config.num_samples} forget, {config.num_samples} retain)", flush=True)
        before_images = generate_evaluation_samples(model, conditions, config, before_dir, "Baseline")
    if isinstance(config, UCEConfig):
        print(f"[4/6] Applying UCE closed-form edit ({config.edit_scope} cross-attention key/value projections)", flush=True)
    else:
        print(f"[4/6] Training {config.method.upper()} ({config.training_steps} optimizer steps, micro-batch {config.batch_size}, accumulation {config.gradient_accumulation_steps}, effective batch {config.batch_size * config.gradient_accumulation_steps}, ISM every {config.evaluation_every or 'disabled'} steps)", flush=True)
    if isinstance(config, SISSConfig): print(f"      SISS normalization batch {config.gradient_batch_size}, {config.batch_size * config.gradient_accumulation_steps // config.gradient_batch_size} gradient groups per update, gradient checkpointing {config.gradient_checkpointing}", flush=True)
    if siss_inputs is not None: checkpoint_path = unlearn_identity(model, split, conditions, config, siss_inputs=siss_inputs)
    elif wid_inputs is None: checkpoint_path = unlearn_identity(model, split, conditions, config)
    else: checkpoint_path = unlearn_identity(model, split, conditions, config, wid_inputs)
    del wid_inputs, siss_inputs
    print(f"[5/6] Generating post-unlearning samples ({config.num_samples} forget, {config.num_samples} retain)", flush=True)
    after_images = generate_evaluation_samples(model, conditions, config, after_dir, "Post-unlearning")
    print("[6/6] Extracting face embeddings and computing evaluation metrics", flush=True)
    evaluation = evaluate_before_after(before_images, after_images, conditions, split, config)
    result = UnlearningResult(identity_id=config.identity_id, checkpoint_path=checkpoint_path, before_images=before_images, after_images=after_images, evaluation=evaluation, method=config.method)
    curves_path, grid_path = create_summary_visuals(before_images, after_images, conditions, config.output_dir, config.method)
    write_summary(result, config.output_dir / "summary.json")
    print_comparison(result)
    print(f"Summary visuals: {', '.join(str(path) for path in (curves_path, grid_path) if path is not None)}", flush=True)
    print(f"Finished. Results written to {config.output_dir}", flush=True)
    return result


def write_summary(result: UnlearningResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result), indent=2, default=str) + "\n", encoding="utf-8")


def write_config(config: RunConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(config), indent=2, default=str) + "\n", encoding="utf-8")


def print_comparison(result: UnlearningResult) -> None:
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
    config = parse_config()
    prepare_for_demo(config)
    run_demo(config)


if __name__ == "__main__":
    main()
