from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from piu_unlearning.models.arcface import ArcFaceExtractor
from piu_unlearning.models.arc2face import GeneratedSamples, generate_evaluation_samples

if TYPE_CHECKING:
    from piu_unlearning.config import RunConfig
    from piu_unlearning.data import EvaluationConditions, ExperimentSplit
    from diffusers import StableDiffusionPipeline


@dataclass(frozen=True)
class SplitMetrics:
    ism: float


@dataclass(frozen=True)
class SRKMetrics:
    forget_accuracy: float
    retain_accuracy: float
    score: float


@dataclass(frozen=True)
class PhaseMetrics:
    forget: SplitMetrics
    retain: SplitMetrics
    srk: SRKMetrics


@dataclass(frozen=True)
class EvaluationReport:
    before: PhaseMetrics
    after: PhaseMetrics


def extract_face_embeddings(image_paths: list[Path], extractor: ArcFaceExtractor, description: str, embedding_dim: int = 512) -> torch.Tensor:
    from PIL import Image

    embeddings = []
    for image_path in tqdm(image_paths, desc=description, unit="image"):
        embedding = extractor(np.asarray(Image.open(image_path).convert("RGB")))
        if embedding is None:
            logging.getLogger(__name__).warning("No face detected in generated image: %s; scoring as zero ISM and unrecognized identity", image_path)
            embedding = np.zeros(embedding_dim, dtype=np.float32)
        embeddings.append(torch.from_numpy(embedding))
    return torch.stack(embeddings).float()


def compute_ism(generated_embeddings: torch.Tensor, generated_labels: torch.Tensor, centroids: torch.Tensor, centroid_labels: torch.Tensor) -> float:
    centroid_rows = {int(label): index for index, label in enumerate(centroid_labels.tolist())}
    expected_centroids = centroids[torch.tensor([centroid_rows[int(label)] for label in generated_labels.tolist()])]
    return F.cosine_similarity(generated_embeddings, expected_centroids, dim=1).mean().item()


def compute_srk(forget_embeddings: torch.Tensor, forget_labels: torch.Tensor, retain_embeddings: torch.Tensor, retain_labels: torch.Tensor, centroids: torch.Tensor, centroid_labels: torch.Tensor, epsilon: float = 1e-2) -> SRKMetrics:
    normalized_centroids = F.normalize(centroids, dim=1)
    forget_predictions = centroid_labels[(F.normalize(forget_embeddings, dim=1) @ normalized_centroids.T).argmax(dim=1)]
    retain_predictions = centroid_labels[(F.normalize(retain_embeddings, dim=1) @ normalized_centroids.T).argmax(dim=1)]
    forget_accuracy = ((forget_predictions == forget_labels) & forget_embeddings.any(dim=1)).float().mean().item()
    retain_accuracy = ((retain_predictions == retain_labels) & retain_embeddings.any(dim=1)).float().mean().item()
    return SRKMetrics(forget_accuracy=forget_accuracy, retain_accuracy=retain_accuracy, score=retain_accuracy / (forget_accuracy + epsilon))


def evaluate_phase(samples: GeneratedSamples, conditions: EvaluationConditions, split: ExperimentSplit, extractor: ArcFaceExtractor, phase: str) -> PhaseMetrics:
    forget_embeddings = extract_face_embeddings(samples.forget, extractor, f"{phase}: forget embeddings", split.centroids.shape[1])
    retain_embeddings = extract_face_embeddings(samples.retain, extractor, f"{phase}: retain embeddings", split.centroids.shape[1])
    forget = SplitMetrics(ism=compute_ism(forget_embeddings, conditions.forget_labels, split.centroids, split.centroid_labels))
    retain = SplitMetrics(ism=compute_ism(retain_embeddings, conditions.retain_labels, split.centroids, split.centroid_labels))
    srk = compute_srk(forget_embeddings, conditions.forget_labels, retain_embeddings, conditions.retain_labels, split.centroids, split.centroid_labels)
    return PhaseMetrics(forget=forget, retain=retain, srk=srk)


def evaluate_training_ism(model: StableDiffusionPipeline, conditions: EvaluationConditions, split: ExperimentSplit, config: RunConfig, extractor: ArcFaceExtractor, output_dir: Path, step: int) -> tuple[float, float]:
    """Generate fixed validation conditions and return forget and retain ISM at one training step."""
    samples = generate_evaluation_samples(model, conditions, config, output_dir / f"step_{step:04d}", f"ISM step {step}")
    metrics = evaluate_phase(samples, conditions, split, extractor, f"ISM step {step}")
    return metrics.forget.ism, metrics.retain.ism


def evaluate_before_after(before: GeneratedSamples, after: GeneratedSamples, conditions: EvaluationConditions, split: ExperimentSplit, config: RunConfig) -> EvaluationReport:
    """Evaluate aligned samples from the original and unlearned models."""
    extractor = ArcFaceExtractor(config.embeddings_path.parent / "face_models", config.evaluation_device)
    return EvaluationReport(before=evaluate_phase(before, conditions, split, extractor, "Baseline evaluation"), after=evaluate_phase(after, conditions, split, extractor, "Post-unlearning evaluation"))
