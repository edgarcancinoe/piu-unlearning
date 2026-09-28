from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from piu_unlearning.models.arcface import ArcFaceExtractor

if TYPE_CHECKING:
    from piu_unlearning.config import PIUConfig
    from piu_unlearning.data import EvaluationConditions, ExperimentSplit
    from piu_unlearning.models.arc2face import GeneratedSamples


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


def extract_face_embeddings(image_paths: list[Path], extractor: ArcFaceExtractor) -> torch.Tensor:
    from PIL import Image

    embeddings = []
    for image_path in image_paths:
        embedding = extractor(np.asarray(Image.open(image_path).convert("RGB")))
        if embedding is None:
            raise RuntimeError(f"No face detected in generated image: {image_path}")
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
    forget_accuracy = (forget_predictions == forget_labels).float().mean().item()
    retain_accuracy = (retain_predictions == retain_labels).float().mean().item()
    return SRKMetrics(forget_accuracy=forget_accuracy, retain_accuracy=retain_accuracy, score=retain_accuracy / (forget_accuracy + epsilon))


def evaluate_phase(samples: GeneratedSamples, conditions: EvaluationConditions, split: ExperimentSplit, extractor: ArcFaceExtractor) -> PhaseMetrics:
    forget_embeddings = extract_face_embeddings(samples.forget, extractor)
    retain_embeddings = extract_face_embeddings(samples.retain, extractor)
    forget = SplitMetrics(ism=compute_ism(forget_embeddings, conditions.forget_labels, split.centroids, split.centroid_labels))
    retain = SplitMetrics(ism=compute_ism(retain_embeddings, conditions.retain_labels, split.centroids, split.centroid_labels))
    srk = compute_srk(forget_embeddings, conditions.forget_labels, retain_embeddings, conditions.retain_labels, split.centroids, split.centroid_labels)
    return PhaseMetrics(forget=forget, retain=retain, srk=srk)


def evaluate_before_after(before: GeneratedSamples, after: GeneratedSamples, conditions: EvaluationConditions, split: ExperimentSplit, config: PIUConfig) -> EvaluationReport:
    """Evaluate aligned samples from the original and unlearned models."""
    extractor = ArcFaceExtractor(config.embeddings_path.parent / "face_models", config.device)
    return EvaluationReport(before=evaluate_phase(before, conditions, split, extractor), after=evaluate_phase(after, conditions, split, extractor))
