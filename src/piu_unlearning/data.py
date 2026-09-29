from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler

if TYPE_CHECKING:
    from piu_unlearning.config import ESDConfig, PIUConfig, RunConfig


@dataclass(frozen=True)
class EmbeddingPartition:
    embeddings: torch.Tensor
    labels: torch.Tensor
    indices: torch.Tensor


@dataclass(frozen=True)
class ExperimentSplit:
    identity_id: int
    forget_train: EmbeddingPartition
    forget_validation: EmbeddingPartition
    retain_train: EmbeddingPartition
    retain_validation: EmbeddingPartition
    centroids: torch.Tensor
    centroid_labels: torch.Tensor

    @property
    def forget_centroid(self) -> torch.Tensor:
        return self.centroids[self.centroid_labels == self.identity_id]


@dataclass(frozen=True)
class EvaluationConditions:
    forget_embeddings: torch.Tensor
    forget_labels: torch.Tensor
    retain_embeddings: torch.Tensor
    retain_labels: torch.Tensor
    forget_source_indices: tuple[tuple[int, ...], ...]
    retain_indices: torch.Tensor
    forget_seeds: tuple[int, ...]
    retain_seeds: tuple[int, ...]


class DirichletEmbeddingDataset(Dataset[torch.Tensor]):
    """Sample normalized mixtures of up to three identity embeddings."""

    def __init__(self, embeddings: torch.Tensor) -> None:
        self.embeddings = embeddings

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def __getitem__(self, _index: int) -> torch.Tensor:
        num_sources = min(3, len(self))
        source_indices = torch.randperm(len(self))[:num_sources]
        weights = torch.distributions.Dirichlet(torch.ones(num_sources)).sample()
        return F.normalize((self.embeddings[source_indices] * weights.unsqueeze(1)).sum(dim=0), dim=0)


def load_prepared_data(config: RunConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    embeddings = torch.from_numpy(np.load(config.embeddings_path)).float()
    labels = torch.from_numpy(np.load(config.labels_path)).long()
    centroids = torch.from_numpy(np.load(config.centroids_path)).float()
    centroid_labels = torch.from_numpy(np.load(config.centroid_labels_path)).long()
    return embeddings, labels, centroids, centroid_labels


def create_experiment_split(embeddings: torch.Tensor, labels: torch.Tensor, centroids: torch.Tensor, centroid_labels: torch.Tensor, config: RunConfig) -> ExperimentSplit:
    """Create disjoint training and evaluation partitions for one identity."""
    generator = torch.Generator().manual_seed(config.seed)
    forget_indices = torch.where(labels == config.identity_id)[0]
    forget_indices = forget_indices[torch.randperm(forget_indices.numel(), generator=generator)]
    num_forget_validation = max(1, int(forget_indices.numel() * config.forget_validation_ratio))
    forget_train_indices = forget_indices[:-num_forget_validation]
    forget_validation_indices = forget_indices[-num_forget_validation:]
    if not forget_train_indices.numel(): raise ValueError(f"Identity {config.identity_id} needs at least two samples for a train/validation split")

    retain_identity_ids = torch.unique(labels[labels != config.identity_id])
    forced_train_ids = torch.empty(0, dtype=labels.dtype)
    if config.anchor_overrides is not None:
        forced_train_ids = torch.tensor([config.anchor_overrides[config.identity_id][config.proximity_threshold]], dtype=labels.dtype)
    holdout_candidates = retain_identity_ids[~torch.isin(retain_identity_ids, forced_train_ids)]
    holdout_candidates = holdout_candidates[torch.randperm(holdout_candidates.numel(), generator=generator)]
    num_retain_validation = max(1, int(retain_identity_ids.numel() * config.retain_validation_ratio))
    retain_validation_ids = holdout_candidates[:num_retain_validation]
    retain_mask = labels != config.identity_id
    retain_validation_indices = torch.where(retain_mask & torch.isin(labels, retain_validation_ids))[0]
    retain_train_indices = torch.where(retain_mask & ~torch.isin(labels, retain_validation_ids))[0]

    def partition(indices: torch.Tensor) -> EmbeddingPartition: return EmbeddingPartition(embeddings[indices], labels[indices], indices)
    return ExperimentSplit(
        identity_id=config.identity_id,
        forget_train=partition(forget_train_indices),
        forget_validation=partition(forget_validation_indices),
        retain_train=partition(retain_train_indices),
        retain_validation=partition(retain_validation_indices),
        centroids=centroids,
        centroid_labels=centroid_labels,
    )


def create_evaluation_conditions(split: ExperimentSplit, config: RunConfig) -> EvaluationConditions:
    """Create conditions and seeds reused before and after unlearning."""
    rng = np.random.default_rng(config.seed)
    forget_conditions = []
    forget_source_indices = []
    for _ in range(config.num_samples):
        num_sources = min(3, len(split.forget_validation.embeddings))
        source_indices = torch.from_numpy(rng.choice(len(split.forget_validation.embeddings), num_sources, replace=False))
        weights = torch.from_numpy(rng.dirichlet(np.ones(num_sources))).to(split.forget_validation.embeddings.dtype)
        forget_conditions.append(F.normalize((split.forget_validation.embeddings[source_indices] * weights.unsqueeze(1)).sum(dim=0), dim=0))
        forget_source_indices.append(tuple(split.forget_validation.indices[source_indices].tolist()))
    retain_indices = torch.from_numpy(rng.choice(len(split.retain_validation.embeddings), config.num_samples, replace=False))
    return EvaluationConditions(
        forget_embeddings=torch.stack(forget_conditions),
        forget_labels=torch.full((config.num_samples,), split.identity_id, dtype=split.forget_validation.labels.dtype),
        retain_embeddings=split.retain_validation.embeddings[retain_indices],
        retain_labels=split.retain_validation.labels[retain_indices],
        forget_source_indices=tuple(forget_source_indices),
        retain_indices=split.retain_validation.indices[retain_indices],
        forget_seeds=tuple(config.seed + index for index in range(config.num_samples)),
        retain_seeds=tuple(config.seed + 100_000 + index for index in range(config.num_samples)),
    )


def save_evaluation_conditions(conditions: EvaluationConditions, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        forget_embeddings=conditions.forget_embeddings.numpy(),
        forget_labels=conditions.forget_labels.numpy(),
        retain_embeddings=conditions.retain_embeddings.numpy(),
        retain_labels=conditions.retain_labels.numpy(),
        forget_source_indices=np.asarray(conditions.forget_source_indices, dtype=np.int64),
        retain_indices=conditions.retain_indices.numpy(),
        forget_seeds=np.asarray(conditions.forget_seeds, dtype=np.int64),
        retain_seeds=np.asarray(conditions.retain_seeds, dtype=np.int64),
    )


def write_split_manifest(split: ExperimentSplit, path: Path) -> None:
    payload = {
        "identity_id": split.identity_id,
        "forget_train_indices": split.forget_train.indices.tolist(),
        "forget_validation_indices": split.forget_validation.indices.tolist(),
        "retain_train_indices": split.retain_train.indices.tolist(),
        "retain_validation_indices": split.retain_validation.indices.tolist(),
        "retain_validation_identity_ids": torch.unique(split.retain_validation.labels).tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def create_embedding_loaders(split: ExperimentSplit, config: PIUConfig | ESDConfig) -> tuple[DataLoader[torch.Tensor], DataLoader[torch.Tensor] | None]:
    forget_embeddings = split.forget_train.embeddings
    if config.forget_sampling == "dirichlet":
        forget_dataset = DirichletEmbeddingDataset(forget_embeddings)
    elif config.forget_sampling == "centroid":
        forget_dataset = F.normalize(forget_embeddings.mean(dim=0, keepdim=True), dim=1)
    else:
        forget_dataset = forget_embeddings
    num_samples = config.training_steps * config.gradient_accumulation_steps * config.batch_size
    forget_sampler = RandomSampler(forget_dataset, replacement=True, num_samples=num_samples, generator=torch.Generator().manual_seed(config.seed))
    forget_loader = DataLoader(forget_dataset, batch_size=config.batch_size, sampler=forget_sampler)
    retain_loader = None
    if config.preservation_weight > 0:
        if not len(split.retain_train.embeddings): raise ValueError("Preservation requires nonempty retain training data")
        retain_sampler = RandomSampler(split.retain_train.embeddings, replacement=True, num_samples=num_samples, generator=torch.Generator().manual_seed(config.seed + 1))
        retain_loader = DataLoader(split.retain_train.embeddings, batch_size=config.batch_size, sampler=retain_sampler)
    return forget_loader, retain_loader


def select_anchor_embedding(split: ExperimentSplit, config: RunConfig) -> tuple[torch.Tensor, int, float]:
    """Select a retained identity centroid from the configured proximity band."""
    embeddings = split.retain_train.embeddings
    labels = split.retain_train.labels
    retain_labels = torch.unique(labels)
    retain_centroids = torch.stack([F.normalize(embeddings[labels == identity].mean(dim=0), dim=0) for identity in retain_labels])
    similarities = retain_centroids @ split.forget_centroid.squeeze(0)
    candidate_indices = torch.where(torch.abs(similarities - config.proximity_threshold) < config.anchor_tolerance)[0]
    if candidate_indices.numel() == 0:
        raise ValueError(f"No anchor identity satisfies |similarity - {config.proximity_threshold}| < {config.anchor_tolerance}.")
    if config.anchor_overrides is not None:
        anchor_id = config.anchor_overrides[config.identity_id][config.proximity_threshold]
        selected_index = torch.where(retain_labels == anchor_id)[0].item()
    else:
        selected_index = candidate_indices[torch.randint(candidate_indices.numel(), (), generator=torch.Generator().manual_seed(config.seed))].item()
        anchor_id = int(retain_labels[selected_index])
    return retain_centroids[selected_index], int(anchor_id), float(similarities[selected_index])
