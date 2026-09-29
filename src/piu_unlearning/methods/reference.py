from dataclasses import dataclass

import torch

from piu_unlearning.config import RunConfig
from piu_unlearning.data import ExperimentSplit, select_anchor_embedding


@dataclass(frozen=True)
class ReferenceIdentity:
    embedding: torch.Tensor
    mode: str
    identity_id: int | None = None
    similarity: float | None = None


def proximity_reference(split: ExperimentSplit, config: RunConfig) -> ReferenceIdentity:
    embedding, identity_id, similarity = select_anchor_embedding(split, config)
    return ReferenceIdentity(embedding.unsqueeze(0).to(config.device), "proximity", identity_id, similarity)
