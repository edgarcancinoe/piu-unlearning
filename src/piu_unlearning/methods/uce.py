from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from piu_unlearning.config import UCEConfig
from piu_unlearning.data import ExperimentSplit
from piu_unlearning.methods.reference import ReferenceIdentity, proximity_reference
from piu_unlearning.models.arc2face import Arc2FaceIdentityConditioner


def prepare_edit_embeddings(split: ExperimentSplit, reference: ReferenceIdentity, config: UCEConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use only training rows; preserve one centroid per selected retain identity."""
    rng = np.random.default_rng(config.seed)
    forget = split.forget_train.embeddings
    mixtures = []
    for _ in range(config.num_forget_combinations):
        count = min(3, len(forget))
        indices = torch.from_numpy(rng.choice(len(forget), count, replace=False))
        weights = torch.from_numpy(rng.dirichlet(np.ones(count))).to(forget)
        mixtures.append(F.normalize((forget[indices] * weights[:, None]).sum(dim=0), dim=0))
    erase = F.normalize(torch.cat([forget, torch.stack(mixtures)]) if mixtures else forget, dim=1)
    labels = split.retain_train.labels
    preserve_ids = torch.unique(labels)
    if config.num_preserve_ids and len(preserve_ids) > config.num_preserve_ids:
        # Keep the guide in the preserve pool, as in the legacy experiment setup.
        candidates = preserve_ids[preserve_ids != reference.identity_id]
        chosen = rng.choice(len(candidates), config.num_preserve_ids - 1, replace=False)
        preserve_ids = torch.cat([labels.new_tensor([reference.identity_id]), candidates[torch.from_numpy(chosen)]]).sort().values
    preserve = torch.stack([F.normalize(split.retain_train.embeddings[labels == identity].mean(dim=0), dim=0) for identity in preserve_ids]) if len(preserve_ids) else forget.new_empty((0, forget.shape[1]))
    return erase, preserve, preserve_ids


def collect_edit_modules(unet: nn.Module, config: UCEConfig) -> dict[str, nn.Linear]:
    modules = {}
    for name, module in unet.named_modules():
        if not isinstance(module, nn.Linear) or "attn2" not in name.split("."): continue
        if name.rsplit(".", 1)[-1] not in ("to_k", "to_v"): continue
        if config.edit_scope == "all" or any(layer in name for layer in config.surgical_layers): modules[name] = module
    if not modules: raise ValueError("UCE layer selection matched no cross-attention key/value projections")
    return modules


@torch.no_grad()
def build_edit_weights(modules: dict[str, nn.Linear], erase: torch.Tensor, guide: torch.Tensor, preserve: torch.Tensor, config: UCEConfig) -> dict[str, torch.Tensor]:
    """Solve the legacy UCE normal equations using cached token-mean conditions."""
    erase, guide, preserve = erase.float(), guide.float(), preserve.float()
    erase_scale = config.erase_scale / (len(erase) if config.normalize_branch_weights else 1)
    preserve_scale = config.preserve_scale / (max(1, len(preserve)) if config.normalize_branch_weights else 1)
    erase_sum, preserve_sum = erase.sum(dim=0), preserve.sum(dim=0)
    preserve_gram = preserve_scale * (preserve.T @ preserve)
    denominator = config.lamb * torch.eye(erase.shape[1], device=erase.device) + erase_scale * (erase.T @ erase) + preserve_gram
    # Every selected Arc2Face projection uses the same conditioning space.
    lu, pivots = torch.linalg.lu_factor(denominator.T)
    weights = {}
    for name, module in modules.items():
        original = module.weight.float()
        guide_output = F.linear(guide, original, module.bias.float() if module.bias is not None else None)
        numerator = config.lamb * original + erase_scale * (guide_output.T @ erase_sum[None]) + original @ preserve_gram
        if module.bias is not None: numerator += preserve_scale * module.bias.float()[:, None] * preserve_sum[None]
        updated = torch.linalg.lu_solve(lu, pivots, numerator.T).T.to(module.weight.dtype)
        if not torch.isfinite(updated).all(): raise ValueError(f"UCE produced nonfinite weights for {name}")
        weights[f"{name}.weight"] = updated
    return weights


class UCE:
    prepare_reference = staticmethod(proximity_reference)

    @torch.no_grad()
    def edit(self, unet: nn.Module, split: ExperimentSplit, conditioner: Arc2FaceIdentityConditioner, reference: ReferenceIdentity, config: UCEConfig) -> Path:
        modules = collect_edit_modules(unet, config)
        erase, preserve, preserve_ids = prepare_edit_embeddings(split, reference, config)

        def encode(embeddings: torch.Tensor) -> torch.Tensor:
            return torch.cat([conditioner.encode(batch.to(config.device)).mean(dim=1).float() for batch in embeddings.split(config.conditioning_batch_size)])

        guide_conditions = encode(reference.embedding)
        erase_conditions = encode(erase)
        preserve_conditions = encode(preserve) if len(preserve) else guide_conditions.new_empty((0, guide_conditions.shape[1]))
        weights = build_edit_weights(modules, erase_conditions, guide_conditions, preserve_conditions, config)
        for name, module in modules.items(): module.weight.copy_(weights[f"{name}.weight"])
        metadata = {
            "edited_parameters": list(weights),
            "num_erase_embeddings": len(erase),
            "num_preserve_embeddings": len(preserve),
            "preserve_identity_ids": preserve_ids.tolist(),
            "normalize_branch_weights": config.normalize_branch_weights,
            "effective_erase_scale": config.erase_scale / (len(erase) if config.normalize_branch_weights else 1),
            "effective_preserve_scale": config.preserve_scale / (max(1, len(preserve)) if config.normalize_branch_weights else 1),
        }
        output_dir = config.output_dir / "checkpoints"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "edit.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        checkpoint_path = output_dir / "uce_unet.pt"
        torch.save({"method": "uce", "config": json.loads(json.dumps(asdict(config), default=str)), "unet_state_dict": unet.state_dict(), **metadata}, checkpoint_path)
        print(f"UCE edited {len(weights)} projections using {len(erase)} erase embeddings and {len(preserve)} preserve identities", flush=True)
        return checkpoint_path
