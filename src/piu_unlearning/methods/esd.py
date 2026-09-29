import torch
import torch.nn.functional as F

from piu_unlearning.config import ESDConfig
from piu_unlearning.data import ExperimentSplit
from piu_unlearning.methods.reference import ReferenceIdentity
from piu_unlearning.training.losses import GuidedNoiseLoss


class ESD(GuidedNoiseLoss):
    """Arc2Face ESD adaptation using an encoded synthetic identity as reference."""

    def prepare_reference(self, split: ExperimentSplit, config: ESDConfig) -> ReferenceIdentity:
        shape = (1, split.forget_train.embeddings.shape[-1])
        if config.reference_mode == "zero_uncond":
            embedding = torch.zeros(shape, device=config.device, dtype=split.forget_train.embeddings.dtype)
        else:
            generator = torch.Generator(device=config.device).manual_seed(config.seed)
            embedding = F.normalize(torch.randn(shape, generator=generator, device=config.device, dtype=split.forget_train.embeddings.dtype), dim=-1)
        return ReferenceIdentity(embedding, config.reference_mode)
