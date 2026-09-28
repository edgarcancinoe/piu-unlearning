from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from piu_unlearning.models.arc2face_text_encoder import CLIPTextModelWrapper
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    from piu_unlearning.config import PIUConfig
    from piu_unlearning.data import EvaluationConditions


@dataclass(frozen=True)
class GeneratedSamples:
    forget: list[Path]
    retain: list[Path]


class Arc2FaceIdentityConditioner:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, text_encoder: CLIPTextModelWrapper) -> None:
        self.text_encoder = text_encoder
        self.identity_token_id = tokenizer.encode("id", add_special_tokens=False)[0]
        self.input_ids = tokenizer("photo of a id person", return_tensors="pt", padding="max_length", max_length=tokenizer.model_max_length).input_ids

    def encode(self, face_embeddings: torch.Tensor) -> torch.Tensor:
        input_ids = self.input_ids.to(face_embeddings.device)
        input_ids = input_ids.repeat(face_embeddings.shape[0], 1)
        token_embeddings = self.text_encoder(input_ids=input_ids, return_token_embs=True)
        face_embeddings = F.pad(face_embeddings, (0, self.text_encoder.config.hidden_size - face_embeddings.shape[-1])).to(token_embeddings.dtype)
        token_embeddings[input_ids == self.identity_token_id] = face_embeddings
        return self.text_encoder(input_ids=input_ids, input_token_embs=token_embeddings)[0]


def select_trainable_layers(unet: UNet2DConditionModel, surgical_layers: tuple[str, ...]) -> None:
    for name, parameter in unet.named_parameters():
        parameter.requires_grad = "attn2" in name and any(layer in name for layer in surgical_layers)


def load_arc2face(config: PIUConfig) -> StableDiffusionPipeline:
    from piu_unlearning.models.arc2face_text_encoder import CLIPTextModelWrapper
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline, UNet2DConditionModel

    text_encoder = CLIPTextModelWrapper.from_pretrained(config.arc2face_model, subfolder="encoder", revision=config.arc2face_revision, torch_dtype=torch.float32)
    unet = UNet2DConditionModel.from_pretrained(config.arc2face_model, subfolder="arc2face", revision=config.arc2face_revision, torch_dtype=torch.float32)
    pipeline = StableDiffusionPipeline.from_pretrained(config.base_model, revision=config.base_model_revision, text_encoder=text_encoder, unet=unet, torch_dtype=torch.float32, safety_checker=None)
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.vae.requires_grad_(False)

    select_trainable_layers(pipeline.unet, config.surgical_layers)

    return pipeline.to(config.device)


@torch.no_grad()
def generate_conditioned_samples(model: StableDiffusionPipeline, embeddings: torch.Tensor, seeds: tuple[int, ...], config: PIUConfig, output_dir: Path) -> list[Path]:
    """Generate one image for each identity embedding and seed."""
    identity_conditioner = Arc2FaceIdentityConditioner(model.tokenizer, model.text_encoder)
    identity_conditioning = identity_conditioner.encode(embeddings.to(model.device))
    output_dir.mkdir(parents=True, exist_ok=True)
    model.unet.eval()
    image_paths = []
    for index, seed in enumerate(seeds):
        generator = torch.Generator(device=model.device).manual_seed(seed)
        image = model(prompt_embeds=identity_conditioning[index : index + 1], num_inference_steps=config.num_inference_steps, guidance_scale=config.guidance_scale, output_type="pil", generator=generator).images[0]
        image_path = output_dir / f"{index:04d}.png"
        image.save(image_path)
        image_paths.append(image_path)
    return image_paths


def generate_evaluation_samples(model: StableDiffusionPipeline, conditions: EvaluationConditions, config: PIUConfig, output_dir: Path) -> GeneratedSamples:
    """Generate the forget and retain images for one evaluation phase."""
    forget = generate_conditioned_samples(model, conditions.forget_embeddings, conditions.forget_seeds, config, output_dir / "forget")
    retain = generate_conditioned_samples(model, conditions.retain_embeddings, conditions.retain_seeds, config, output_dir / "retain")
    return GeneratedSamples(forget=forget, retain=retain)
