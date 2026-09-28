from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from piu_unlearning.anchor_overrides import CELEBAHQ_512_ANCHOR_OVERRIDES


ARC2FACE_MODEL = "FoivosPar/Arc2Face"
ARC2FACE_REVISION = "321709b2e9697ea2a5bd06c8a51b340592333e1e"
BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
BASE_MODEL_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
OUTPUT_DIR = Path("outputs/demo")
DATA_DIR = Path("data/celebahq_512")
SURGICAL_LAYERS = ("down_blocks.2", "mid_block", "up_blocks.1", "up_blocks.2")


@dataclass(frozen=True)
class PIUConfig:
    identity_id: int
    data_dir: Path = DATA_DIR
    output_dir: Path = OUTPUT_DIR
    arc2face_model: str = ARC2FACE_MODEL
    arc2face_revision: str = ARC2FACE_REVISION
    base_model: str = BASE_MODEL
    base_model_revision: str = BASE_MODEL_REVISION
    surgical_layers: tuple[str, ...] = SURGICAL_LAYERS
    device: str = "cuda"
    seed: int = 42
    forget_validation_ratio: float = 0.35
    retain_validation_ratio: float = 0.10
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    training_steps: int = 400
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    preservation_weight: float = 10.0
    negative_guidance_scale: float = 1.0
    proximity_threshold: float = 0.2
    anchor_tolerance: float = 1e-2
    anchor_overrides: dict[int, dict[float, int]] | None = None
    log_every: int = 10
    evaluation_every: int = 50
    evaluation_device: str = "cpu"
    num_samples: int = 25
    num_inference_steps: int = 25
    guidance_scale: float = 3.0
    reuse_baseline: bool = False

    @property
    def embeddings_path(self) -> Path: return self.data_dir / "embeddings.npy"

    @property
    def labels_path(self) -> Path: return self.data_dir / "labels.npy"

    @property
    def centroids_path(self) -> Path: return self.data_dir / "centroids.npy"

    @property
    def centroid_labels_path(self) -> Path: return self.data_dir / "centroid_labels.npy"


def parse_config() -> PIUConfig:
    parser = argparse.ArgumentParser(description="Run a before-and-after PIU identity-unlearning demo.")
    parser.add_argument("--identity-id", type=int, required=True)
    parser.add_argument("--data-dir", type=Path, default=PIUConfig.data_dir)
    parser.add_argument("--output-dir", type=Path, default=PIUConfig.output_dir)
    parser.add_argument("--arc2face-model", default=PIUConfig.arc2face_model)
    parser.add_argument("--arc2face-revision", default=PIUConfig.arc2face_revision)
    parser.add_argument("--base-model", default=PIUConfig.base_model)
    parser.add_argument("--base-model-revision", default=PIUConfig.base_model_revision)
    parser.add_argument("--device", default=PIUConfig.device)
    parser.add_argument("--seed", type=int, default=PIUConfig.seed)
    parser.add_argument("--forget-validation-ratio", type=float, default=PIUConfig.forget_validation_ratio)
    parser.add_argument("--retain-validation-ratio", type=float, default=PIUConfig.retain_validation_ratio)
    parser.add_argument("--batch-size", type=int, default=PIUConfig.batch_size)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=PIUConfig.gradient_accumulation_steps)
    parser.add_argument("--training-steps", type=int, default=PIUConfig.training_steps)
    parser.add_argument("--learning-rate", type=float, default=PIUConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=PIUConfig.weight_decay)
    parser.add_argument("--preservation-weight", type=float, default=PIUConfig.preservation_weight)
    parser.add_argument("--negative-guidance-scale", type=float, default=PIUConfig.negative_guidance_scale)
    parser.add_argument("--proximity-threshold", type=float, default=PIUConfig.proximity_threshold)
    parser.add_argument("--anchor-tolerance", type=float, default=PIUConfig.anchor_tolerance)
    parser.add_argument("--use-anchor-overrides", action="store_true", help="Replay the recorded CelebA-HQ anchor selections.")
    parser.add_argument("--log-every", type=int, default=PIUConfig.log_every)
    parser.add_argument("--evaluation-every", type=int, default=PIUConfig.evaluation_every, help="Run ISM evaluation every N optimizer steps; 0 disables it.")
    parser.add_argument("--evaluation-device", default=PIUConfig.evaluation_device)
    parser.add_argument("--num-samples", type=int, default=PIUConfig.num_samples)
    parser.add_argument("--num-inference-steps", type=int, default=PIUConfig.num_inference_steps)
    parser.add_argument("--guidance-scale", type=float, default=PIUConfig.guidance_scale)
    parser.add_argument("--reuse-baseline", action="store_true", help="Reuse baseline images already present in OUTPUT_DIR/before.")
    values = vars(parser.parse_args())
    use_anchor_overrides = values.pop("use_anchor_overrides")
    values["anchor_overrides"] = CELEBAHQ_512_ANCHOR_OVERRIDES if use_anchor_overrides else None
    return PIUConfig(**values)
