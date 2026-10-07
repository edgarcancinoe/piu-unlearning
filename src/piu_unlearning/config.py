from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from piu_unlearning.anchor_overrides import CELEBAHQ_512_ANCHOR_OVERRIDES


ARC2FACE_MODEL = "FoivosPar/Arc2Face"
ARC2FACE_REVISION = "321709b2e9697ea2a5bd06c8a51b340592333e1e"
BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
BASE_MODEL_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
OUTPUT_DIR = Path("outputs/demo")
DATA_DIR = Path("data/celebahq_512")
PAPER_LABELS_FILE = "srk_labels_eps0.35.npy"
SURGICAL_LAYERS = ("down_blocks.2", "mid_block", "up_blocks.1", "up_blocks.2")


@dataclass(frozen=True)
class RunConfig:
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
    proximity_threshold: float = 0.2
    anchor_tolerance: float = 1e-2
    anchor_overrides: dict[int, dict[float, int]] | None = None
    evaluation_device: str = "cpu"
    num_samples: int = 25
    num_inference_steps: int = 25
    guidance_scale: float = 3.0
    reuse_baseline: bool = False

    def __post_init__(self) -> None:
        for name in ("num_samples", "num_inference_steps"):
            if getattr(self, name) < 1: raise ValueError(f"{name} must be positive")
        for ratio in (self.forget_validation_ratio, self.retain_validation_ratio):
            if not 0 < ratio < 1: raise ValueError("Validation ratios must be between zero and one")
        if self.anchor_overrides is not None and self.proximity_threshold not in self.anchor_overrides.get(self.identity_id, {}):
            raise ValueError(f"No recorded anchor for identity {self.identity_id} at proximity threshold {self.proximity_threshold}; omit --use-anchor-overrides")

    @property
    def embeddings_path(self) -> Path: return self.data_dir / "embeddings.npy"

    @property
    def labels_path(self) -> Path:
        paper_labels = self.data_dir / PAPER_LABELS_FILE
        recomputed_labels = self.data_dir / "labels.npy"
        if paper_labels.is_file() and recomputed_labels.is_file():
            raise ValueError(f"Paper and recomputed labels coexist in {self.data_dir}; use separate data directories")
        return paper_labels if paper_labels.is_file() else recomputed_labels

    @property
    def centroids_path(self) -> Path: return self.data_dir / "centroids.npy"

    @property
    def centroid_labels_path(self) -> Path: return self.data_dir / "centroid_labels.npy"


@dataclass(frozen=True)
class TrainingConfig(RunConfig):
    train_mode: str = "surgical"
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    training_steps: int = 400
    learning_rate: float = 1e-4
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    log_every: int = 10
    evaluation_every: int = 50
    max_grad_norm: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("batch_size", "gradient_accumulation_steps", "training_steps", "log_every"):
            if getattr(self, name) < 1: raise ValueError(f"{name} must be positive")
        if self.evaluation_every < 0: raise ValueError("evaluation_every must be nonnegative")
        if self.optimizer not in ("adam", "adamw"): raise ValueError(f"Unknown optimizer: {self.optimizer}")
        if self.train_mode not in ("surgical", "x", "full"): raise ValueError(f"Unknown train_mode: {self.train_mode}")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0: raise ValueError("max_grad_norm must be positive")


@dataclass(frozen=True)
class PIUConfig(TrainingConfig):
    method: str = field(default="piu", init=False)
    forget_sampling: str = "dirichlet"
    preservation_weight: float = 10.0
    negative_guidance_scale: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.preservation_weight < 0: raise ValueError("preservation_weight must be nonnegative")
        if self.forget_sampling not in ("dirichlet", "individual", "centroid"): raise ValueError(f"Unknown forget_sampling: {self.forget_sampling}")


@dataclass(frozen=True)
class SISSConfig(TrainingConfig):
    """Arc2Face SISS defaults from PIU Appendix C.1."""

    method: str = field(default="siss", init=False)
    output_dir: Path = Path("outputs/siss_demo")
    train_mode: str = "full"
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    training_steps: int = 60
    learning_rate: float = 5e-6
    weight_decay: float = 0.0
    max_grad_norm: float | None = 1.0
    beta: float = 0.1
    use_ema: bool = True
    gradient_batch_size: int = 16
    gradient_checkpointing: bool = True
    num_preserve_ids: int = 1000
    image_manifest: Path | None = None
    image_root: Path | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0 <= self.beta < float("inf"): raise ValueError("SISS beta must be finite and nonnegative")
        if self.anchor_overrides is not None: raise ValueError("SISS does not use anchors")
        if self.gradient_batch_size < 1: raise ValueError("gradient_batch_size must be positive")
        if self.num_preserve_ids < 0: raise ValueError("num_preserve_ids must be nonnegative")
        if self.gradient_batch_size % self.batch_size: raise ValueError("SISS batch_size must divide gradient_batch_size")
        if self.batch_size * self.gradient_accumulation_steps % self.gradient_batch_size: raise ValueError("SISS effective batch must be a multiple of gradient_batch_size")


@dataclass(frozen=True)
class UCEConfig(RunConfig):
    """PIU Appendix C.2 weights; sampling defaults from the legacy final-run script."""

    method: str = field(default="uce", init=False)
    output_dir: Path = Path("outputs/uce_demo")
    erase_scale: float = 20.0
    preserve_scale: float = 1.0
    lamb: float = 0.7
    edit_scope: str = "all"
    conditioning_batch_size: int = 64
    num_forget_combinations: int = 0
    num_preserve_ids: int = 20
    normalize_branch_weights: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.erase_scale < 0 or self.preserve_scale < 0: raise ValueError("UCE branch weights must be nonnegative")
        if self.lamb <= 0: raise ValueError("lamb must be positive")
        if self.edit_scope not in ("all", "surgical"): raise ValueError(f"Unknown edit_scope: {self.edit_scope}")
        if self.conditioning_batch_size < 1: raise ValueError("conditioning_batch_size must be positive")
        if self.num_forget_combinations < 0 or self.num_preserve_ids < 0: raise ValueError("UCE sampling counts must be nonnegative")


@dataclass(frozen=True)
class WIDConfig(TrainingConfig):
    """Single-step Arc2Face WID; learning values selected by the legacy final-run script."""

    method: str = field(default="wid", init=False)
    output_dir: Path = Path("outputs/wid_demo")
    train_mode: str = "full"
    batch_size: int = 4
    gradient_accumulation_steps: int = 16
    training_steps: int = 100
    learning_rate: float = 5e-6
    weight_decay: float = 0.0
    max_grad_norm: float | None = 1.0
    model_loss_weight: float = 1.0
    identity_loss_weight: float = 0.1
    preservation_weight: float = 0.0
    image_manifest: Path | None = None
    image_root: Path | None = None
    identity_checkpoint: Path | None = None
    identity_channel_order: str = "bgr"

    def __post_init__(self) -> None:
        super().__post_init__()
        if min(self.model_loss_weight, self.identity_loss_weight, self.preservation_weight) < 0: raise ValueError("WID loss weights must be nonnegative")
        if self.model_loss_weight + self.identity_loss_weight == 0: raise ValueError("WID needs a model or identity objective")
        if self.identity_channel_order not in ("rgb", "bgr"): raise ValueError("identity_channel_order must be rgb or bgr")


def parse_config(argv: list[str] | None = None) -> PIUConfig | SISSConfig | UCEConfig | WIDConfig:
    method_parser = argparse.ArgumentParser(add_help=False)
    method_parser.add_argument("--method", choices=("piu", "siss", "uce", "wid"), default="piu")
    method_args, _ = method_parser.parse_known_args(argv)
    config_type = {"piu": PIUConfig, "siss": SISSConfig, "uce": UCEConfig, "wid": WIDConfig}[method_args.method]
    parser = argparse.ArgumentParser(description="Run a before-and-after identity-unlearning demo.", parents=[method_parser])
    parser.add_argument("--identity-id", type=int, required=True)
    parser.add_argument("--data-dir", type=Path, default=config_type.data_dir)
    parser.add_argument("--output-dir", type=Path, default=config_type.output_dir)
    parser.add_argument("--arc2face-model", default=config_type.arc2face_model)
    parser.add_argument("--arc2face-revision", default=config_type.arc2face_revision)
    parser.add_argument("--base-model", default=config_type.base_model)
    parser.add_argument("--base-model-revision", default=config_type.base_model_revision)
    parser.add_argument("--surgical-layers", nargs="+", default=SURGICAL_LAYERS)
    parser.add_argument("--device", default=config_type.device)
    parser.add_argument("--seed", type=int, default=config_type.seed)
    parser.add_argument("--forget-validation-ratio", type=float, default=config_type.forget_validation_ratio)
    parser.add_argument("--retain-validation-ratio", type=float, default=config_type.retain_validation_ratio)
    if method_args.method == "uce":
        parser.add_argument("--erase-scale", type=float, default=UCEConfig.erase_scale)
        parser.add_argument("--preserve-scale", type=float, default=UCEConfig.preserve_scale)
        parser.add_argument("--lamb", type=float, default=UCEConfig.lamb, help="Regularization toward the original weights.")
        parser.add_argument("--edit-scope", choices=("all", "surgical"), default=UCEConfig.edit_scope)
        parser.add_argument("--conditioning-batch-size", type=int, default=UCEConfig.conditioning_batch_size)
        parser.add_argument("--num-forget-combinations", type=int, default=UCEConfig.num_forget_combinations)
        parser.add_argument("--num-preserve-ids", type=int, default=UCEConfig.num_preserve_ids, help="Retain identities used in the edit; 0 uses all retain-training identities.")
        parser.add_argument("--normalize-branch-weights", action=argparse.BooleanOptionalAction, default=UCEConfig.normalize_branch_weights)
    else:
        parser.add_argument("--train-mode", choices=("surgical", "x", "full"), default=config_type.train_mode, help="Surgical cross-attention, all cross-attention (x), or the full U-Net.")
        if method_args.method == "piu": parser.add_argument("--forget-sampling", choices=("dirichlet", "individual", "centroid"), default=config_type.forget_sampling)
        parser.add_argument("--batch-size", type=int, default=config_type.batch_size)
        parser.add_argument("--gradient-accumulation-steps", type=int, default=config_type.gradient_accumulation_steps)
        parser.add_argument("--training-steps", type=int, default=config_type.training_steps)
        parser.add_argument("--learning-rate", type=float, default=config_type.learning_rate)
        parser.add_argument("--optimizer", choices=("adam", "adamw"), default=config_type.optimizer)
        parser.add_argument("--weight-decay", type=float, default=config_type.weight_decay)
        if method_args.method != "siss": parser.add_argument("--preservation-weight", type=float, default=config_type.preservation_weight)
        if method_args.method == "piu": parser.add_argument("--negative-guidance-scale", type=float, default=config_type.negative_guidance_scale)
        parser.add_argument("--max-grad-norm", type=float, default=config_type.max_grad_norm)
        parser.add_argument("--log-every", type=int, default=config_type.log_every)
        parser.add_argument("--evaluation-every", type=int, default=config_type.evaluation_every, help="Run ISM evaluation every N optimizer steps; 0 disables it.")
    if method_args.method == "siss":
        parser.add_argument("--beta", type=float, default=SISSConfig.beta, help="Forget gradient norm relative to retain gradient norm.")
        parser.add_argument("--num-preserve-ids", type=int, default=SISSConfig.num_preserve_ids, help="Retain-training identities; 0 uses all.")
        parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=SISSConfig.use_ema)
        parser.add_argument("--gradient-batch-size", type=int, default=SISSConfig.gradient_batch_size, help="Examples per branch used to compute each SISS gradient norm ratio; independent of GPU batch size.")
        parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=SISSConfig.gradient_checkpointing)
    if method_args.method in ("siss", "wid"):
        parser.add_argument("--image-manifest", type=Path, help="Row-aligned manifest produced by piu-prepare-images.")
        parser.add_argument("--image-root", type=Path, help="Override the image root recorded in the manifest.")
    if method_args.method == "wid":
        parser.add_argument("--identity-checkpoint", type=Path, help="Local IR-SE50 state-dict checkpoint; omitted downloads pinned pretrained weights automatically.")
        parser.add_argument("--identity-channel-order", choices=("rgb", "bgr"), default=WIDConfig.identity_channel_order)
        parser.add_argument("--model-loss-weight", type=float, default=WIDConfig.model_loss_weight)
        parser.add_argument("--identity-loss-weight", type=float, default=WIDConfig.identity_loss_weight)
    if method_args.method in ("piu", "uce", "wid"):
        parser.add_argument("--proximity-threshold", type=float, default=PIUConfig.proximity_threshold)
        parser.add_argument("--anchor-tolerance", type=float, default=PIUConfig.anchor_tolerance)
    parser.add_argument("--use-anchor-overrides", action="store_true", help="Replay the recorded CelebA-HQ anchor selections.")
    parser.add_argument("--evaluation-device", default=config_type.evaluation_device)
    parser.add_argument("--num-samples", type=int, default=config_type.num_samples)
    parser.add_argument("--num-inference-steps", type=int, default=config_type.num_inference_steps)
    parser.add_argument("--guidance-scale", type=float, default=config_type.guidance_scale)
    parser.add_argument("--reuse-baseline", action="store_true", help="Reuse baseline images already present in OUTPUT_DIR/before.")
    values = vars(parser.parse_args(argv))
    values.pop("method")
    use_anchor_overrides = values.pop("use_anchor_overrides")
    if method_args.method == "siss" and use_anchor_overrides: parser.error("--use-anchor-overrides is only supported for PIU, UCE, and WID")
    values["anchor_overrides"] = CELEBAHQ_512_ANCHOR_OVERRIDES if use_anchor_overrides else None
    values["surgical_layers"] = tuple(values["surgical_layers"])
    try:
        return config_type(**values)
    except ValueError as error:
        parser.error(str(error))
