from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from piu_unlearning.config import PAPER_LABELS_FILE


DATASET_REPO = "edgarcancinoe/celebahq_512_id_clusters"
DATASET_REVISION = "47ac08e6c0f80da13752b8f2694a515da26f352e"
CANONICAL_REVISION = DATASET_REVISION
CANONICAL_ARTIFACTS = {
    "embeddings.npy": ("paper_artifacts/embeddings.npy", "ae68d164dc633fb0942ee97215b64cfa94b92d9c358d6e7be41a23a8eb1746e6"),
    PAPER_LABELS_FILE: (f"paper_artifacts/{PAPER_LABELS_FILE}", "7428ab95f25f245fc53d10dcc0178ad30587c5a854bc57c8a0a737fe1625d79f"),
    "centroids.npy": ("paper_artifacts/srk_centroids_eps0.35.npy", "8479af129822f102bb4245708c66a7ee14631eccc6c2f6fcc87efc4d732f1a03"),
}
DETECTOR_REPO, DETECTOR_FILE = "DIAMONIK7777/antelopev2", "scrfd_10g_bnkps.onnx"
DETECTOR_REVISION = "ba0c3e10f4548361eb9a63265d87ce1140ab5a05"
RECOGNIZER_REPO, RECOGNIZER_FILE = "FoivosPar/Arc2Face", "arcface.onnx"
RECOGNIZER_REVISION = "321709b2e9697ea2a5bd06c8a51b340592333e1e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_face_models(models_root: Path) -> Path:
    from huggingface_hub import hf_hub_download

    model_dir = models_root / "models" / "antelopev2"
    model_dir.mkdir(parents=True, exist_ok=True)
    hf_hub_download(DETECTOR_REPO, DETECTOR_FILE, revision=DETECTOR_REVISION, local_dir=model_dir)
    hf_hub_download(RECOGNIZER_REPO, RECOGNIZER_FILE, revision=RECOGNIZER_REVISION, local_dir=model_dir)
    return models_root


def prepare_canonical(output_dir: Path, revision: str = DATASET_REVISION, cache_dir: Path | None = None) -> None:
    """Download the checksummed embeddings, labels, and centroids used in the paper."""
    if (output_dir / "labels.npy").exists(): raise ValueError(f"Recomputed labels exist in {output_dir}; use a separate directory for paper data")
    from huggingface_hub import hf_hub_download

    output_dir.mkdir(parents=True, exist_ok=True)
    for local_name, (remote_name, expected_hash) in CANONICAL_ARTIFACTS.items():
        source = Path(hf_hub_download(DATASET_REPO, remote_name, repo_type="dataset", revision=revision, cache_dir=cache_dir))
        if sha256(source) != expected_hash:
            raise RuntimeError(f"Checksum mismatch for {remote_name}")
        shutil.copyfile(source, output_dir / local_name)
    labels = np.load(output_dir / PAPER_LABELS_FILE)
    np.save(output_dir / "centroid_labels.npy", np.unique(labels).astype(np.int64))
    download_face_models(output_dir / "face_models")
    metadata = {"dataset": {"repo": DATASET_REPO, "revision": revision}, "artifacts": {name: {"source": source, "sha256": checksum} for name, (source, checksum) in CANONICAL_ARTIFACTS.items()}, "num_rows": len(labels), "num_identities": len(np.unique(labels))}
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def download_dataset(revision: str = DATASET_REVISION, cache_dir: Path | None = None) -> list[Path]:
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(DATASET_REPO, repo_type="dataset", revision=revision, allow_patterns="data/*.parquet", cache_dir=cache_dir)
    shards = sorted(Path(snapshot).glob("data/*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No parquet shards found in {snapshot}")
    return shards
