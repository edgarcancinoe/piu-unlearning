from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from piu_unlearning.config import PAPER_LABELS_FILE

from .hub import DATASET_REPO, DATASET_REVISION, download_dataset, sha256


def materialize_images(shards: list[Path], image_root: Path, expected_names: list[str] | None = None) -> list[str]:
    import pyarrow.parquet as pq

    names, found = [], set()
    image_root.mkdir(parents=True, exist_ok=True)
    for shard in shards:
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=64, columns=["image", "file_name"]):
            for image, name in zip(batch.column("image").to_pylist(), batch.column("file_name").to_pylist()):
                if not name or Path(name).name != name:
                    raise ValueError(f"Dataset image name must be a plain file name: {name!r}")
                if name in found:
                    raise ValueError(f"Duplicate dataset image name: {name}")
                if expected_names is not None and (len(names) >= len(expected_names) or name != expected_names[len(names)]):
                    raise ValueError(f"Image row order differs from file_names.txt at row {len(names)}: {name}")
                names.append(name)
                found.add(name)
                destination = image_root / name
                if destination.exists():
                    if destination.read_bytes() != image["bytes"]:
                        raise ValueError(f"Image already exists with different contents: {destination}")
                else:
                    destination.write_bytes(image["bytes"])
    if expected_names is not None and len(names) != len(expected_names):
        raise ValueError(f"Dataset has {len(names)} rows; file_names.txt has {len(expected_names)}")
    return names


def write_image_manifest(data_dir: Path, image_paths: Path, image_root: Path, revision: str | None, manifest_path: Path | None = None) -> Path:
    """Record the dataset row mapping and image/file hashes used for training."""
    from PIL import Image
    from tqdm import tqdm

    names = image_paths.read_text(encoding="utf-8").splitlines()
    label_path = data_dir / PAPER_LABELS_FILE if (data_dir / PAPER_LABELS_FILE).is_file() else data_dir / "labels.npy"
    embeddings, labels = np.load(data_dir / "embeddings.npy", mmap_mode="r"), np.load(label_path, mmap_mode="r")
    if len(names) != len(embeddings) or len(names) != len(labels):
        raise ValueError("Image names, embeddings, and labels must have the same row count")
    if any(not name or Path(name).name != name for name in names) or len(set(names)) != len(names):
        raise ValueError("Image names must be nonempty, unique plain file names")
    rows = []
    for name in tqdm(names, desc="Indexing images", unit="image"):
        path = image_root / name
        with Image.open(path) as image:
            image.verify()
        rows.append({"path": name, "sha256": sha256(path)})
    manifest = {
        "version": 2,
        "image_root": os.path.relpath(image_root.resolve(), data_dir.resolve()),
        "embeddings_sha256": sha256(data_dir / "embeddings.npy"),
        "labels_sha256": sha256(label_path),
        "source_paths_sha256": sha256(image_paths),
        "source": {"dataset": DATASET_REPO, "revision": revision, "row_order": "dataset_parquet" if revision else "file_names.txt"},
        "rows": rows,
    }
    path = manifest_path or data_dir / "image_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def prepare_dataset_images(data_dir: Path, image_root: Path | None = None, manifest_path: Path | None = None, revision: str = DATASET_REVISION, cache_dir: Path | None = None) -> Path:
    image_paths = data_dir / "file_names.txt"
    image_root = image_root or data_dir / "images"
    manifest = manifest_path or data_dir / "image_manifest.json"
    if manifest.is_file() and image_paths.is_file():
        label_path = data_dir / PAPER_LABELS_FILE if (data_dir / PAPER_LABELS_FILE).is_file() else data_dir / "labels.npy"
        names = image_paths.read_text(encoding="utf-8").splitlines()
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        if (
            metadata.get("version") == 2
            and metadata.get("embeddings_sha256") == sha256(data_dir / "embeddings.npy")
            and metadata.get("labels_sha256") == sha256(label_path)
            and metadata.get("source_paths_sha256") == sha256(image_paths)
            and [row["path"] for row in metadata["rows"]] == names
            and all((image_root / name).is_file() for name in names)
        ):
            return manifest
    names = image_paths.read_text(encoding="utf-8").splitlines() if image_paths.is_file() else None
    names = materialize_images(download_dataset(revision, cache_dir), image_root, names)
    if not image_paths.exists():
        image_paths.write_text("\n".join(names) + "\n", encoding="utf-8")
    return write_image_manifest(data_dir, image_paths, image_root, revision, manifest)
