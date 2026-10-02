from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

from piu_unlearning.models.arcface import DET_SIZE, DET_THRESH, ArcFaceExtractor

from .hub import DATASET_REPO, DATASET_REVISION, DETECTOR_FILE, DETECTOR_REPO, DETECTOR_REVISION, RECOGNIZER_FILE, RECOGNIZER_REPO, RECOGNIZER_REVISION, download_dataset, download_face_models


DBSCAN_EPS = 0.35
DBSCAN_MIN_SAMPLES = 2


def extract_shard(shard: Path, extractor: ArcFaceExtractor) -> dict[str, np.ndarray]:
    import pyarrow.parquet as pq
    from PIL import Image
    from tqdm import tqdm

    embeddings, names = [], []
    parquet = pq.ParquetFile(shard)
    with tqdm(total=parquet.metadata.num_rows, desc=shard.name, unit="img") as progress:
        for batch in parquet.iter_batches(batch_size=64, columns=["image", "file_name"]):
            for image, name in zip(batch.column("image").to_pylist(), batch.column("file_name").to_pylist()):
                embedding = extractor(np.asarray(Image.open(io.BytesIO(image["bytes"])).convert("RGB")))
                if embedding is None:
                    raise RuntimeError(f"No face detected in {name}")
                embeddings.append(embedding)
                names.append(name)
                progress.update(1)
    return {"embeddings": np.stack(embeddings).astype(np.float32), "file_names": np.asarray(names, dtype=str)}


def extract_embeddings(output_dir: Path, revision: str = DATASET_REVISION, device: str = "cuda", cache_dir: Path | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_cache = output_dir / "embedding_shards" / revision
    shard_cache.mkdir(parents=True, exist_ok=True)
    extractor = ArcFaceExtractor(download_face_models(output_dir / "face_models"), device)
    parts = []
    for shard in download_dataset(revision, cache_dir):
        cache_path = shard_cache / f"{shard.stem}.npz"
        if not cache_path.exists():
            np.savez(cache_path, **extract_shard(shard, extractor))
        with np.load(cache_path) as cached:
            parts.append({name: cached[name] for name in ("embeddings", "file_names")})
    np.save(output_dir / "embeddings.npy", np.concatenate([part["embeddings"] for part in parts]))
    names = np.concatenate([part["file_names"] for part in parts])
    (output_dir / "file_names.txt").write_text("\n".join(names.tolist()) + "\n", encoding="utf-8")


def assign_singleton_labels(labels: np.ndarray) -> np.ndarray:
    labels = labels.copy()
    next_label = int(labels.max()) + 1
    for index in np.flatnonzero(labels == -1):
        labels[index] = next_label
        next_label += 1
    return labels


def compute_centroids(embeddings: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centroid_labels, cluster_sizes = np.unique(labels, return_counts=True)
    centroids = np.stack([embeddings[labels == label].mean(axis=0) for label in centroid_labels]).astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    return centroids, centroid_labels.astype(np.int64), cluster_sizes.astype(np.int64)


def cluster_embeddings(output_dir: Path, revision: str = DATASET_REVISION, eps: float = DBSCAN_EPS, min_samples: int = DBSCAN_MIN_SAMPLES) -> None:
    from sklearn.cluster import DBSCAN

    embeddings = np.load(output_dir / "embeddings.npy").astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    labels = assign_singleton_labels(DBSCAN(metric="cosine", eps=eps, min_samples=min_samples).fit_predict(embeddings))
    centroids, centroid_labels, cluster_sizes = compute_centroids(embeddings, labels)
    np.save(output_dir / "labels.npy", labels)
    np.save(output_dir / "centroids.npy", centroids)
    np.save(output_dir / "centroid_labels.npy", centroid_labels)
    np.save(output_dir / "cluster_sizes.npy", cluster_sizes)
    metadata = {
        "dataset": {"repo": DATASET_REPO, "revision": revision},
        "extractor": {"detector": f"{DETECTOR_REPO}/{DETECTOR_FILE}", "detector_revision": DETECTOR_REVISION, "recognizer": f"{RECOGNIZER_REPO}/{RECOGNIZER_FILE}", "recognizer_revision": RECOGNIZER_REVISION, "det_size": DET_SIZE, "det_thresh": DET_THRESH},
        "clustering": {"algorithm": "DBSCAN", "metric": "cosine", "eps": eps, "min_samples": min_samples, "noise_policy": "singleton"},
        "num_rows": len(labels),
        "num_identities": len(centroid_labels),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Recomputed {len(centroid_labels)} identities from {len(labels)} embeddings.")


def recompute(output_dir: Path, revision: str = DATASET_REVISION, device: str = "cuda", cache_dir: Path | None = None, eps: float = DBSCAN_EPS, min_samples: int = DBSCAN_MIN_SAMPLES) -> None:
    extract_embeddings(output_dir, revision, device, cache_dir)
    cluster_embeddings(output_dir, revision, eps, min_samples)
