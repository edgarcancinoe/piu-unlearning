from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from piu_unlearning.config import DATA_DIR
from piu_unlearning.models.arcface import DET_SIZE, DET_THRESH, ArcFaceExtractor


DATASET_REPO = "edgarcancinoe/celebahq_512_id_clusters"
DATASET_REVISION = "09617f99a0f31ea39b4fd7486161b2a7f75f16c8"
DETECTOR_REPO, DETECTOR_FILE = "DIAMONIK7777/antelopev2", "scrfd_10g_bnkps.onnx"
DETECTOR_REVISION = "ba0c3e10f4548361eb9a63265d87ce1140ab5a05"
RECOGNIZER_REPO, RECOGNIZER_FILE = "FoivosPar/Arc2Face", "arcface.onnx"
RECOGNIZER_REVISION = "321709b2e9697ea2a5bd06c8a51b340592333e1e"
EXPECTED_ROWS = 27_996
EXPECTED_IDENTITIES = 9_683
DBSCAN_EPS = 0.35
DBSCAN_MIN_SAMPLES = 2


def download_dataset(revision: str, cache_dir: Path | None) -> list[Path]:
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(DATASET_REPO, repo_type="dataset", revision=revision, allow_patterns="data/*.parquet", cache_dir=cache_dir)
    shards = sorted(Path(snapshot).glob("data/*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No parquet shards found in {snapshot}")
    return shards


def download_face_models(models_root: Path) -> Path:
    from huggingface_hub import hf_hub_download

    model_dir = models_root / "models" / "antelopev2"
    model_dir.mkdir(parents=True, exist_ok=True)
    hf_hub_download(DETECTOR_REPO, DETECTOR_FILE, revision=DETECTOR_REVISION, local_dir=model_dir)
    hf_hub_download(RECOGNIZER_REPO, RECOGNIZER_FILE, revision=RECOGNIZER_REVISION, local_dir=model_dir)
    return models_root


def extract_shard(shard: Path, extractor: ArcFaceExtractor) -> dict[str, np.ndarray]:
    import pyarrow.parquet as pq
    from PIL import Image
    from tqdm import tqdm

    embeddings, labels, names = [], [], []
    parquet = pq.ParquetFile(shard)
    with tqdm(total=parquet.metadata.num_rows, desc=shard.name, unit="img") as progress:
        for batch in parquet.iter_batches(batch_size=64, columns=["image", "file_name", "cluster_id"]):
            for image, name, label in zip(batch.column("image").to_pylist(), batch.column("file_name").to_pylist(), batch.column("cluster_id").to_pylist()):
                embedding = extractor(np.asarray(Image.open(io.BytesIO(image["bytes"])).convert("RGB")))
                if embedding is None:
                    raise RuntimeError(f"No face detected in {name}")
                embeddings.append(embedding)
                labels.append(int(label))
                names.append(name)
                progress.update(1)
    return {"embeddings": np.stack(embeddings).astype(np.float32), "published_labels": np.asarray(labels, dtype=np.int64), "file_names": np.asarray(names, dtype=str)}


def extract_embeddings(output_dir: Path, revision: str, device: str, cache_dir: Path | None) -> None:
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
            parts.append({name: cached[name] for name in cached.files})
    embeddings = np.concatenate([part["embeddings"] for part in parts])
    published_labels = np.concatenate([part["published_labels"] for part in parts])
    file_names = np.concatenate([part["file_names"] for part in parts])
    np.save(output_dir / "embeddings.npy", embeddings)
    np.save(output_dir / "published_labels.npy", published_labels)
    (output_dir / "file_names.txt").write_text("\n".join(file_names.tolist()) + "\n", encoding="utf-8")


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cluster_embeddings(output_dir: Path, revision: str, eps: float, min_samples: int) -> None:
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import adjusted_rand_score

    embeddings = np.load(output_dir / "embeddings.npy").astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    published_labels = np.load(output_dir / "published_labels.npy")
    recomputed_labels = assign_singleton_labels(DBSCAN(metric="cosine", eps=eps, min_samples=min_samples).fit_predict(embeddings))
    adjusted_rand = adjusted_rand_score(published_labels, recomputed_labels)
    if adjusted_rand != 1.0:
        raise RuntimeError(f"Recomputed clustering differs from the published partition (adjusted Rand index: {adjusted_rand:.8f})")
    labels = published_labels.astype(np.int64)
    centroids, centroid_labels, cluster_sizes = compute_centroids(embeddings, labels)
    if len(labels) != EXPECTED_ROWS or len(centroid_labels) != EXPECTED_IDENTITIES:
        raise RuntimeError(f"Expected {EXPECTED_ROWS} rows and {EXPECTED_IDENTITIES} identities, found {len(labels)} and {len(centroid_labels)}")
    np.save(output_dir / "labels.npy", labels)
    np.save(output_dir / "centroids.npy", centroids)
    np.save(output_dir / "centroid_labels.npy", centroid_labels)
    np.save(output_dir / "cluster_sizes.npy", cluster_sizes)
    artifacts = ("embeddings.npy", "labels.npy", "centroids.npy", "centroid_labels.npy", "cluster_sizes.npy")
    metadata = {
        "dataset": {"repo": DATASET_REPO, "revision": revision},
        "extractor": {"detector": f"{DETECTOR_REPO}/{DETECTOR_FILE}", "detector_revision": DETECTOR_REVISION, "recognizer": f"{RECOGNIZER_REPO}/{RECOGNIZER_FILE}", "recognizer_revision": RECOGNIZER_REVISION, "det_size": DET_SIZE, "det_thresh": DET_THRESH},
        "clustering": {"algorithm": "DBSCAN", "metric": "cosine", "eps": eps, "min_samples": min_samples, "noise_policy": "singleton", "adjusted_rand_index": adjusted_rand},
        "num_rows": len(labels),
        "num_identities": len(centroid_labels),
        "published_partition_match": True,
        "sha256": {name: sha256(output_dir / name) for name in artifacts},
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def prepare(output_dir: Path, revision: str, device: str, cache_dir: Path | None, eps: float, min_samples: int) -> None:
    extract_embeddings(output_dir, revision, device, cache_dir)
    cluster_embeddings(output_dir, revision, eps, min_samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare embeddings, identity clusters, and centroids for PIU.")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--revision", default=DATASET_REVISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--stage", choices=("all", "embeddings", "clusters"), default="all")
    parser.add_argument("--eps", type=float, default=DBSCAN_EPS)
    parser.add_argument("--min-samples", type=int, default=DBSCAN_MIN_SAMPLES)
    args = parser.parse_args()
    if args.stage == "all":
        prepare(args.output_dir, args.revision, args.device, args.cache_dir, args.eps, args.min_samples)
    elif args.stage == "embeddings":
        extract_embeddings(args.output_dir, args.revision, args.device, args.cache_dir)
    else:
        cluster_embeddings(args.output_dir, args.revision, args.eps, args.min_samples)


if __name__ == "__main__":
    main()
