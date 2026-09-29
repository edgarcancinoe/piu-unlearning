from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
from pathlib import Path

import numpy as np

from piu_unlearning.config import DATA_DIR
from piu_unlearning.models.arcface import DET_SIZE, DET_THRESH, ArcFaceExtractor


DATASET_REPO = "edgarcancinoe/celebahq_512_id_clusters"
DATASET_REVISION = "09617f99a0f31ea39b4fd7486161b2a7f75f16c8"
CANONICAL_REVISION = "47ac08e6c0f80da13752b8f2694a515da26f352e"
CANONICAL_ARTIFACTS = {
    "embeddings.npy": ("paper_artifacts/embeddings.npy", "ae68d164dc633fb0942ee97215b64cfa94b92d9c358d6e7be41a23a8eb1746e6"),
    "labels.npy": ("paper_artifacts/srk_labels_eps0.35.npy", "7428ab95f25f245fc53d10dcc0178ad30587c5a854bc57c8a0a737fe1625d79f"),
    "centroids.npy": ("paper_artifacts/srk_centroids_eps0.35.npy", "8479af129822f102bb4245708c66a7ee14631eccc6c2f6fcc87efc4d732f1a03"),
}
DETECTOR_REPO, DETECTOR_FILE = "DIAMONIK7777/antelopev2", "scrfd_10g_bnkps.onnx"
DETECTOR_REVISION = "ba0c3e10f4548361eb9a63265d87ce1140ab5a05"
RECOGNIZER_REPO, RECOGNIZER_FILE = "FoivosPar/Arc2Face", "arcface.onnx"
RECOGNIZER_REVISION = "321709b2e9697ea2a5bd06c8a51b340592333e1e"
DBSCAN_EPS = 0.35
DBSCAN_MIN_SAMPLES = 2


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


def prepare_canonical(output_dir: Path, revision: str, cache_dir: Path | None) -> None:
    """Download the exact embeddings, labels, and centroids used by the paper."""
    from huggingface_hub import hf_hub_download

    output_dir.mkdir(parents=True, exist_ok=True)
    for local_name, (remote_name, expected_hash) in CANONICAL_ARTIFACTS.items():
        source = Path(hf_hub_download(DATASET_REPO, remote_name, repo_type="dataset", revision=revision, cache_dir=cache_dir))
        if sha256(source) != expected_hash:
            raise RuntimeError(f"Checksum mismatch for {remote_name}")
        shutil.copyfile(source, output_dir / local_name)
    labels = np.load(output_dir / "labels.npy")
    np.save(output_dir / "centroid_labels.npy", np.unique(labels).astype(np.int64))
    download_face_models(output_dir / "face_models")
    metadata = {"dataset": {"repo": DATASET_REPO, "revision": revision}, "artifacts": {name: {"source": source, "sha256": checksum} for name, (source, checksum) in CANONICAL_ARTIFACTS.items()}, "num_rows": len(labels), "num_identities": len(np.unique(labels))}
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def materialize_images(shards: list[Path], names: list[str], image_root: Path) -> None:
    import pyarrow.parquet as pq

    wanted, found = set(names), set()
    image_root.mkdir(parents=True, exist_ok=True)
    for shard in shards:
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=64, columns=["image", "file_name"]):
            for image, raw_name in zip(batch.column("image").to_pylist(), batch.column("file_name").to_pylist()):
                name = Path(raw_name).name
                if name not in wanted: continue
                if name in found: raise ValueError(f"Duplicate dataset image name: {name}")
                found.add(name)
                destination = image_root / name
                if destination.exists():
                    if destination.read_bytes() != image["bytes"]: raise ValueError(f"Image already exists with different contents: {destination}")
                else: destination.write_bytes(image["bytes"])
    if found != wanted: raise ValueError(f"Dataset is missing {len(wanted - found)} requested images")


def prepare_image_manifest(data_dir: Path, image_paths: Path, image_root: Path, extractor: ArcFaceExtractor) -> Path:
    """Bind explicitly ordered image names to embeddings after checking every pair."""
    from PIL import Image
    from tqdm import tqdm

    names = [Path(line.strip()).name for line in image_paths.read_text(encoding="utf-8").splitlines()]
    embeddings, labels = np.load(data_dir / "embeddings.npy"), np.load(data_dir / "labels.npy")
    if len(names) != len(embeddings) or len(names) != len(labels): raise ValueError("Image names, embeddings, and labels must have the same row count")
    if any(not name for name in names) or len(set(names)) != len(names): raise ValueError("Image names must be nonempty and unique; use a flat image directory")
    rows, similarities = [], []
    for index, name in enumerate(tqdm(names, desc="Verifying image/embedding rows")):
        path = image_root / name
        with Image.open(path) as image: embedding = extractor(np.asarray(image.convert("RGB")))
        if embedding is None: raise ValueError(f"No face found in row {index}: {name}")
        stored = embeddings[index].astype(np.float32)
        similarity = float(np.dot(embedding, stored) / (np.linalg.norm(embedding) * np.linalg.norm(stored)))
        if not np.isfinite(similarity) or similarity < 0.99: raise ValueError(f"Image/embedding mismatch at row {index}: {name}, cosine={similarity:.6f}")
        similarities.append(similarity)
        rows.append({"path": name, "sha256": sha256(path)})
    manifest = {"version": 1, "image_root": os.path.relpath(image_root.resolve(), data_dir.resolve()), "embeddings_sha256": sha256(data_dir / "embeddings.npy"), "labels_sha256": sha256(data_dir / "labels.npy"), "source_paths_sha256": sha256(image_paths), "alignment": {"min_cosine": min(similarities), "verified_rows": len(rows)}, "rows": rows}
    path = data_dir / "image_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def images_main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and verify real images for WID without changing embedding rows or labels.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--image-paths", type=Path, help="Original row-aligned image_paths.txt; defaults to recomputed file_names.txt.")
    parser.add_argument("--image-root", type=Path, help="Flat image directory; defaults to DATA_DIR/images.")
    parser.add_argument("--download-images", action="store_true", help="Materialize named images from the dataset parquet shards.")
    parser.add_argument("--revision", default=DATASET_REVISION)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cpu", help="Device used for one-time image/embedding verification.")
    args = parser.parse_args()
    paths = args.image_paths or args.data_dir / "file_names.txt"
    if not paths.is_file(): parser.error("Provide --image-paths with the original row-aligned mapping; canonical ordering cannot be guessed")
    root = args.image_root or args.data_dir / "images"
    if args.download_images:
        names = [Path(line.strip()).name for line in paths.read_text(encoding="utf-8").splitlines()]
        materialize_images(download_dataset(args.revision, args.cache_dir), names, root)
    extractor = ArcFaceExtractor(args.data_dir / "face_models", args.device)
    print(f"Verified image manifest: {prepare_image_manifest(args.data_dir, paths, root, extractor)}")


def download_dataset(revision: str, cache_dir: Path | None) -> list[Path]:
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(DATASET_REPO, repo_type="dataset", revision=revision, allow_patterns="data/*.parquet", cache_dir=cache_dir)
    shards = sorted(Path(snapshot).glob("data/*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No parquet shards found in {snapshot}")
    return shards


def extract_shard(shard: Path, extractor: ArcFaceExtractor) -> dict[str, np.ndarray]:
    import pyarrow.parquet as pq
    from PIL import Image
    from tqdm import tqdm

    embeddings, reference_labels, names = [], [], []
    parquet = pq.ParquetFile(shard)
    with tqdm(total=parquet.metadata.num_rows, desc=shard.name, unit="img") as progress:
        for batch in parquet.iter_batches(batch_size=64, columns=["image", "file_name", "cluster_id"]):
            for image, name, label in zip(batch.column("image").to_pylist(), batch.column("file_name").to_pylist(), batch.column("cluster_id").to_pylist()):
                embedding = extractor(np.asarray(Image.open(io.BytesIO(image["bytes"])).convert("RGB")))
                if embedding is None:
                    raise RuntimeError(f"No face detected in {name}")
                embeddings.append(embedding)
                reference_labels.append(int(label))
                names.append(name)
                progress.update(1)
    return {"embeddings": np.stack(embeddings).astype(np.float32), "reference_labels": np.asarray(reference_labels, dtype=np.int64), "file_names": np.asarray(names, dtype=str)}


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
    reference_labels = np.concatenate([part["reference_labels"] for part in parts])
    file_names = np.concatenate([part["file_names"] for part in parts])
    np.save(output_dir / "embeddings.npy", embeddings)
    np.save(output_dir / "reference_labels.npy", reference_labels)
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


def cluster_embeddings(output_dir: Path, revision: str, eps: float, min_samples: int) -> None:
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import adjusted_rand_score

    embeddings = np.load(output_dir / "embeddings.npy").astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    reference_labels = np.load(output_dir / "reference_labels.npy")
    labels = assign_singleton_labels(DBSCAN(metric="cosine", eps=eps, min_samples=min_samples).fit_predict(embeddings))
    centroids, centroid_labels, cluster_sizes = compute_centroids(embeddings, labels)
    adjusted_rand = adjusted_rand_score(reference_labels, labels)
    np.save(output_dir / "labels.npy", labels)
    np.save(output_dir / "centroids.npy", centroids)
    np.save(output_dir / "centroid_labels.npy", centroid_labels)
    np.save(output_dir / "cluster_sizes.npy", cluster_sizes)
    metadata = {
        "dataset": {"repo": DATASET_REPO, "revision": revision},
        "extractor": {"detector": f"{DETECTOR_REPO}/{DETECTOR_FILE}", "detector_revision": DETECTOR_REVISION, "recognizer": f"{RECOGNIZER_REPO}/{RECOGNIZER_FILE}", "recognizer_revision": RECOGNIZER_REVISION, "det_size": DET_SIZE, "det_thresh": DET_THRESH},
        "clustering": {"algorithm": "DBSCAN", "metric": "cosine", "eps": eps, "min_samples": min_samples, "noise_policy": "singleton"},
        "comparison": {"reference": "published cluster_id", "adjusted_rand_index": adjusted_rand},
        "num_rows": len(labels),
        "num_identities": len(centroid_labels),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Recomputed {len(centroid_labels)} identities from {len(labels)} embeddings (ARI against published labels: {adjusted_rand:.8f}).")


def recompute(output_dir: Path, revision: str, device: str, cache_dir: Path | None, eps: float, min_samples: int) -> None:
    extract_embeddings(output_dir, revision, device, cache_dir)
    cluster_embeddings(output_dir, revision, eps, min_samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the exact data artifacts used by the PIU paper.")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--revision", default=CANONICAL_REVISION)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    prepare_canonical(args.output_dir, args.revision, args.cache_dir)


def recompute_main() -> None:
    parser = argparse.ArgumentParser(description="Independently recompute ArcFace embeddings, DBSCAN labels, and centroids.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/celebahq_512_recomputed"))
    parser.add_argument("--revision", default=DATASET_REVISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--stage", choices=("all", "embeddings", "clusters"), default="all")
    parser.add_argument("--eps", type=float, default=DBSCAN_EPS)
    parser.add_argument("--min-samples", type=int, default=DBSCAN_MIN_SAMPLES)
    args = parser.parse_args()
    if args.stage == "all":
        recompute(args.output_dir, args.revision, args.device, args.cache_dir, args.eps, args.min_samples)
    elif args.stage == "embeddings":
        extract_embeddings(args.output_dir, args.revision, args.device, args.cache_dir)
    else:
        cluster_embeddings(args.output_dir, args.revision, args.eps, args.min_samples)


if __name__ == "__main__":
    main()
