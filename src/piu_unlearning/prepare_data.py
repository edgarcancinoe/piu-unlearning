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


def image_verification_context(models_root: Path, extractor: ArcFaceExtractor) -> dict:
    from importlib.metadata import version
    import onnxruntime
    import torch

    hardware = torch.cuda.get_device_name(torch.device(extractor.device)) if extractor.device.startswith("cuda") else "cpu"
    versions = {"insightface": version("insightface"), "onnxruntime": onnxruntime.__version__, "torch": torch.__version__, "numpy": np.__version__, "pillow": version("pillow")}
    models = {path.name: sha256(path) for path in sorted((models_root / "models" / "antelopev2").glob("*.onnx"))}
    providers = {name: model.session.get_provider_options() for name, model in extractor.app.models.items()}
    return {"device": extractor.device, "hardware": hardware, "versions": versions, "models": models, "providers": providers, "det_size": DET_SIZE, "det_thresh": DET_THRESH}


def load_verification_cache(path: Path, context: str) -> dict:
    cached = {}
    if not path.exists(): return cached
    with path.open("r+b") as stream:
        while True:
            start, line = stream.tell(), stream.readline()
            if not line: break
            # An interrupted append must not corrupt the next run's records.
            if not line.endswith(b"\n"):
                stream.truncate(start)
                break
            record = json.loads(line)
            if record.get("context") == context and "index" in record: cached[record["index"]] = record
    return cached


def verify_image_row(path: Path, stored: np.ndarray, extractor: ArcFaceExtractor) -> dict:
    from PIL import Image

    try:
        digest = sha256(path)
        with Image.open(path) as image: embedding = extractor(np.asarray(image.convert("RGB")))
    except OSError as error:
        return {"sha256": None, "cosine": None, "error": str(error)}
    if embedding is None: return {"sha256": digest, "cosine": None, "error": "No face found"}
    denominator = np.linalg.norm(embedding) * np.linalg.norm(stored)
    similarity = float(np.dot(embedding, stored) / denominator) if denominator else float("nan")
    cosine = similarity if np.isfinite(similarity) else None
    error = None if cosine is not None and cosine >= 0.99 else "Image/embedding mismatch"
    return {"sha256": digest, "cosine": cosine, "error": error}


def prepare_image_manifest(data_dir: Path, image_paths: Path, image_root: Path, extractor: ArcFaceExtractor, *, verification_context: dict | None = None, check_rows: list[int] | None = None) -> Path:
    """Audit all requested pairs; publish a manifest only after a complete passing audit."""
    from tqdm import tqdm

    names = [Path(line.strip()).name for line in image_paths.read_text(encoding="utf-8").splitlines()]
    embeddings, labels = np.load(data_dir / "embeddings.npy"), np.load(data_dir / "labels.npy")
    if len(names) != len(embeddings) or len(names) != len(labels): raise ValueError("Image names, embeddings, and labels must have the same row count")
    if any(not name for name in names) or len(set(names)) != len(names): raise ValueError("Image names must be nonempty and unique; use a flat image directory")
    indices = list(range(len(names))) if check_rows is None else list(dict.fromkeys(check_rows))
    if not indices or any(index < 0 or index >= len(names) for index in indices): raise ValueError("Check rows must be nonempty, zero-based indices within the dataset")
    hashes = {"embeddings_sha256": sha256(data_dir / "embeddings.npy"), "labels_sha256": sha256(data_dir / "labels.npy"), "source_paths_sha256": sha256(image_paths)}
    metadata = {"version": 1, "threshold": 0.99, **hashes, "extractor": verification_context}
    context = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    cache_path = data_dir / "image_verification.jsonl"
    cached = load_verification_cache(cache_path, context)
    if verification_context is None: cached = {}  # Unidentified custom extractors cannot safely reuse checks.
    records, reused = [], 0
    with cache_path.open("a", encoding="utf-8") as log, tqdm(indices, desc="Verifying image/embedding rows") as progress:
        log.write(json.dumps({"context": context, "metadata": metadata}) + "\n")
        log.flush()
        for index in progress:
            name, record = names[index], cached.get(index)
            path = image_root / name
            if record and record["error"] is None and path.is_file() and record["sha256"] == sha256(path):
                reused += 1
            else:
                record = {"context": context, "index": index, "path": name, **verify_image_row(path, embeddings[index].astype(np.float32), extractor)}
                log.write(json.dumps(record, allow_nan=False) + "\n")
                log.flush()
            records.append(record)
            if check_rows is not None or record["error"] is not None:
                progress.write(f"row={index} file={name} cosine={record['cosine']} status={record['error'] or 'passed'}")
    failures = [record for record in records if record["error"] is not None]
    report_path = data_dir / ("image_verification_report.json" if check_rows is None else "image_verification_check.json")
    report = {"metadata": metadata, "checked_rows": len(records), "reused_rows": reused, "passed_rows": len(records) - len(failures), "failures": failures}
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if failures:
        first = failures[0]
        raise ValueError(f"{len(failures)} image verification failures; first mismatch at row {first['index']}: {first['path']}, cosine={first['cosine']}. No new manifest written. See {report_path}; passing checks are saved in {cache_path}")
    if check_rows is not None: return report_path
    rows = [{"path": record["path"], "sha256": record["sha256"]} for record in records]
    manifest = {"version": 1, "image_root": os.path.relpath(image_root.resolve(), data_dir.resolve()), **hashes, "verification": metadata, "alignment": {"min_cosine": min(record["cosine"] for record in records), "verified_rows": len(rows)}, "rows": rows}
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
    parser.add_argument("--check-rows", type=int, nargs="+", help="Diagnose only these zero-based rows; does not create a training manifest.")
    args = parser.parse_args()
    if args.check_rows is not None and args.download_images: parser.error("--check-rows uses existing local images; omit --download-images")
    paths = args.image_paths or args.data_dir / "file_names.txt"
    if not paths.is_file(): parser.error("Provide --image-paths with the original row-aligned mapping; canonical ordering cannot be guessed")
    root = args.image_root or args.data_dir / "images"
    if args.download_images:
        names = [Path(line.strip()).name for line in paths.read_text(encoding="utf-8").splitlines()]
        materialize_images(download_dataset(args.revision, args.cache_dir), names, root)
    models_root = args.data_dir / "face_models"
    extractor = ArcFaceExtractor(models_root, args.device)
    result = prepare_image_manifest(args.data_dir, paths, root, extractor, verification_context=image_verification_context(models_root, extractor), check_rows=args.check_rows)
    print(f"{'Verification report (no manifest created)' if args.check_rows is not None else 'Verified image manifest'}: {result}")


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
