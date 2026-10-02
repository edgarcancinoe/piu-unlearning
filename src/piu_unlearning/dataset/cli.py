from __future__ import annotations

import argparse
from pathlib import Path

from piu_unlearning.config import DATA_DIR
from .hub import DATASET_REVISION, download_dataset, prepare_canonical
from .images import materialize_images, write_image_manifest
from .recompute import DBSCAN_EPS, DBSCAN_MIN_SAMPLES, cluster_embeddings, extract_embeddings, recompute


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the paper's prepared embeddings, labels, and centroids.")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--revision", default=DATASET_REVISION)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    prepare_canonical(args.output_dir, args.revision, args.cache_dir)


def recompute_main() -> None:
    parser = argparse.ArgumentParser(description="Recompute ArcFace embeddings, DBSCAN labels, and centroids from the hosted images.")
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


def images_main() -> None:
    parser = argparse.ArgumentParser(description="Download the hosted images and prepare a row-aligned manifest for SISS/WID.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--image-paths", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--download-images", action="store_true")
    parser.add_argument("--revision", default=DATASET_REVISION)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    paths = args.image_paths or args.data_dir / "file_names.txt"
    root = args.image_root or args.data_dir / "images"
    if args.download_images:
        names = paths.read_text(encoding="utf-8").splitlines() if paths.is_file() else None
        if names is None and args.image_paths is not None:
            parser.error(f"Filename list does not exist: {paths}")
        names = materialize_images(download_dataset(args.revision, args.cache_dir), root, names)
        if not paths.exists():
            paths.parent.mkdir(parents=True, exist_ok=True)
            paths.write_text("\n".join(names) + "\n", encoding="utf-8")
    elif not paths.is_file():
        parser.error("Provide --image-paths or use --download-images to derive filenames from the hosted dataset")
    result = write_image_manifest(args.data_dir, paths, root, args.revision if args.download_images else None)
    print(f"Image manifest: {result}")
