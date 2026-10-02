from __future__ import annotations

from piu_unlearning.config import RunConfig, SISSConfig, WIDConfig

from .hub import CANONICAL_REVISION, DATASET_REVISION, prepare_canonical
from .images import prepare_dataset_images, write_image_manifest


def prepare_for_demo(config: RunConfig) -> None:
    """Fetch the paper data on first use and images for image-based methods."""
    paths = (config.embeddings_path, config.labels_path, config.centroids_path, config.centroid_labels_path)
    if not all(path.is_file() for path in paths):
        if any(path.exists() for path in paths):
            raise FileNotFoundError(f"Prepared dataset is incomplete in {config.data_dir}; run piu-prepare-data to repair it")
        prepare_canonical(config.data_dir, DATASET_REVISION, None)
    if isinstance(config, (SISSConfig, WIDConfig)):
        prepare_dataset_images(config.data_dir, config.image_root, config.image_manifest)


__all__ = ["CANONICAL_REVISION", "DATASET_REVISION", "prepare_canonical", "prepare_dataset_images", "prepare_for_demo", "write_image_manifest"]
