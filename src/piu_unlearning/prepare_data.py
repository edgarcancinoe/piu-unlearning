"""Compatibility imports for the dataset preparation commands."""

from piu_unlearning.dataset.cli import images_main, main, recompute_main
from piu_unlearning.dataset.hub import CANONICAL_ARTIFACTS, CANONICAL_REVISION, DATASET_REPO, DATASET_REVISION, download_dataset, download_face_models, prepare_canonical, sha256
from piu_unlearning.dataset.images import materialize_images, prepare_dataset_images, write_image_manifest
from piu_unlearning.dataset.recompute import DBSCAN_EPS, DBSCAN_MIN_SAMPLES, assign_singleton_labels, cluster_embeddings, compute_centroids, extract_embeddings, extract_shard, recompute


if __name__ == "__main__":
    main()
