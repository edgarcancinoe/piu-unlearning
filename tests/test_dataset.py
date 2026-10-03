import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from piu_unlearning.config import PAPER_LABELS_FILE, PIUConfig, SISSConfig, WIDConfig
from piu_unlearning.dataset.hub import CANONICAL_ARTIFACTS, prepare_canonical
from piu_unlearning.dataset import prepare_for_demo
from piu_unlearning.dataset.recompute import cluster_embeddings, extract_embeddings


class DatasetPreparationTests(unittest.TestCase):
    def test_paper_label_keeps_its_source_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = PIUConfig(identity_id=512, data_dir=root)
            self.assertIn(PAPER_LABELS_FILE, CANONICAL_ARTIFACTS)
            self.assertNotIn("labels.npy", CANONICAL_ARTIFACTS)
            (root / PAPER_LABELS_FILE).touch()
            self.assertEqual(config.labels_path, root / PAPER_LABELS_FILE)

    def test_paper_and_recomputed_data_cannot_share_a_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paper, recomputed = root / PAPER_LABELS_FILE, root / "labels.npy"
            paper.touch()
            recomputed.touch()
            with self.assertRaisesRegex(ValueError, "separate data directories"): PIUConfig(identity_id=512, data_dir=root).labels_path
            with self.assertRaisesRegex(ValueError, "separate directory"): extract_embeddings(root)
            with self.assertRaisesRegex(ValueError, "separate directory"): cluster_embeddings(root)
            with self.assertRaisesRegex(ValueError, "separate directory"): prepare_canonical(root)

    def test_first_demo_use_downloads_paper_data_and_method_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_root, manifest = root / "photos", root / "rows.json"
            config = WIDConfig(identity_id=512, data_dir=root, image_root=image_root, image_manifest=manifest)
            with patch("piu_unlearning.dataset.prepare_canonical") as canonical, patch("piu_unlearning.dataset.prepare_dataset_images") as images:
                prepare_for_demo(config)
            canonical.assert_called_once()
            images.assert_called_once_with(root, image_root, manifest)

    def test_piu_does_not_download_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = PIUConfig(identity_id=512, data_dir=root)
            with patch("piu_unlearning.dataset.prepare_canonical"), patch("piu_unlearning.dataset.prepare_dataset_images") as images:
                prepare_for_demo(config)
            images.assert_not_called()

    def test_existing_data_is_preserved_and_partial_data_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = SISSConfig(identity_id=512, data_dir=root)
            for path in (config.embeddings_path, config.labels_path, config.centroids_path, config.centroid_labels_path):
                path.touch()
            with patch("piu_unlearning.dataset.prepare_canonical") as canonical, patch("piu_unlearning.dataset.prepare_dataset_images"):
                prepare_for_demo(config)
            canonical.assert_not_called()
            config.labels_path.unlink()
            with patch("piu_unlearning.dataset.prepare_canonical") as canonical, patch("piu_unlearning.dataset.prepare_dataset_images"):
                with self.assertRaisesRegex(FileNotFoundError, "incomplete"):
                    prepare_for_demo(config)
            canonical.assert_not_called()


if __name__ == "__main__":
    unittest.main()
