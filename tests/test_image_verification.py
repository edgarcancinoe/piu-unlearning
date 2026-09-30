import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from piu_unlearning.config import WIDConfig
from piu_unlearning.data import load_image_manifest
from piu_unlearning.prepare_data import images_main, prepare_image_manifest


class ImageVerificationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.images = self.root / "images"
        self.images.mkdir()
        self.names = self.root / "file_names.txt"
        self.names.write_text("0.png\n1.png\n2.png\n")
        self.embeddings = np.eye(3, dtype=np.float32)
        np.save(self.root / "embeddings.npy", self.embeddings)
        np.save(self.root / "labels.npy", np.arange(3))
        for index in range(3): Image.new("RGB", (4, 4), (index, 100, 200)).save(self.images / f"{index}.png")
        self.context = {"device": "cpu", "models": "test-model", "version": "test-version"}
        self.calls = []

    def extract(self, image):
        index = int(image[0, 0, 0])
        self.calls.append(index)
        return self.embeddings[index]

    def run_verification(self, extractor=None, **kwargs):
        return prepare_image_manifest(self.root, self.names, self.images, extractor or self.extract, verification_context=self.context, **kwargs)

    def test_full_scan_reports_all_failures_and_resumes_successes(self):
        def bad_extract(image):
            value = self.extract(image)
            return value if int(image[0, 0, 0]) == 1 else np.roll(value, 1)

        with self.assertRaisesRegex(ValueError, "2 image verification failures"): self.run_verification(bad_extract)
        self.assertEqual(self.calls, [0, 1, 2])
        report = json.loads((self.root / "image_verification_report.json").read_text())
        self.assertEqual([row["index"] for row in report["failures"]], [0, 2])
        self.assertFalse((self.root / "image_manifest.json").exists())
        self.calls.clear()
        manifest = self.run_verification()
        self.assertEqual(self.calls, [0, 2])
        self.assertEqual(json.loads(manifest.read_text())["alignment"]["verified_rows"], 3)
        self.calls.clear()
        self.run_verification()
        self.assertEqual(self.calls, [])

    def test_interrupted_scan_resumes_and_discards_partial_log_tail(self):
        def interrupted(image):
            if int(image[0, 0, 0]) == 1: raise KeyboardInterrupt
            return self.extract(image)

        with self.assertRaises(KeyboardInterrupt): self.run_verification(interrupted)
        with (self.root / "image_verification.jsonl").open("ab") as log: log.write(b'{"incomplete":')
        self.calls.clear()
        self.run_verification()
        self.assertEqual(self.calls, [1, 2])
        for line in (self.root / "image_verification.jsonl").read_text().splitlines(): json.loads(line)

    def test_cache_invalidates_changed_images_data_and_runtime(self):
        self.run_verification()
        self.calls.clear()
        Image.new("RGB", (4, 4), (1, 101, 200)).save(self.images / "1.png")
        self.run_verification()
        self.assertEqual(self.calls, [1])
        self.calls.clear()
        self.context["device"] = "cuda"
        self.run_verification()
        self.assertEqual(self.calls, [0, 1, 2])
        self.calls.clear()
        np.save(self.root / "labels.npy", np.array([3, 4, 5]))
        self.run_verification()
        self.assertEqual(self.calls, [0, 1, 2])

    def test_single_row_diagnostic_never_writes_manifest(self):
        path = self.run_verification(check_rows=[2, 2])
        self.assertEqual(self.calls, [2])
        self.assertEqual(path.name, "image_verification_check.json")
        self.assertFalse((self.root / "image_manifest.json").exists())
        self.calls.clear()
        self.run_verification()
        self.assertEqual(self.calls, [0, 1])
        with self.assertRaisesRegex(ValueError, "zero-based"): self.run_verification(check_rows=[3])

    def test_missing_images_no_face_and_nonfinite_values_reported(self):
        (self.images / "0.png").unlink()

        def bad_extract(image): return None if int(image[0, 0, 0]) == 1 else np.full(3, np.nan)

        with self.assertRaisesRegex(ValueError, "3 image verification failures"): self.run_verification(bad_extract)
        failures = json.loads((self.root / "image_verification_report.json").read_text())["failures"]
        self.assertEqual([row["cosine"] for row in failures], [None, None, None])
        self.assertEqual(failures[1]["error"], "No face found")

    def test_cli_diagnostic_uses_local_images_and_no_manifest(self):
        args = ["piu-prepare-images", "--data-dir", str(self.root), "--device", "cpu", "--check-rows", "1"]
        with patch("sys.argv", args), patch("piu_unlearning.prepare_data.ArcFaceExtractor", return_value=self.extract), patch("piu_unlearning.prepare_data.image_verification_context", return_value=self.context), patch("piu_unlearning.prepare_data.download_dataset", side_effect=AssertionError("Diagnostic must not download")):
            images_main()
        self.assertEqual(self.calls, [1])
        self.assertFalse((self.root / "image_manifest.json").exists())

    def test_alignment_boundary_is_shared_by_verification_and_training(self):
        def extract(image):
            index = int(image[0, 0, 0])
            cosine = 0.988 if index == 1 else 1.0
            return cosine * self.embeddings[index] + np.sqrt(1 - cosine**2) * self.embeddings[(index + 1) % 3]

        manifest = self.run_verification(extract)
        config = WIDConfig(identity_id=0, data_dir=self.root)
        load_image_manifest(config, torch.tensor([1]))
        self.assertAlmostEqual(json.loads(manifest.read_text())["alignment"]["min_cosine"], 0.988, places=5)
        self.context["version"] = "changed"

        def too_low(image):
            index = int(image[0, 0, 0])
            cosine = 0.984 if index == 1 else 1.0
            return cosine * self.embeddings[index] + np.sqrt(1 - cosine**2) * self.embeddings[(index + 1) % 3]

        with self.assertRaisesRegex(ValueError, "1 image verification failures"): self.run_verification(too_low)


if __name__ == "__main__": unittest.main()
