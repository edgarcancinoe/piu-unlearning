import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch
from PIL import Image

from piu_unlearning.evaluation import evaluate_phase, extract_face_embeddings
from piu_unlearning.models.arc2face import GeneratedSamples


class MissingFaceTests(unittest.TestCase):
    def test_missing_faces_keep_alignment_and_count_as_unrecognized(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.png"
            Image.new("RGB", (8, 8)).save(path)
            conditions = SimpleNamespace(forget_labels=torch.tensor([0, 1]), retain_labels=torch.tensor([0, 1]))
            split = SimpleNamespace(centroids=torch.eye(2), centroid_labels=torch.tensor([0, 1]))
            for detected in (np.array([0., 1.], dtype=np.float32), None):
                with self.subTest(all_missing=detected is None):
                    extractor = Mock(side_effect=[None, detected, None, detected])
                    with self.assertLogs(level="WARNING") as logs:
                        metrics = evaluate_phase(GeneratedSamples([path, path], [path, path]), conditions, split, extractor, "test")
                    expected = 0.0 if detected is None else 0.5
                    self.assertEqual(metrics.forget.ism, expected)
                    self.assertEqual(metrics.retain.ism, expected)
                    self.assertEqual(metrics.srk.forget_accuracy, expected)
                    self.assertEqual(metrics.srk.retain_accuracy, expected)
                    self.assertEqual(len(logs.output), 4 if detected is None else 2)

    def test_unrelated_extractor_errors_propagate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.png"
            Image.new("RGB", (8, 8)).save(path)
            with self.assertRaisesRegex(RuntimeError, "extractor failure"):
                extract_face_embeddings([path], Mock(side_effect=RuntimeError("extractor failure")), "test")
