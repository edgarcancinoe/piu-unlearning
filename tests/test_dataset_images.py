import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from piu_unlearning.dataset.images import materialize_images, prepare_dataset_images, write_image_manifest
from piu_unlearning.config import PAPER_LABELS_FILE


class DatasetImageTests(unittest.TestCase):
    def test_downloaded_images_keep_dataset_row_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ["first.png", "second.png"]
            source = root / "source"
            source.mkdir()
            for index, name in enumerate(names):
                Image.new("RGB", (4, 4), (index, 20, 30)).save(source / name)
            shard = root / "images.parquet"
            pq.write_table(pa.table({
                "image": [{"bytes": (source / name).read_bytes()} for name in names],
                "file_name": names,
            }), shard)
            target = root / "images"
            self.assertEqual(materialize_images([shard], target), names)
            with self.assertRaisesRegex(ValueError, "row order"):
                materialize_images([shard], target, names[::-1])

    def test_manifest_records_rows_without_extracting_embeddings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, images = root / "data", root / "images"
            data.mkdir()
            images.mkdir()
            names = ["0.png", "1.png"]
            for index, name in enumerate(names):
                Image.new("RGB", (4, 4), (index, 40, 50)).save(images / name)
            np.save(data / "embeddings.npy", np.eye(2, dtype=np.float32))
            np.save(data / PAPER_LABELS_FILE, np.array([0, 1]))
            mapping = data / "file_names.txt"
            mapping.write_text("\n".join(names) + "\n")
            result = write_image_manifest(data, mapping, images, "dataset-revision")
            manifest = json.loads(result.read_text())
            self.assertEqual(manifest["version"], 2)
            self.assertEqual(manifest["source"]["row_order"], "dataset_parquet")
            self.assertEqual([row["path"] for row in manifest["rows"]], names)
            (images / names[0]).unlink()
            with patch("piu_unlearning.dataset.images.download_dataset", side_effect=AssertionError("unexpected download")):
                self.assertEqual(prepare_dataset_images(data, images), result)


if __name__ == "__main__":
    unittest.main()
