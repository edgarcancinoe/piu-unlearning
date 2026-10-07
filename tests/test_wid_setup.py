import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from piu_unlearning.dataset.hub import sha256
from piu_unlearning.methods.wid import prepare_wid_inputs
from piu_unlearning.models.identity_encoder import IRSE50_FILENAME, IRSE50_REPO, IRSE50_REVISION, resolve_identity_checkpoint
from test_wid import TinyRecognizer, make_image_fixture


class WIDSetupTests(unittest.TestCase):
    def test_default_download_is_pinned(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "weights.pt"
            checkpoint.write_bytes(b"weights")
            with patch("huggingface_hub.hf_hub_download", return_value=str(checkpoint)) as download:
                self.assertEqual(resolve_identity_checkpoint(None), checkpoint.resolve())
            download.assert_called_once_with(IRSE50_REPO, IRSE50_FILENAME, revision=IRSE50_REVISION)

    def test_local_override_skips_download_and_missing_override_fails(self):
        with tempfile.TemporaryDirectory() as directory, patch("huggingface_hub.hf_hub_download") as download:
            checkpoint = Path(directory) / "weights.pt"
            checkpoint.write_bytes(b"weights")
            self.assertEqual(resolve_identity_checkpoint(checkpoint), checkpoint.resolve())
            with self.assertRaisesRegex(FileNotFoundError, "IR-SE50 checkpoint not found"):
                resolve_identity_checkpoint(Path(directory) / "missing.pt")
            download.assert_not_called()

    def test_default_preflight_loads_download_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            config, split, _ = make_image_fixture(Path(directory))
            checkpoint = Path(directory) / "downloaded.pt"
            torch.save(TinyRecognizer().state_dict(), checkpoint)
            anchor_id = int(split.retain_train.labels[0])
            with patch("piu_unlearning.methods.reference.select_anchor_embedding", return_value=(torch.tensor([1., 0., 0.]), anchor_id, 0.2)), patch("huggingface_hub.hf_hub_download", return_value=str(checkpoint)) as download, patch("piu_unlearning.models.identity_encoder.IRSE50", return_value=TinyRecognizer()):
                inputs = prepare_wid_inputs(split, replace(config, identity_checkpoint=None))
            download.assert_called_once()
            self.assertEqual(inputs.metadata["checkpoint"], str(checkpoint.resolve()))
            self.assertEqual(inputs.metadata["checkpoint_sha256"], sha256(checkpoint))
            self.assertEqual(len(inputs.identity_target), len(split.forget_train.indices))
            self.assertFalse(inputs.identity_target.requires_grad)
            self.assertTrue(all(not parameter.requires_grad for parameter in inputs.encoder.parameters()))

if __name__ == "__main__": unittest.main()
