import json
import os
import shlex
import subprocess
import sys
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

    def test_helper_accepts_default_and_local_checkpoint(self):
        helper = Path(__file__).resolve().parents[1] / "baselines/run_wid.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            for name in ("embeddings.npy", "centroids.npy", "centroid_labels.npy", "labels.npy"):
                (data / name).touch()
            checkpoint = root / "custom weights.pt"
            checkpoint.touch()
            (root / "piu-prepare-data").write_text("#!/bin/sh\nexit 99\n")
            (root / "piu-prepare-data").chmod(0o755)
            (root / "python").write_text(f"#!{sys.executable}\nimport json, os, sys\nif sys.argv[1] == '-c': code = sys.argv[2]; sys.argv = ['-c', *sys.argv[3:]]; exec(code)\nelse: open(os.environ['WID_TEST_ARGS'], 'w').write(json.dumps(sys.argv[1:]))\n")
            (root / "python").chmod(0o755)
            captured = root / "args.json"
            env = {**os.environ, "PATH": str(root) + os.pathsep + os.environ["PATH"], "WID_TEST_ARGS": str(captured)}
            for weights in ([], [str(checkpoint)]):
                result = subprocess.run(["bash", str(helper), "512", *weights, "--data-dir", str(data), "--use-anchor-overrides"], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                args = json.loads(captured.read_text())
                self.assertIn("--use-anchor-overrides", args)
                wid_args = next(arg.split("=", 1)[1] for arg in args if arg.startswith("--wid-args="))
                self.assertEqual(shlex.split(wid_args), ["--identity-checkpoint", str(checkpoint)] if weights else [])


if __name__ == "__main__": unittest.main()
