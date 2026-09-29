import contextlib
import copy
import io
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from piu_unlearning.config import UCEConfig, parse_config
from piu_unlearning.data import create_evaluation_conditions
from piu_unlearning.methods import build_method
from piu_unlearning.methods.reference import ReferenceIdentity
from piu_unlearning.methods.uce import build_edit_weights, collect_edit_modules, prepare_edit_embeddings
from piu_unlearning.models.arc2face import GeneratedSamples
from piu_unlearning.visualization import create_summary_visuals
from test_methods import TinyConditioner, make_split


def make_unet(bias=False):
    model = nn.Module()
    model.down_blocks = nn.ModuleList([nn.ModuleDict({
        attention: nn.ModuleDict({projection: nn.Linear(3, 4, bias=bias) for projection in ("to_k", "to_v", "to_q", "to_out")})
        for attention in ("attn1", "attn2")
    }) for _ in range(2)])
    return model


class UCETests(unittest.TestCase):
    def test_defaults_and_cli_overrides(self):
        config = parse_config(["--method", "uce", "--identity-id", "512"])
        self.assertEqual(config, UCEConfig(identity_id=512))
        self.assertEqual((config.erase_scale, config.preserve_scale, config.lamb), (20, 1, 0.7))
        self.assertEqual((config.num_preserve_ids, config.num_forget_combinations, config.normalize_branch_weights), (20, 0, False))
        self.assertNotIn("optimizer", asdict(config))
        self.assertNotIn("training_steps", asdict(config))
        config = parse_config(["--method", "uce", "--identity-id", "512", "--use-anchor-overrides", "--edit-scope", "surgical", "--normalize-branch-weights", "--num-preserve-ids", "0", "--num-forget-combinations", "32", "--erase-scale", "5", "--preserve-scale", "2", "--lamb", "0.9"])
        self.assertIsNotNone(config.anchor_overrides)
        self.assertEqual((config.edit_scope, config.normalize_branch_weights, config.num_preserve_ids, config.num_forget_combinations), ("surgical", True, 0, 32))
        self.assertEqual((config.erase_scale, config.preserve_scale, config.lamb), (5, 2, 0.9))

    def test_invalid_and_gradient_options_rejected(self):
        for arguments in (["--lamb", "0"], ["--erase-scale", "-1"], ["--preserve-scale", "-1"], ["--conditioning-batch-size", "0"], ["--num-preserve-ids", "-1"], ["--num-forget-combinations", "-1"], ["--learning-rate", "1e-5"], ["--training-steps", "1"]):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_config(["--method", "uce", "--identity-id", "0", *arguments])

    def test_only_cross_attention_keys_and_values_selected(self):
        model = make_unet()
        config = UCEConfig(identity_id=0)
        self.assertEqual(len(collect_edit_modules(model, config)), 4)
        selected = collect_edit_modules(model, replace(config, edit_scope="surgical", surgical_layers=("down_blocks.1",)))
        self.assertEqual(set(selected), {"down_blocks.1.attn2.to_k", "down_blocks.1.attn2.to_v"})
        with self.assertRaises(ValueError): collect_edit_modules(model, replace(config, edit_scope="surgical", surgical_layers=("missing",)))

    def test_edit_data_is_seeded_and_training_only(self):
        config = UCEConfig(identity_id=0, device="cpu", num_preserve_ids=2, num_forget_combinations=4)
        split = make_split(config)
        anchor = int(split.retain_train.labels[0])
        reference = ReferenceIdentity(torch.ones(1, 3), "proximity", anchor, 0.2)
        before = torch.random.get_rng_state()
        erase, preserve, ids = prepare_edit_embeddings(split, reference, config)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(len(erase), len(split.forget_train.embeddings) + 4)
        self.assertEqual(len(preserve), 2)
        self.assertIn(anchor, ids.tolist())
        self.assertFalse(torch.isin(ids, split.retain_validation.labels).any())
        torch.testing.assert_close(erase.norm(dim=1), torch.ones(len(erase)))
        for centroid, identity in zip(preserve, ids):
            expected = F.normalize(split.retain_train.embeddings[split.retain_train.labels == identity].mean(dim=0), dim=0)
            torch.testing.assert_close(centroid, expected)
        for first, second in zip((erase, preserve, ids), prepare_edit_embeddings(split, reference, config)):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        changed = replace(split, forget_validation=replace(split.forget_validation, embeddings=torch.full_like(split.forget_validation.embeddings, 999)), retain_validation=replace(split.retain_validation, embeddings=torch.full_like(split.retain_validation.embeddings, -999)))
        for first, second in zip((erase, preserve, ids), prepare_edit_embeddings(changed, reference, config)):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        _, all_preserve, all_ids = prepare_edit_embeddings(split, reference, replace(config, num_preserve_ids=0))
        self.assertEqual(len(all_preserve), len(torch.unique(split.retain_train.labels)))
        self.assertEqual(all_ids.tolist(), torch.unique(split.retain_train.labels).tolist())

    def test_solve_matches_legacy_normal_equations(self):
        torch.manual_seed(7)
        erase, guide, retain = torch.randn(5, 3), torch.randn(1, 3), torch.randn(7, 3)
        for normalize in (False, True):
            for bias in (False, True):
                for preserve in (retain, retain[:0]):
                    with self.subTest(normalize=normalize, bias=bias, preserve=len(preserve)):
                        config = UCEConfig(identity_id=0, normalize_branch_weights=normalize)
                        modules = collect_edit_modules(make_unet(bias), config)
                        actual = build_edit_weights(modules, erase, guide, preserve, config)
                        es = config.erase_scale / (len(erase) if normalize else 1)
                        ps = config.preserve_scale / (max(1, len(preserve)) if normalize else 1)
                        for name, module in modules.items():
                            original = module.weight.detach().clone()
                            numerator, denominator = config.lamb * original, config.lamb * torch.eye(3)
                            for row in erase.split(2):
                                numerator += es * (module(guide).T @ row.sum(dim=0, keepdim=True))
                                denominator += es * (row.T @ row)
                            for row in preserve.split(2):
                                numerator += ps * (module(row).T @ row)
                                denominator += ps * (row.T @ row)
                            expected = torch.linalg.solve(denominator.T, numerator.T).T
                            torch.testing.assert_close(actual[f"{name}.weight"], expected, rtol=2e-5, atol=2e-6)
                            torch.testing.assert_close(module.weight, original, rtol=0, atol=0)

    def test_zero_edit_is_identity(self):
        config = UCEConfig(identity_id=0, erase_scale=0)
        modules = collect_edit_modules(make_unet(), config)
        weights = build_edit_weights(modules, torch.randn(5, 3), torch.randn(1, 3), torch.randn(7, 3), config)
        for name, module in modules.items(): torch.testing.assert_close(weights[f"{name}.weight"], module.weight)

    def test_demo_edit_saves_checkpoint_without_training(self):
        from piu_unlearning.main import unlearn_identity

        with tempfile.TemporaryDirectory() as directory:
            config = UCEConfig(identity_id=0, device="cpu", output_dir=Path(directory), num_samples=2, conditioning_batch_size=2, num_preserve_ids=2)
            split = make_split(config)
            model = SimpleNamespace(unet=make_unet().requires_grad_(False), tokenizer=None, text_encoder=None)
            original = copy.deepcopy(model.unet.state_dict())
            anchor = int(split.retain_train.labels[0])
            conditioner = TinyConditioner()
            with patch("piu_unlearning.methods.reference.select_anchor_embedding", return_value=(torch.tensor([1., 0., 0.]), anchor, 0.2)), patch("piu_unlearning.main.Arc2FaceIdentityConditioner", return_value=conditioner), patch("piu_unlearning.main.train_model", side_effect=AssertionError("UCE must not train")), patch("piu_unlearning.main.copy") as copy_module:
                path = unlearn_identity(model, split, create_evaluation_conditions(split, config), config)
                copy_module.deepcopy.assert_not_called()
            checkpoint = torch.load(path, weights_only=True)
            self.assertEqual(checkpoint["method"], "uce")
            self.assertEqual(path.name, "uce_unet.pt")
            self.assertNotIn("optimizer_state_dict", checkpoint)
            self.assertIsNone(conditioner.weight.grad)
            self.assertFalse((path.parent / "loss_history.jsonl").exists())
            metadata = json.loads((path.parent / "edit.json").read_text())
            self.assertEqual(metadata["preserve_identity_ids"], checkpoint["preserve_identity_ids"])
            self.assertEqual(json.loads((config.output_dir / "reference.json").read_text())["identity_id"], anchor)
            for name, value in model.unet.state_dict().items():
                torch.testing.assert_close(value, checkpoint["unet_state_dict"][name], rtol=0, atol=0)
                if name in metadata["edited_parameters"]: self.assertFalse(torch.equal(value, original[name]))
                else: torch.testing.assert_close(value, original[name], rtol=0, atol=0)

    def test_visualization_needs_no_loss_history(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            config = UCEConfig(identity_id=0, device="cpu", num_samples=1)
            conditions = create_evaluation_conditions(make_split(config), config)
            path = output_dir / "sample.png"
            Image.new("RGB", (16, 16), "white").save(path)
            images = GeneratedSamples([path], [path])
            curves, grid = create_summary_visuals(images, images, conditions, output_dir, "uce")
            self.assertIsNone(curves)
            self.assertTrue(grid.is_file())

    def test_full_demo_orchestration(self):
        from piu_unlearning.evaluation import EvaluationReport, PhaseMetrics, SplitMetrics, SRKMetrics
        from piu_unlearning.main import run_demo

        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            config = UCEConfig(identity_id=0, device="cpu", output_dir=Path(directory), num_samples=1)
            split = make_split(config)
            model = SimpleNamespace(unet=make_unet().requires_grad_(False), tokenizer=None, text_encoder=None)
            image_path = config.output_dir / "sample.png"
            Image.new("RGB", (16, 16), "white").save(image_path)
            images = GeneratedSamples([image_path], [image_path])
            metrics = PhaseMetrics(SplitMetrics(0.5), SplitMetrics(0.7), SRKMetrics(1, 1, 0.99))
            for name, value in {
                "load_prepared_data": (None,) * 4,
                "create_experiment_split": split,
                "load_arc2face": model,
                "Arc2FaceIdentityConditioner": TinyConditioner(),
                "generate_evaluation_samples": images,
                "evaluate_before_after": EvaluationReport(metrics, metrics),
            }.items(): stack.enter_context(patch(f"piu_unlearning.main.{name}", return_value=value))
            stack.enter_context(patch("piu_unlearning.methods.reference.select_anchor_embedding", return_value=(torch.tensor([1., 0., 0.]), int(split.retain_train.labels[0]), 0.2)))
            result = run_demo(config)
            self.assertEqual(result.method, "uce")
            for name in ("config.json", "split.json", "evaluation_conditions.npz", "summary.json", "summary/fixed_conditions.png", "checkpoints/uce_unet.pt"):
                self.assertTrue((config.output_dir / name).is_file(), name)
            self.assertFalse((config.output_dir / "summary/training_curves.png").exists())


if __name__ == "__main__":
    unittest.main()
