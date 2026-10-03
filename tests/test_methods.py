import contextlib
import copy
import io
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from piu_unlearning.config import PIUConfig, parse_config
from piu_unlearning.data import create_embedding_loaders, create_evaluation_conditions, create_experiment_split
from piu_unlearning.main import baseline_metadata, check_baseline, run_demo
from piu_unlearning.methods import build_method
from piu_unlearning.models.arc2face import select_trainable_layers
from piu_unlearning.training.losses import GuidedNoiseLoss, LossOutput, NoisePredictionContext
from piu_unlearning.training.runner import train_model


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.6))
        self.config = SimpleNamespace(in_channels=1, sample_size=2)
        self.calls = 0

    def forward(self, latents, timesteps, encoder_hidden_states):
        self.calls += 1
        conditioning = encoder_hidden_states.mean(dim=(1, 2)).reshape(-1, 1, 1, 1)
        return SimpleNamespace(sample=self.weight * (latents + conditioning) + timesteps.reshape(-1, 1, 1, 1) / 100)


class TinyConditioner(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.1))

    def encode(self, embeddings):
        return embeddings.unsqueeze(1) * self.weight


class TinyScheduler:
    config = SimpleNamespace(num_train_timesteps=10)

    def add_noise(self, latents, noise, timesteps):
        return latents + noise * (timesteps.reshape(-1, 1, 1, 1) + 1) / 10


def make_context():
    model = TinyUNet()
    teacher = copy.deepcopy(model).requires_grad_(False)
    teacher.weight.fill_(0.9)
    return NoisePredictionContext(model, teacher, TinyScheduler(), TinyConditioner(), torch.ones(1, 1, 3), torch.device("cpu"))


def make_split(config):
    embeddings = F.normalize(torch.arange(1, 73).reshape(24, 3).float(), dim=1)
    labels = torch.arange(6).repeat_interleave(4)
    centroids = F.normalize(embeddings.reshape(6, 4, 3).mean(dim=1), dim=1)
    return create_experiment_split(embeddings, labels, centroids, torch.arange(6), config)


class ConfigurationTests(unittest.TestCase):
    def test_existing_piu_command_keeps_defaults(self):
        config = parse_config(["--identity-id", "512", "--use-anchor-overrides"])
        self.assertIsInstance(config, PIUConfig)
        self.assertEqual((config.train_mode, config.preservation_weight, config.negative_guidance_scale), ("surgical", 10, 1))
        self.assertEqual((config.batch_size, config.gradient_accumulation_steps, config.training_steps), (16, 2, 400))
        self.assertEqual((config.optimizer, config.learning_rate, config.weight_decay), ("adamw", 1e-4, 0.01))
        self.assertEqual(config.output_dir, Path("outputs/demo"))
        self.assertIsNotNone(config.anchor_overrides)


    def test_invalid_options_fail_before_training(self):
        for arguments in (["--method", "siss", "--use-anchor-overrides"], ["--training-steps", "0"], ["--gradient-accumulation-steps", "0"], ["--evaluation-every", "-1"], ["--preservation-weight", "-1"]):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_config(["--identity-id", "512", *arguments])


class BaselineTests(unittest.TestCase):
    def test_reuse_requires_matching_conditions_and_generation_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            config = PIUConfig(identity_id=0, output_dir=Path(directory), num_samples=1)
            split = make_split(config)
            conditions = create_evaluation_conditions(split, config)
            before = config.output_dir / "before"
            before.mkdir()
            with patch("piu_unlearning.main.load_arc2face") as load_model:
                with self.assertRaisesRegex(ValueError, "do not match"): run_demo(replace(config, reuse_baseline=True), split=split)
                load_model.assert_not_called()
            (before / "baseline.json").write_text(json.dumps(baseline_metadata(config, conditions)))
            check_baseline(config, conditions, before)
            for changed in (replace(config, seed=1), replace(config, guidance_scale=4), replace(config, base_model_revision="other")):
                with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "do not match"):
                    check_baseline(changed, conditions, before)
            with self.assertRaisesRegex(ValueError, "do not match"):
                check_baseline(config, replace(conditions, forget_embeddings=conditions.forget_embeddings + 0.01), before)


class MethodTests(unittest.TestCase):

    def test_piu_uses_selected_anchor(self):
        config = PIUConfig(identity_id=0, device="cpu")
        split = make_split(config)
        with patch("piu_unlearning.methods.reference.select_anchor_embedding", return_value=(torch.ones(3), 4, 0.2)):
            reference = build_method(config).prepare_reference(split, config)
        self.assertEqual((reference.mode, reference.identity_id, reference.similarity), ("proximity", 4, 0.2))
        self.assertEqual(reference.embedding.shape, (1, 3))

    def test_loss_and_gradient_match_teacher_target_equation(self):
        for preservation in (0.0, 10.0):
            with self.subTest(preservation=preservation):
                context = make_context()
                forget = torch.tensor([[0.2, 0.4, 0.6], [0.1, 0.3, 0.5]])
                retain = forget + 0.5
                latents = torch.arange(8).reshape(2, 1, 2, 2).float() / 10
                timesteps = torch.tensor([2, 7])
                with patch("piu_unlearning.training.losses.sample_noisy_latents", return_value=(latents, timesteps)):
                    result = GuidedNoiseLoss(preservation, 1.5).compute_loss(forget, retain if preservation else None, context)
                reference = context.reference_conditioning.expand(2, -1, -1).mean(dim=(1, 2)).reshape(-1, 1, 1, 1)
                forget_condition = (forget * 1.1).mean(dim=1).reshape(-1, 1, 1, 1)
                timestep_term = timesteps.reshape(-1, 1, 1, 1) / 100
                anchor_noise = 0.9 * (latents + reference) + timestep_term
                forget_noise = 0.9 * (latents + forget_condition) + timestep_term
                weight = context.model.weight.detach().clone().requires_grad_()
                prediction = weight * (latents + forget_condition) + timestep_term
                expected = F.mse_loss(prediction, anchor_noise - 1.5 * (forget_noise - anchor_noise))
                if preservation:
                    retain_condition = (retain * 1.1).mean(dim=1).reshape(-1, 1, 1, 1)
                    expected += preservation * F.mse_loss(weight * (latents + retain_condition), 0.9 * (latents + retain_condition))
                torch.testing.assert_close(result.loss, expected)
                result.loss.backward()
                expected.backward()
                torch.testing.assert_close(context.model.weight.grad, weight.grad)
                self.assertIsNone(context.teacher.weight.grad)
                self.assertIsNone(context.conditioner.weight.grad)
                self.assertEqual(context.teacher.calls, 3 if preservation else 2)
                self.assertEqual(context.model.calls, 2 if preservation else 1)
                if not preservation: self.assertEqual(result.metrics["preserve_loss"].item(), 0)

    def test_layer_modes(self):
        model = nn.Module()
        model.down_blocks = nn.ModuleList([nn.ModuleDict({"attn2": nn.Linear(2, 2), "attn1": nn.Linear(2, 2)}) for _ in range(3)])
        model.output = nn.Linear(2, 2)
        for mode in ("full", "x", "surgical"):
            select_trainable_layers(model, ("down_blocks.2",), mode)
            expected = [name for name, _ in model.named_parameters() if mode == "full" or ("attn2" in name and (mode == "x" or "down_blocks.2" in name))]
            self.assertEqual([name for name, parameter in model.named_parameters() if parameter.requires_grad], expected)
        with self.assertRaises(ValueError): select_trainable_layers(model, ("missing",), "surgical")


class TrainingTests(unittest.TestCase):
    def test_optimizer_selection_matches_torch(self):
        for name, optimizer_type in (("adam", torch.optim.Adam), ("adamw", torch.optim.AdamW)):
            with self.subTest(optimizer=name), tempfile.TemporaryDirectory() as directory:
                config = PIUConfig(identity_id=0, output_dir=Path(directory), optimizer=name, learning_rate=0.01, weight_decay=0.2, training_steps=2, evaluation_every=0)
                model = nn.Linear(1, 1, bias=False)
                with torch.no_grad(): model.weight.fill_(1)
                reference = copy.deepcopy(model)
                samples = torch.ones(1, 1)

                def loss(batch, _retain):
                    value = model(batch).square().mean()
                    return LossOutput(value, {"forget_loss": value, "preserve_loss": value.new_zeros(())})

                checkpoint_path = train_model(model, loss, DataLoader(samples), None, config)
                optimizer = optimizer_type(reference.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
                for _ in range(config.training_steps):
                    optimizer.zero_grad(set_to_none=True)
                    reference(samples).square().mean().backward()
                    optimizer.step()
                torch.testing.assert_close(model.weight, reference.weight, rtol=0, atol=0)
                checkpoint = torch.load(checkpoint_path, weights_only=True)
                self.assertEqual(checkpoint["config"]["optimizer"], name)


    def test_sampling_modes_and_optional_retain_loader(self):
        config = PIUConfig(identity_id=0, device="cpu", batch_size=2, preservation_weight=0)
        split = make_split(config)
        for mode in ("dirichlet", "centroid", "individual"):
            forget_loader, retain_loader = create_embedding_loaders(split, replace(config, forget_sampling=mode))
            batch = next(iter(forget_loader))
            self.assertIsNone(retain_loader)
            torch.testing.assert_close(batch.norm(dim=1), torch.ones(2))
            if mode == "centroid":
                expected = F.normalize(split.forget_train.embeddings.mean(dim=0), dim=0)
                torch.testing.assert_close(batch, expected.expand_as(batch))
            if mode == "individual":
                self.assertTrue(all(any(torch.equal(row, sample) for sample in split.forget_train.embeddings) for row in batch))
        _, retain_loader = create_embedding_loaders(split, replace(config, preservation_weight=1))
        self.assertIsNotNone(retain_loader)

    def test_runner_checkpoints_and_evaluation_for_piu(self):
        for config_type in (PIUConfig,):
            with self.subTest(method=config_type.__name__), tempfile.TemporaryDirectory() as directory:
                config = config_type(identity_id=0, device="cpu", output_dir=Path(directory), training_steps=2, gradient_accumulation_steps=2, batch_size=2, evaluation_every=1)
                context = make_context()
                initial = context.model.weight.detach().clone()
                forget_loader, retain_loader = create_embedding_loaders(make_split(config), config)
                objective = build_method(config)
                calls = []

                def evaluate(step):
                    calls.append(step)
                    context.model.eval()
                    return 0.25, 0.75

                checkpoint_path = train_model(context.model, partial(objective.compute_loss, context=context), forget_loader, retain_loader, config, evaluate)
                checkpoint = torch.load(checkpoint_path, weights_only=True)
                self.assertEqual(checkpoint_path.name, f"{config.method}_unet.pt")
                self.assertEqual(checkpoint["config"]["method"], config.method)
                self.assertEqual(checkpoint["trainable_parameters"], ["weight"])
                self.assertFalse(torch.equal(initial, context.model.weight))
                self.assertEqual(calls, [1, 2])
                self.assertTrue(context.model.training)
                loss_history = [json.loads(line) for line in (checkpoint_path.parent / "loss_history.jsonl").read_text().splitlines()]
                self.assertEqual([row["step"] for row in loss_history], [1, 2])
                self.assertTrue(all({"loss", "forget_loss", "preserve_loss"} <= row.keys() for row in loss_history))

    def test_gradient_accumulation_matches_full_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            config = PIUConfig(identity_id=0, output_dir=Path(directory), training_steps=1, gradient_accumulation_steps=2, evaluation_every=0)
            model = nn.Linear(1, 1, bias=False)
            reference = copy.deepcopy(model)
            samples = torch.tensor([[1.0], [3.0]])

            def loss(batch, _retain):
                value = model(batch).square().mean()
                return LossOutput(value, {"forget_loss": value, "preserve_loss": value.new_zeros(())})

            train_model(model, loss, DataLoader(samples, batch_size=1), None, config)
            optimizer = torch.optim.AdamW(reference.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            reference(samples).square().mean().backward()
            optimizer.step()
            torch.testing.assert_close(model.weight, reference.weight, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
