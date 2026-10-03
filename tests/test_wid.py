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

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from piu_unlearning.config import WIDConfig, parse_config
from piu_unlearning.data import PairedImageDataset, create_evaluation_conditions, create_experiment_split, load_image_manifest, load_training_image
from piu_unlearning.methods import build_method
from piu_unlearning.methods.wid import WIDContext, prepare_wid_inputs, reconstruct_clean_latents
from piu_unlearning.models.identity_encoder import IdentityEncoder, IRSE50, load_identity_encoder, preprocess_identity_images
from piu_unlearning.dataset.images import write_image_manifest
from piu_unlearning.training.losses import NoisePredictionContext
from piu_unlearning.training.runner import train_model
from piu_unlearning.visualization import write_training_curves
from test_methods import TinyConditioner


class TinyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.8))
        self.config = SimpleNamespace(scaling_factor=0.18215)
        self.decode_calls = 0

    def encode(self, pixels):
        latent = F.interpolate(pixels, size=(2, 2)) * self.scale
        return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: latent))

    def decode(self, latents):
        self.decode_calls += 1
        return SimpleNamespace(sample=F.interpolate(latents * self.scale, size=(8, 8)))


class TinyWIDUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.2, 0.3, 0.4]).reshape(1, 3, 1, 1))
        self.config = SimpleNamespace(in_channels=3, sample_size=2)
        self.calls = []

    def forward(self, latents, timesteps, encoder_hidden_states):
        self.calls.append((latents, timesteps))
        condition = encoder_hidden_states.mean(dim=1).reshape(-1, 3, 1, 1)
        return SimpleNamespace(sample=latents * self.weight + condition * 0.1)


class TinyRecognizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[1., 0.1, 0.2], [0.1, 1., -0.2], [-0.2, 0.1, 1.]]))

    def forward(self, images): return images.mean(dim=(2, 3)) @ self.weight


def make_wid_context():
    from diffusers import DDPMScheduler

    model = TinyWIDUNet()
    teacher = copy.deepcopy(model).eval().requires_grad_(False)
    teacher.weight.add_(0.1)
    noise = NoisePredictionContext(model, teacher, DDPMScheduler(num_train_timesteps=10), TinyConditioner(), torch.tensor([[[0., 1., 0.]]]), torch.device("cpu"))
    return WIDContext(noise, TinyVAE().eval().requires_grad_(False), IdentityEncoder(TinyRecognizer(), "bgr"))


def make_image_fixture(root):
    data_dir, image_root = root / "data", root / "images"
    data_dir.mkdir()
    image_root.mkdir()
    embeddings = F.normalize(torch.randn(24, 3, generator=torch.Generator().manual_seed(42)), dim=1)
    labels = torch.arange(6).repeat_interleave(4)
    names = [f"{index:04d}.png" for index in range(24)]
    for index, name in enumerate(names): Image.new("RGB", (20, 24), (index, 100 + index, 200 - index)).save(image_root / name)
    np.save(data_dir / "embeddings.npy", embeddings.numpy())
    np.save(data_dir / "labels.npy", labels.numpy())
    paths_file = root / "image_paths.txt"
    paths_file.write_text("\n".join(names) + "\n")
    manifest = write_image_manifest(data_dir, paths_file, image_root, "test-revision")
    config = WIDConfig(identity_id=0, data_dir=data_dir, output_dir=root / "run", device="cpu", identity_checkpoint=root / "recognizer.pt", num_samples=1, batch_size=1, gradient_accumulation_steps=1, training_steps=1, evaluation_every=0)
    config.identity_checkpoint.write_bytes(b"mock checkpoint")
    centroids = F.normalize(embeddings.reshape(6, 4, 3).mean(dim=1), dim=1)
    split = create_experiment_split(embeddings, labels, centroids, torch.arange(6), config)
    return config, split, manifest


class WIDTests(unittest.TestCase):
    def test_defaults_and_cli(self):
        config = parse_config(["--method", "wid", "--identity-id", "512"])
        self.assertEqual(config, WIDConfig(identity_id=512))
        self.assertEqual((config.learning_rate, config.training_steps, config.identity_loss_weight), (5e-6, 100, 0.1))
        self.assertEqual(config.weight_decay, 0)
        self.assertEqual((config.batch_size, config.gradient_accumulation_steps, config.max_grad_norm), (4, 16, 1))
        self.assertEqual((config.train_mode, config.optimizer, config.preservation_weight), ("full", "adamw", 0))
        self.assertNotIn("forget_sampling", asdict(config))
        self.assertNotIn("negative_guidance_scale", asdict(config))
        with contextlib.redirect_stderr(io.StringIO()):
            for args in (["--forget-sampling", "centroid"], ["--negative-guidance-scale", "1"], ["--identity-loss-weight", "-1"], ["--max-grad-norm", "0"]):
                with self.subTest(args=args), self.assertRaises(SystemExit): parse_config(["--method", "wid", "--identity-id", "512", *args])

    def test_preprocessing_preserves_gradient_and_channels(self):
        pixels = torch.tensor([[[[-1.]], [[0.]], [[1.]]]], requires_grad=True)
        output = preprocess_identity_images(pixels, "bgr")
        torch.testing.assert_close(output[0, :, 0, 0], torch.tensor([1., 0., -1.]))
        self.assertEqual(output.shape, (1, 3, 112, 112))
        output.sum().backward()
        self.assertTrue((pixels.grad != 0).all())
        encoder = IdentityEncoder(TinyRecognizer(), "rgb")
        self.assertFalse(encoder.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in encoder.parameters()))

    def test_checkpoint_loading_is_strict_and_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            model = TinyRecognizer()
            torch.save({"state_dict": {"module." + key: value for key, value in model.state_dict().items()}}, path)
            with patch("piu_unlearning.models.identity_encoder.IRSE50", return_value=TinyRecognizer()):
                loaded = load_identity_encoder(path, "rgb")
            torch.testing.assert_close(loaded.backbone.weight, model.weight)
            self.assertFalse(loaded.backbone.weight.requires_grad)
            torch.save({"wrong": torch.ones(1)}, path)
            with patch("piu_unlearning.models.identity_encoder.IRSE50", return_value=TinyRecognizer()), self.assertRaises(RuntimeError): load_identity_encoder(path, "rgb")

    def test_real_backbone_shape_and_input_gradients(self):
        model = IRSE50().eval().requires_grad_(False)
        images = torch.randn(1, 3, 112, 112, requires_grad=True)
        embedding = model(images)
        self.assertEqual(embedding.shape, (1, 512))
        torch.testing.assert_close(embedding.norm(dim=-1), torch.ones(1))
        embedding[0, 0].backward()
        self.assertGreater(float(images.grad.abs().sum()), 0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_manifest_dataset_pairing_and_lazy_image_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            config, split, manifest = make_image_fixture(Path(directory))
            paths, metadata = load_image_manifest(config)
            dataset = PairedImageDataset(split.forget_train, paths)
            self.assertEqual(len(dataset), len(split.forget_train.indices))
            batch = dataset[0]
            self.assertEqual(batch["pixel_values"].shape, (3, 512, 512))
            self.assertEqual(batch["pixel_values"].dtype, torch.float32)
            self.assertGreaterEqual(float(batch["pixel_values"].min()), -1)
            self.assertLessEqual(float(batch["pixel_values"].max()), 1)
            torch.testing.assert_close(batch["face_embs"], split.forget_train.embeddings[0])
            index = split.forget_train.indices[0].item()
            self.assertAlmostEqual(float(batch["pixel_values"][0, 0, 0]), index / 127.5 - 1, places=6)
            self.assertFalse(set(dataset.paths) & {paths[index] for index in split.forget_validation.indices.tolist()})
            dataset.paths[0].write_bytes(b"changed")
            load_image_manifest(config)
            with self.assertRaises(OSError): dataset[0]
            np.save(config.labels_path, np.zeros(24, dtype=np.int64))
            with self.assertRaisesRegex(ValueError, "labels.npy"): load_image_manifest(config)

    def test_target_uses_each_original_forget_training_image(self):
        with tempfile.TemporaryDirectory() as directory:
            config, split, _ = make_image_fixture(Path(directory))
            anchor_id = int(split.retain_train.labels[0])
            encoder = IdentityEncoder(TinyRecognizer(), "bgr")
            with patch("piu_unlearning.methods.reference.select_anchor_embedding", return_value=(torch.tensor([1., 0., 0.]), anchor_id, 0.2)), patch("piu_unlearning.methods.wid.load_identity_encoder", return_value=encoder):
                inputs = prepare_wid_inputs(split, config)
            expected_indices = split.forget_train.indices
            self.assertEqual(inputs.metadata["target_mode"], "original_images")
            self.assertEqual(inputs.metadata["forget_indices"], expected_indices.tolist())
            root = Path(directory) / "images"
            with torch.no_grad(): target = encoder(torch.stack([load_training_image(root / f"{index:04d}.png") for index in expected_indices]))
            torch.testing.assert_close(inputs.identity_target, target)
            self.assertFalse(inputs.identity_target.requires_grad)
            for index in range(len(inputs.dataset)):
                torch.testing.assert_close(inputs.dataset[index]["identity_target"], target[index])
            self.assertEqual(len({tuple(row.tolist()) for row in target}), len(target))

    def test_reconstruction_equation(self):
        context = make_wid_context()
        clean, noise = torch.randn(2, 3, 2, 2), torch.randn(2, 3, 2, 2)
        timesteps = torch.tensor([0, 9])
        noisy = context.noise.scheduler.add_noise(clean, noise, timesteps)
        torch.testing.assert_close(reconstruct_clean_latents(noisy, noise, timesteps, context.noise.scheduler), clean)

    def test_disabled_identity_loss_skips_encoder_and_download(self):
        with tempfile.TemporaryDirectory() as directory:
            config, split, _ = make_image_fixture(Path(directory))
            anchor_id = int(split.retain_train.labels[0])
            reference = F.normalize(split.retain_train.embeddings[split.retain_train.labels == anchor_id].mean(dim=0), dim=0)
            with patch("piu_unlearning.methods.reference.select_anchor_embedding", return_value=(reference, anchor_id, 0.2)), patch("piu_unlearning.methods.wid.load_identity_encoder", side_effect=AssertionError("Disabled identity branch must not load weights")):
                inputs = prepare_wid_inputs(split, replace(config, identity_loss_weight=0, identity_checkpoint=None))
                self.assertIsNone(inputs.encoder)
                self.assertIsNone(inputs.identity_target)
                with patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("Disabled identity branch must not download weights")):
                    prepare_wid_inputs(split, replace(config, identity_loss_weight=0, identity_checkpoint=None))

    def test_loss_equation_shared_inputs_and_identity_only_gradients(self):
        for model_weight in (0.0, 1.0):
            context = make_wid_context()
            config = WIDConfig(identity_id=0, model_loss_weight=model_weight)
            batch = {"face_embs": torch.tensor([[0.2, 0.3, 0.7]]), "pixel_values": torch.linspace(-1, 1, 192).reshape(1, 3, 8, 8), "identity_target": torch.tensor([[0., 1., 0.]])}
            with patch("torch.randint", return_value=torch.tensor([4])), patch("torch.randn_like", side_effect=lambda value: torch.full_like(value, 0.5)):
                result = build_method(config).compute_loss(batch, None, context)
            student_call, teacher_call = context.noise.model.calls[0], context.noise.teacher.calls[0]
            self.assertIs(student_call[0], teacher_call[0])
            self.assertIs(student_call[1], teacher_call[1])
            clean = F.interpolate(batch["pixel_values"], size=(2, 2)) * context.vae.scale * context.vae.config.scaling_factor
            expected_noisy = context.noise.scheduler.add_noise(clean, torch.full_like(clean, 0.5), torch.tensor([4]))
            torch.testing.assert_close(student_call[0], expected_noisy)
            prediction = expected_noisy * context.noise.model.weight + (batch["face_embs"] * 1.1).reshape(1, 3, 1, 1) * 0.1
            teacher = expected_noisy * context.noise.teacher.weight + context.noise.reference_conditioning.reshape(1, 3, 1, 1) * 0.1
            alpha = context.noise.scheduler.alphas_cumprod[4]
            z0 = (expected_noisy - (1 - alpha).sqrt() * prediction) / alpha.sqrt()
            decoded = F.interpolate(z0 / context.vae.config.scaling_factor * context.vae.scale, size=(8, 8))
            ids = context.identity_encoder(decoded)
            expected_id_loss = (ids - batch["identity_target"]).square().mean()
            expected = model_weight * F.mse_loss(prediction, teacher) + config.identity_loss_weight * expected_id_loss
            torch.testing.assert_close(result.loss, expected)
            torch.testing.assert_close(result.metrics["identity_loss"], expected_id_loss)
            result.loss.backward()
            self.assertGreater(float(context.noise.model.weight.grad.abs().sum()), 0)
            for module in (context.vae, context.identity_encoder, context.noise.teacher, context.noise.conditioner):
                self.assertTrue(all(parameter.grad is None for parameter in module.parameters()))

    def test_disabled_identity_and_optional_preservation(self):
        context = replace(make_wid_context(), identity_encoder=None)
        config = WIDConfig(identity_id=0, identity_loss_weight=0, preservation_weight=2)
        batch = {"face_embs": torch.ones(1, 3), "pixel_values": torch.randn(1, 3, 8, 8)}
        result = build_method(config).compute_loss(batch, torch.ones(1, 3), context)
        self.assertEqual(context.vae.decode_calls, 0)
        self.assertEqual(result.metrics["identity_loss"].item(), 0)
        self.assertGreater(result.metrics["preserve_loss"].item(), 0)
        result.loss.backward()

    def test_runner_clips_once_after_accumulation_and_saves_wid(self):
        with tempfile.TemporaryDirectory() as directory:
            context = make_wid_context()
            config = WIDConfig(identity_id=0, output_dir=Path(directory), training_steps=1, gradient_accumulation_steps=2, evaluation_every=0)
            batch = {"face_embs": torch.ones(1, 3), "pixel_values": torch.randn(1, 3, 8, 8), "identity_target": torch.tensor([[0., 1., 0.]])}
            method = build_method(config)
            with patch("torch.nn.utils.clip_grad_norm_", wraps=torch.nn.utils.clip_grad_norm_) as clip:
                path = train_model(context.noise.model, lambda forget, retain: method.compute_loss(forget, retain, context), [batch, batch], None, config)
            self.assertEqual(clip.call_count, 1)
            self.assertEqual(clip.call_args.args[1], 1.0)
            checkpoint = torch.load(path, weights_only=True)
            self.assertEqual(checkpoint["method"], "wid")
            self.assertNotIn("negative_guidance_scale", checkpoint)
            history = json.loads((path.parent / "loss_history.jsonl").read_text())
            self.assertTrue({"model_loss", "identity_loss", "model_term", "identity_term"} <= history.keys())
            write_training_curves(path.parent / "loss_history.jsonl", path.parent / "ism_history.jsonl", Path(directory) / "curves.png", "wid")
            self.assertTrue((Path(directory) / "curves.png").exists())

    def test_unlearn_integration_and_dependency_preflight(self):
        from diffusers import DDPMScheduler
        from piu_unlearning.main import run_demo, unlearn_identity

        with tempfile.TemporaryDirectory() as directory:
            config, split, _ = make_image_fixture(Path(directory))
            anchor_id = int(split.retain_train.labels[0])
            model = SimpleNamespace(unet=TinyWIDUNet(), vae=TinyVAE(), scheduler=DDPMScheduler(num_train_timesteps=10), tokenizer=None, text_encoder=None)
            with patch("piu_unlearning.methods.reference.select_anchor_embedding", return_value=(torch.tensor([1., 0., 0.]), anchor_id, 0.2)), patch("piu_unlearning.methods.wid.load_identity_encoder", return_value=IdentityEncoder(TinyRecognizer(), "bgr")), patch("piu_unlearning.main.Arc2FaceIdentityConditioner", return_value=TinyConditioner()):
                checkpoint = unlearn_identity(model, split, create_evaluation_conditions(split, config), config)
            self.assertTrue(checkpoint.is_file())
            self.assertEqual(json.loads((config.output_dir / "wid_identity.json").read_text())["target_mode"], "original_images")
            self.assertTrue((config.output_dir / "identity_target.pt").is_file())
            bad = replace(config, image_manifest=Path(directory) / "missing.json")
            with patch("piu_unlearning.methods.reference.select_anchor_embedding", return_value=(torch.tensor([1., 0., 0.]), anchor_id, 0.2)), patch("piu_unlearning.main.load_arc2face") as load_model:
                with self.assertRaises(FileNotFoundError): run_demo(bad, split=split)
                load_model.assert_not_called()

    def test_full_wid_demo_outputs(self):
        from diffusers import DDPMScheduler
        from piu_unlearning.evaluation import EvaluationReport, PhaseMetrics, SplitMetrics, SRKMetrics
        from piu_unlearning.main import run_demo
        from piu_unlearning.models.arc2face import GeneratedSamples

        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            config, split, _ = make_image_fixture(Path(directory))
            model = SimpleNamespace(unet=TinyWIDUNet(), vae=TinyVAE(), scheduler=DDPMScheduler(num_train_timesteps=10), tokenizer=None, text_encoder=None)
            image = Path(directory) / "result.png"
            Image.new("RGB", (16, 16), "white").save(image)
            metrics = PhaseMetrics(SplitMetrics(0.4), SplitMetrics(0.7), SRKMetrics(0.2, 0.9, 4.28))
            for name, value in {"load_arc2face": model, "Arc2FaceIdentityConditioner": TinyConditioner(), "generate_evaluation_samples": GeneratedSamples([image], [image]), "evaluate_before_after": EvaluationReport(metrics, metrics)}.items():
                stack.enter_context(patch(f"piu_unlearning.main.{name}", return_value=value))
            stack.enter_context(patch("piu_unlearning.methods.reference.select_anchor_embedding", return_value=(torch.tensor([1., 0., 0.]), int(split.retain_train.labels[0]), 0.2)))
            stack.enter_context(patch("piu_unlearning.methods.wid.load_identity_encoder", return_value=IdentityEncoder(TinyRecognizer(), "bgr")))
            result = run_demo(config, split=split)
            self.assertEqual(result.method, "wid")
            for path in ("config.json", "split.json", "reference.json", "wid_identity.json", "identity_target.pt", "summary.json", "checkpoints/wid_unet.pt", "summary/training_curves.png", "summary/fixed_conditions.png"):
                self.assertTrue((config.output_dir / path).is_file(), path)


if __name__ == "__main__":
    unittest.main()
