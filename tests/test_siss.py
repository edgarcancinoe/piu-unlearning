import contextlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from piu_unlearning.config import SISSConfig, parse_config
from piu_unlearning.data import EmbeddingPartition, create_training_loaders, create_evaluation_conditions
from piu_unlearning.main import unlearn_identity, run_demo
from piu_unlearning.methods import build_method
from piu_unlearning.methods.siss import SISSContext, SISSInputs, combine_gradients, diffusion_loss, prepare_siss_inputs
from piu_unlearning.training.runner import train_model
from test_methods import TinyUNet, TinyConditioner, make_split


class TinyVAE(nn.Module):
    config = SimpleNamespace(scaling_factor=0.5)

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.seen = []

    def encode(self, pixels):
        self.seen.append(pixels.clone())
        return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: pixels * self.weight))


def samples(offset=0):
    return [{"pixel_values": torch.ones(1, 2, 2) * (i + offset), "face_embs": torch.tensor([0.1, 0.3, 0.5])} for i in (1, 2)]


class SISSTests(unittest.TestCase):
    def test_defaults_and_method_specific_options(self):
        config = parse_config(["--method", "siss", "--identity-id", "512"])
        self.assertEqual(config, SISSConfig(identity_id=512))
        self.assertEqual((config.training_steps, config.batch_size, config.gradient_accumulation_steps), (60, 16, 4))
        self.assertEqual((config.optimizer, config.learning_rate, config.weight_decay, config.beta, config.train_mode), ("adamw", 5e-6, 0, 0.1, "full"))
        self.assertEqual(config.num_preserve_ids, 1000)
        for option in (["--reference-mode", "zero_uncond"], ["--preservation-weight", "0"], ["--use-anchor-overrides"], ["--beta", "nan"], ["--beta", "-1"], ["--num-preserve-ids", "-1"]):
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_config(["--method", "siss", "--identity-id", "512", *option])
        override = parse_config(["--method", "siss", "--identity-id", "512", "--beta", "0.2", "--image-root", "/tmp/images"])
        self.assertEqual((override.beta, override.image_root), (0.2, Path('/tmp/images')))

    def test_gradient_rule_and_zero_forget_norm(self):
        retain = (torch.tensor([3., 4.]),)
        forget = (torch.tensor([0., 2.]),)
        combined, scale, rnorm, fnorm = combine_gradients(retain, forget, 0.1)
        torch.testing.assert_close(combined[0], torch.tensor([3., 3.5]))
        self.assertAlmostEqual(scale.item(), 0.25)
        combined, *_ = combine_gradients((torch.tensor([3., 4.]),), (torch.zeros(2),), 0.1)
        torch.testing.assert_close(combined[0], torch.tensor([3., 4.]))
        with self.assertRaises(FloatingPointError): combine_gradients((torch.tensor([float('nan')]),), (torch.ones(1),), 0.1)

    def test_real_image_loss_matches_independent_gradients(self):
        from diffusers import DDPMScheduler
        config = SISSConfig(identity_id=0, device="cpu", batch_size=2, gradient_batch_size=2, training_steps=1)
        forget_loader, retain_loader = create_training_loaders(samples(), samples(3), config)
        forget, retain = next(iter(forget_loader)), next(iter(retain_loader))
        context = SISSContext(TinyUNet(), TinyVAE(), DDPMScheduler(num_train_timesteps=10), TinyConditioner(), torch.device('cpu'))
        original_loss = diffusion_loss
        captured = []

        def capture(*args):
            captured.append(tuple(value.detach().clone() for value in args[2:]))
            return original_loss(*args)

        with patch('piu_unlearning.methods.siss.diffusion_loss', side_effect=capture):
            result = build_method(config).compute_loss(forget, retain, context)
        torch.testing.assert_close(captured[0][0], retain['pixel_values'] * 0.5)
        torch.testing.assert_close(captured[1][0], forget['pixel_values'] * 0.5)
        torch.testing.assert_close(captured[0][2], captured[1][2])
        torch.testing.assert_close(captured[0][3], captured[1][3])
        losses = [original_loss(context.model, context.scheduler, *args) for args in captured]
        retain_grad, = torch.autograd.grad(losses[0], context.model.parameters())
        forget_grad, = torch.autograd.grad(losses[1], context.model.parameters())
        expected = retain_grad - config.beta * retain_grad.norm() / (forget_grad.norm() + 1e-8) * forget_grad
        torch.testing.assert_close(result.gradients[0], expected)
        self.assertIsNone(context.model.weight.grad)
        self.assertIsNone(context.vae.weight.grad)
        self.assertIsNone(context.conditioner.weight.grad)
        for kind in ('epsilon', 'v_prediction'):
            scheduler = DDPMScheduler(num_train_timesteps=10, prediction_type=kind)
            clean, cond, noise, steps = captured[0]
            target = noise if kind == 'epsilon' else scheduler.get_velocity(clean, noise, steps)
            prediction = context.model(scheduler.add_noise(clean, noise, steps), steps, encoder_hidden_states=cond).sample
            expected_loss = (prediction - target).square().mean()
            torch.testing.assert_close(diffusion_loss(context.model, scheduler, clean, cond, noise, steps), expected_loss)

    def test_manifest_pairs_both_training_partitions(self):
        config = SISSConfig(identity_id=0, device='cpu')
        split = make_split(config)
        paths = [Path(f'/tmp/{i}.png') for i in range(24)]
        with patch('piu_unlearning.methods.siss.load_image_manifest', return_value=(paths, {})) as load:
            inputs = prepare_siss_inputs(split, config)
        load.assert_called_once_with(config)
        self.assertEqual(inputs.forget.paths, [paths[i] for i in split.forget_train.indices])
        self.assertEqual(inputs.retain.paths, [paths[i] for i in split.retain_train.indices])

    def test_retain_pool_matches_legacy_seeded_cap(self):
        retain = EmbeddingPartition(torch.zeros(1003, 3), torch.arange(1003), torch.arange(1003))
        forget = EmbeddingPartition(torch.zeros(1, 3), torch.zeros(1, dtype=torch.long), torch.tensor([1003]))
        split = SimpleNamespace(forget_train=forget, retain_train=retain)
        paths = [Path(f'/tmp/{index}.png') for index in range(1004)]
        ids = list(range(1003))
        random.Random(42).shuffle(ids)
        expected = sorted(ids[:1000])
        with patch('piu_unlearning.methods.siss.load_image_manifest', return_value=(paths, {})):
            inputs = prepare_siss_inputs(split, SISSConfig(identity_id=0))
            all_inputs = prepare_siss_inputs(split, SISSConfig(identity_id=0, num_preserve_ids=0))
        self.assertEqual(inputs.metadata['retain_identity_ids'], expected)
        self.assertEqual(inputs.metadata['retain_indices'], expected)
        self.assertEqual(inputs.retain.paths, [paths[index] for index in expected])
        self.assertEqual(inputs.metadata['sampler'], 'epoch_shuffle')
        self.assertEqual(len(all_inputs.retain), 1003)

    def test_missing_manifest_fails_before_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            config = SISSConfig(identity_id=0, device='cpu', output_dir=Path(directory), num_samples=2)
            with patch('piu_unlearning.main.prepare_siss_inputs', side_effect=FileNotFoundError('missing manifest')), patch('piu_unlearning.main.load_arc2face') as load:
                with self.assertRaises(FileNotFoundError): run_demo(config, make_split(config))
                load.assert_not_called()

    def test_training_dispatch_checkpoint_and_evaluation(self):
        from diffusers import DDPMScheduler
        with tempfile.TemporaryDirectory() as directory:
            config = SISSConfig(identity_id=0, device='cpu', output_dir=Path(directory), num_samples=2, training_steps=2, batch_size=2, gradient_batch_size=2, gradient_accumulation_steps=2, evaluation_every=1)
            split = make_split(config)
            model = SimpleNamespace(unet=TinyUNet(), vae=TinyVAE(), scheduler=DDPMScheduler(num_train_timesteps=10), tokenizer=None, text_encoder=None)
            initial = model.unet.weight.detach().clone()
            inputs = SISSInputs(samples(), samples(3), {"source": "test"})
            with patch('piu_unlearning.main.Arc2FaceIdentityConditioner', return_value=TinyConditioner()), patch('piu_unlearning.main.ArcFaceExtractor'), patch('piu_unlearning.main.evaluate_training_ism', return_value=(0.3, 0.7)) as evaluate, patch('piu_unlearning.methods.reference.select_anchor_embedding', side_effect=AssertionError('SISS has no anchor')):
                path = unlearn_identity(model, split, create_evaluation_conditions(split, config), config, siss_inputs=inputs)
            checkpoint = torch.load(path, weights_only=True)
            self.assertEqual(path.name, 'siss_unet.pt')
            self.assertEqual(checkpoint['beta'], 0.1)
            self.assertNotIn('negative_guidance_scale', checkpoint)
            self.assertEqual(evaluate.call_count, 2)
            self.assertFalse(torch.equal(initial, model.unet.weight))
            self.assertTrue(model.unet.training)
            self.assertFalse((Path(directory) / 'reference.json').exists())
            history = [json.loads(line) for line in (path.parent / 'loss_history.jsonl').read_text().splitlines()]
            self.assertEqual([row['step'] for row in history], [1, 2])
            self.assertTrue(all('retain_grad_norm' in row for row in history))

    def test_accumulated_gradients_and_ema_match_manual_updates(self):
        from diffusers.training_utils import EMAModel
        from piu_unlearning.training.losses import LossOutput
        from torch.utils.data import DataLoader
        from piu_unlearning.visualization import write_training_curves

        with tempfile.TemporaryDirectory() as directory:
            config = SISSConfig(identity_id=0, output_dir=Path(directory), training_steps=3, gradient_accumulation_steps=2, evaluation_every=1, max_grad_norm=None, learning_rate=0.01)
            model = nn.Linear(2, 1, bias=False)
            initial = model.weight.detach().clone()
            reference = nn.Parameter(initial.clone())
            optimizer = torch.optim.AdamW([reference], lr=config.learning_rate, weight_decay=0)
            ema = EMAModel([reference])
            retained = [torch.tensor([[3., 4.]]), torch.tensor([[4., 3.]])]
            forgotten = [torch.tensor([[0., 2.]]), torch.tensor([[2., 0.]])]
            manual = [r - 0.1 * r.norm() / f.norm() * f for r, f in zip(retained, forgotten)]
            expected_evaluations = []
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                reference.grad = sum(manual) / 2
                optimizer.step()
                ema.step([reference])
                expected_evaluations.append(ema.shadow_params[0].clone())
            calls = []

            def objective(batch, _retain):
                index = int(batch.item())
                gradients, *_ = combine_gradients((retained[index].clone(),), (forgotten[index].clone(),), 0.1)
                return LossOutput(torch.tensor(1.), {"forget_loss": torch.tensor(2.), "retain_loss": torch.tensor(3.)}, gradients)

            def evaluate(step):
                calls.append(model.weight.detach().clone())
                return 0.3, 0.7

            path = train_model(model, objective, DataLoader(torch.arange(2), batch_size=1), None, config, evaluate)
            checkpoint = torch.load(path, weights_only=True)
            torch.testing.assert_close(checkpoint['unet_state_dict']['weight'], reference)
            torch.testing.assert_close(model.weight, ema.shadow_params[0])
            for actual, expected in zip(calls, expected_evaluations): torch.testing.assert_close(actual, expected)
            write_training_curves(path.parent / 'loss_history.jsonl', path.parent / 'ism_history.jsonl', Path(directory) / 'curves.png', 'siss')
            self.assertTrue((Path(directory) / 'curves.png').is_file())

    def test_image_manifest_loads_pairs_and_opens_images_on_demand(self):
        from test_wid import make_image_fixture
        with tempfile.TemporaryDirectory() as directory:
            wid, split, _ = make_image_fixture(Path(directory))
            config = SISSConfig(identity_id=0, data_dir=wid.data_dir, batch_size=2, gradient_batch_size=2, training_steps=1)
            inputs = prepare_siss_inputs(split, config)
            forget, retain = create_training_loaders(inputs.forget, inputs.retain, config)
            self.assertEqual(next(iter(forget))['pixel_values'].shape, (2, 3, 512, 512))
            self.assertEqual(next(iter(retain))['pixel_values'].shape, (2, 3, 512, 512))
            inputs.retain.paths[0].write_bytes(b'changed')
            prepare_siss_inputs(split, config)
            with self.assertRaises(OSError): inputs.retain[0]
