import copy
import json
import tempfile
import unittest
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from piu_unlearning.config import SISSConfig, parse_config
from piu_unlearning.data import create_training_loaders
from piu_unlearning.methods import build_method
from piu_unlearning.methods.siss import SISSContext, combine_gradients, diffusion_loss
from piu_unlearning.models.arc2face import load_arc2face
from piu_unlearning.training.runner import train_model
from test_methods import TinyConditioner
from test_siss import TinyVAE


class SpatialUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[[[0.3, -0.1], [0.5, 0.2]]]]))
        self.bias = nn.Parameter(torch.tensor(0.4))

    def forward(self, noisy, timesteps, encoder_hidden_states):
        return SimpleNamespace(sample=noisy * self.weight + self.bias * encoder_hidden_states.mean(dim=(1, 2))[:, None, None, None])


class SISSBatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def datasets(self):
        generator = torch.Generator().manual_seed(31)
        return [[{'pixel_values': torch.randn(1, 2, 2, generator=generator) + offset, 'face_embs': torch.randn(3, generator=generator)} for _ in range(21)] for offset in (0, 2)]

    def test_config_rejects_incomplete_groups(self):
        for kwargs in ({'gradient_batch_size': 0}, {'batch_size': 32}, {'batch_size': 3}, {'batch_size': 4, 'gradient_accumulation_steps': 3}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError): SISSConfig(identity_id=0, **kwargs)
        for batch in (1, 4, 16):
            config = parse_config(['--method', 'siss', '--identity-id', '0', '--batch-size', str(batch), '--gradient-accumulation-steps', str(64 // batch)])
            self.assertEqual(config.gradient_batch_size, 16)
            self.assertTrue(config.gradient_checkpointing)
        config = parse_config(['--method', 'siss', '--identity-id', '0', '--gradient-batch-size', '32', '--no-gradient-checkpointing'])
        self.assertEqual(config.gradient_batch_size, 32)
        self.assertFalse(config.gradient_checkpointing)

    def test_microbatch_gradients_match_full_batch_oracle(self):
        from diffusers import DDPMScheduler
        config = SISSConfig(identity_id=0)
        loaders = create_training_loaders(*self.datasets(), config)
        forget, retain = [next(iter(loader)) for loader in loaders]
        context = SISSContext(SpatialUNet(), TinyVAE(), DDPMScheduler(num_train_timesteps=10), TinyConditioner(), torch.device('cpu'))
        torch.manual_seed(9)
        noise = torch.randn_like(retain['pixel_values'])
        timesteps = torch.randint(10, (16,))
        losses = [diffusion_loss(context.model, context.scheduler, batch['pixel_values'] * 0.5, context.conditioner.encode(batch['face_embs']).detach(), noise, timesteps) for batch in (retain, forget)]
        retained, forgotten = [torch.autograd.grad(loss, tuple(context.model.parameters())) for loss in losses]
        rnorm, fnorm = [sum(g.square().sum() for g in gradients).sqrt() for gradients in (retained, forgotten)]
        scale = 0.1 * rnorm / (fnorm + 1e-8)
        expected = [r - scale * f for r, f in zip(retained, forgotten)]
        for batch_size in (1, 4, 16):
            config = SISSConfig(identity_id=0, batch_size=batch_size, gradient_accumulation_steps=64 // batch_size)
            context.vae.seen.clear()
            torch.manual_seed(9)
            result = build_method(config).compute_loss(forget, retain, context)
            for actual, reference in zip(result.gradients, expected): torch.testing.assert_close(actual, reference)
            for key, reference in [('retain_grad_norm', rnorm), ('forget_grad_norm', fnorm), ('scaling_factor', scale)]: torch.testing.assert_close(result.metrics[key], reference)
            self.assertEqual(max(len(pixels) for pixels in context.vae.seen), batch_size)
            self.assertTrue(all(parameter.grad is None for parameter in context.model.parameters()))

    def test_memory_settings_preserve_optimizer_ema_and_evaluation(self):
        from diffusers import DDPMScheduler
        results = []
        datasets = self.datasets()
        with tempfile.TemporaryDirectory() as directory:
            for batch_size in (16, 4, 1):
                config = SISSConfig(identity_id=0, output_dir=Path(directory) / str(batch_size), device='cpu', batch_size=batch_size, gradient_accumulation_steps=64 // batch_size, training_steps=2, evaluation_every=1, learning_rate=0.01, max_grad_norm=0.2)
                context = SISSContext(SpatialUNet(), TinyVAE(), DDPMScheduler(num_train_timesteps=10), TinyConditioner(), torch.device('cpu'))
                loaders = create_training_loaders(*datasets, config)
                self.assertEqual([len(loader) for loader in loaders], [8, 8])
                self.assertEqual([loader.batch_size for loader in loaders], [16, 16])
                evaluate = []

                def evaluation(step):
                    evaluate.append((step, copy.deepcopy(context.model.state_dict())))
                    return 0.2, 0.8

                objective = partial(build_method(config).compute_loss, context=context)
                with patch('piu_unlearning.methods.siss.combine_gradients', wraps=combine_gradients) as combine:
                    torch.manual_seed(9)
                    path = train_model(context.model, objective, *loaders, config, evaluation)
                    self.assertEqual(combine.call_count, 8)
                checkpoint = torch.load(path, weights_only=True)
                self.assertEqual(checkpoint['config']['gradient_batch_size'], 16)
                self.assertEqual(checkpoint['gradient_accumulation_steps'], 64 // batch_size)
                self.assertEqual(checkpoint['ema_state_dict']['optimization_step'], 2)
                history = [json.loads(line) for line in (path.parent / 'loss_history.jsonl').read_text().splitlines()]
                results.append((checkpoint, evaluate, context.model.state_dict(), history))
        reference, evaluations, final, history = results[0]
        for actual, actual_evaluations, actual_final, actual_history in results[1:]:
            torch.testing.assert_close(actual['unet_state_dict'], reference['unet_state_dict'])
            torch.testing.assert_close(actual['ema_state_dict'], reference['ema_state_dict'])
            torch.testing.assert_close(actual_final, final)
            for (step, weights), (ref_step, ref_weights) in zip(actual_evaluations, evaluations):
                self.assertEqual(step, ref_step)
                torch.testing.assert_close(weights, ref_weights)
            for row, ref_row in zip(actual_history, history):
                for key in row: self.assertAlmostEqual(row[key], ref_row[key], places=5)

    def test_checkpointing_supports_siss_autograd_and_model_loading(self):
        from diffusers import DDPMScheduler, UNet2DConditionModel
        model = UNet2DConditionModel(sample_size=4, in_channels=1, out_channels=1, block_out_channels=(8, 16), down_block_types=('CrossAttnDownBlock2D', 'DownBlock2D'), up_block_types=('UpBlock2D', 'CrossAttnUpBlock2D'), layers_per_block=1, norm_num_groups=4, cross_attention_dim=3, attention_head_dim=2).train()
        plain = copy.deepcopy(model)
        pipeline = SimpleNamespace(unet=model, text_encoder=nn.Linear(3, 3), vae=TinyVAE(), scheduler=DDPMScheduler(num_train_timesteps=10))
        pipeline.to = lambda device: pipeline
        with patch('piu_unlearning.models.arc2face_text_encoder.CLIPTextModelWrapper.from_pretrained', return_value=pipeline.text_encoder), patch('diffusers.UNet2DConditionModel.from_pretrained', return_value=model), patch('diffusers.StableDiffusionPipeline.from_pretrained', return_value=pipeline):
            loaded = load_arc2face(SISSConfig(identity_id=0, device='cpu'))
        self.assertIs(loaded.unet, model)
        self.assertTrue(model.is_gradient_checkpointing)
        config = SISSConfig(identity_id=0, batch_size=1, gradient_accumulation_steps=16)
        batch = {'pixel_values': torch.randn(2, 1, 4, 4), 'face_embs': torch.randn(2, 3)}
        gradients = []
        for unet in (plain, model):
            context = SISSContext(unet, TinyVAE(), DDPMScheduler(num_train_timesteps=10), TinyConditioner(), torch.device('cpu'))
            torch.manual_seed(7)
            gradients.append(build_method(config).compute_loss(batch, batch, context).gradients)
        torch.testing.assert_close(gradients[0], gradients[1])
