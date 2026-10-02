# SISS baseline

[Back to PIU](../../README.md)

This implements the Arc2Face SISS adaptation described in [PIU Appendix C.1](https://arxiv.org/html/2605.22311v2#A3.SS1), following the legacy `unlearning-identities` SISS gradient update.

## Prepare images and run

SISS needs a row-aligned `image_manifest.json`. The demo downloads the hosted images and prepares the manifest automatically when they are missing. To stage them in advance or use recomputed embeddings, follow the [image preparation instructions](WID.md#prepare-real-images). Both forget-training and retain-training images must be available. A recognition checkpoint is not needed for SISS training.

```bash
piu-demo --method siss --identity-id 512 --data-dir data/celebahq_512
```

Use `--image-manifest` or `--image-root` to override the manifest or image directory. Missing or changed images fail before diffusion model loading. Validation images are excluded from training.

## Training

Defaults are 60 optimizer updates, GPU microbatch 16 with 4 accumulation steps, and a normalization batch of 16 (effective batch 64 for each branch), AdamW at `5e-6`, zero weight decay, beta `0.1`, and full U-Net training. Learning rate is constant; the accumulated gradient is clipped to norm 1, matching the legacy runner. VAE and identity-conditioning encoder remain frozen. EMA is enabled for intermediate and final evaluation, matching the legacy runner; use `--no-use-ema` to disable it. Checkpoints retain raw optimizer weights and the separate `ema_state_dict`.

Both branches encode real images into VAE latents and condition on their corresponding stored identity embeddings. Each paired normalization batch shares noise and timesteps. VAE encoding, conditioning, and U-Net forward/backward passes are split into GPU microbatches. Raw gradients are averaged separately for each branch before computing the ratio. The update is:

```text
g = g_retain - scale * g_forget
scale = beta * norm(g_retain) / (norm(g_forget) + 1e-8)
```

The scale is computed once per normalization batch before averaging the combined gradients across groups. The default has four groups of 16 per optimizer update, matching the legacy ratio mode with sample conditioning. `--beta` controls the relative forget-gradient strength. There is no anchor, negative-guidance setting, or PIU preservation weight.

### GPU memory and batching

`--batch-size` controls the GPU microbatch. `--gradient-accumulation-steps` retains its meaning: physical microbatches per optimizer update. Their product is the effective batch per branch. SISS additionally uses `--gradient-batch-size` (default 16) for the norm ratio; changing it changes the objective. The GPU batch must divide the normalization batch, and the effective batch must contain complete normalization batches.

These settings all keep four normalization groups of 16 and effective batch 64:

| GPU batch | Accumulation steps | Normalization batch |
| --- | --- | --- |
| 16 | 4 | 16 |
| 8 | 8 | 16 |
| 4 | 16 | 16 |
| 2 | 32 | 16 |
| 1 | 64 | 16 |

For example, use `--siss-args='--batch-size 4 --gradient-accumulation-steps 16'` with the comparison launcher. Gradient checkpointing is enabled by default to reduce U-Net activation memory; `--no-gradient-checkpointing` disables it. Smaller microbatches reduce activation memory, but model weights, optimizer/EMA state, and full parameter-gradient buffers still consume memory.

Clipping, optimizer updates, and EMA updates happen once per effective batch. CPU tests check the same gradients, optimizer state, and EMA weights for 16×4, 4×16, and 1×64 with fixed inputs and noise. Floating-point summation and stochastic VAE sampling can differ between GPU batch sizes; bitwise identical runs are not promised.

Earlier versions computed the ratio per physical microbatch, so 4×16 and 1×64 did not preserve the 16×4 update. Rerun affected experiments from the original weights with these corrected settings.

This is the gradient-based Arc2Face adaptation reported in PIU, not a claim to reproduce every setting of the original SISS paper. The legacy optional fixed scaling and centroid-conditioning variants are not included. Exact experiment replay also requires the saved split, image mapping, model versions, and seeds.

## Outputs and comparison

Results go to `outputs/siss_demo`. Outputs include resolved configuration, split, `siss_data.json` with manifest hashes and training indices, `checkpoints/siss_unet.pt`, before/after images, metrics, and training curves. Logs record forget/retain diffusion losses, gradient norms, and the adaptive scale. The reported total loss is `retain_loss - scale * forget_loss`; the optimizer uses the explicitly combined gradients.

ISM evaluation runs every 50 optimizer updates by default (`--evaluation-every 0` disables intermediate evaluation). Final evaluation still runs after training. Generation uses the shared 25 denoising steps and guidance scale 3.

```bash
python baselines/run_all.py --identity-id 512 --data-dir data/celebahq_512 \
  --methods piu siss uce --use-anchor-overrides \
  --output-dir outputs/comparison_512_piu_siss_uce
```

The default comparison order is PIU, SISS, UCE, WID. WID additionally requires its recognition checkpoint. Use `--siss-args='--beta 0.2'` for method-specific overrides, or `--dry-run` to inspect all resolved settings. CPU tests verify gradients, accumulation, image handling, and dispatch; full Arc2Face/CUDA reproduction remains unvalidated.
