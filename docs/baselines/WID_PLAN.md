# WID implementation plan

Status: implemented as the initial single-step adaptation. See [WID usage](WID.md) for artifact requirements, the explicit identity-target correction, and validation limits. This document preserves the pre-implementation analysis.

This plan maps the inspected `unlearning-identities` implementation onto the smaller `piu-unlearning` codebase. It targets the legacy Arc2Face adaptation, not an independently verified reproduction of the original WID paper. No runtime changes are required for this planning step.

## What the old code actually does

The relevant legacy files are `src/unlearning/strategies/wid.py`, `src/unlearning/data_utils.py`, `src/unlearning/experiment_helpers.py`, `src/unlearning/experiment_runner.py`, `src/utils/torch_arcface_utils.py`, `src/unlearning/presets/wid.py`, and `train_scripts/wid/wid_grid_search.py`.

1. Load real forget-training images paired with their individual ArcFace embeddings. The WID loader uses the raw training rows, even when the generic preset advertises centroid sampling.
2. Resize/center-crop RGB images to 512 pixels and normalize to [-1, 1]. Encode with the frozen VAE, sample its latent posterior, and apply the VAE scaling factor.
3. Sample diffusion timesteps across the scheduler's full range and add noise. The student and frozen teacher receive exactly the same noisy latent and timestep tensor, conditioned on the forget sample and selected anchor respectively.
4. Compute mean-squared noise-prediction error. There is no negative-guidance repulsion in this branch; the old negative-guidance config value is unused.
5. Reconstruct the clean latent from that same student prediction, divide by the VAE scaling factor, and decode. Pass the decoded image through a frozen, differentiable face recognizer, then compare its normalized embedding with the normalized anchor target using mean MSE.
6. Add the weighted identity loss and, optionally, retain distillation. Use the shared optimizer loop with gradient accumulation and gradient-norm clipping at 1.0.

For the first implementation:

```text
z0 = VAE.encode(real_image).latent_dist.sample() * scaling_factor
zt = scheduler.add_noise(z0, noise, t)
eps = student(zt, t, forget_condition)
eps_anchor = teacher(zt, t, anchor_condition)        # no gradients
model_loss = mean((eps - eps_anchor)^2)
z0_pred = (zt - sqrt(1 - alpha_bar[t]) * eps) / sqrt(alpha_bar[t])
id_pred = normalize(recognizer(VAE.decode(z0_pred / scaling_factor)))
identity_loss = mean((id_pred - identity_target)^2)
loss = model_loss + identity_weight * identity_loss
```

Only VAE encoding, conditioning, teacher inference, and target preparation use `no_grad`. VAE decoding and recognizer inference must retain input gradients while their parameters stay frozen and their modules remain in evaluation mode.

## Resolve before claiming reproduction

### Image-row mapping

Canonical preparation currently downloads embeddings, labels, and centroids, but no image manifest. Recomputed preparation writes `file_names.txt`, but does not materialize an image-backed training dataset.

Add an explicit row-aligned manifest plus an image root. Select images using `split.forget_train.indices`; never infer alignment from a sorted directory listing or just matching row counts. Obtain the canonical mapping from the original `image_paths.txt`/export metadata and validate it against the published artifact ordering. The legacy dataset upload script iterates over aligned paths and labels, but that alone does not establish equivalence between every dataset/artifact revision.

Make image preparation opt-in. PIU and UCE work with embeddings alone; SISS and WID require verified real images. Missing mappings or files should fail before model loading, without silently dropping rows or recomputing identity labels.

### Recognition-space compatibility

The final-run preset selects `ir_se50` and `/home/jose/backbone_ir_se_50.pth`. The identity target is still taken directly from the stored ArcFace anchor embedding. There is no compatibility check in the training path. An alignment diagnostic exists, but the inspected tests use synthetic weights and do not establish compatibility of the real models.

Do not assume two 512-dimensional face embeddings are comparable. First establish checkpoint provenance, preprocessing, and output parity with the recognizer that produced the stored embeddings. Test identical aligned crops to isolate model differences, then test the complete image preprocessing path. The old training preprocessing resizes the entire decoded image to 112 pixels and swaps RGB to BGR; evaluation instead detects and aligns a face. Channel order, normalization, and cropping need explicit validation.

Preferred path: one verified compatible differentiable recognizer. If that is unavailable, compute the identity-loss anchor target from retained anchor images using the same training recognizer as the decoded prediction. Keep the original ArcFace anchor for diffusion conditioning and selection. This is a deliberate correction to the legacy target construction, not numerically identical reproduction, and must be documented before adoption. Do not silently default to the unchecked cross-backend comparison.

The required real checkpoint and canonical image manifest were not found among the inspected repository files. The core loss can be implemented and tested with small models while these artifacts are resolved, but full reproduction cannot be certified without them.

### Preserve the loss scale

The old code's log mentions cosine loss, but the active loss is `F.mse_loss(..., reduction="mean")`. For normalized D-dimensional vectors, this equals `2 * (1 - cosine) / D`, not squared Euclidean distance summed over features. Preserve mean MSE for legacy parity; do not substitute cosine or summed MSE while retaining the same identity weight.

## Defaults and provenance

The current final-run grid script selects learning rate `5e-6`, identity weight `0.1`, and 100 steps. These values also appear in the local final-results HTML report. The base preset instead says `1e-4` and `0.3`; do not use those as the selected experimental defaults.

| Setting | Proposed initial value | Evidence |
| --- | --- | --- |
| Learning rate / identity weight / steps | `5e-6` / `0.1` / `100` | Final-run script and results report |
| Optimizer / weight decay | AdamW / `0.01` | Shared legacy runner and preset |
| Trainable layers | Full U-Net | Preset |
| Micro-batch / accumulation | `4` / `16` on one device | Preset; effective batch 64 |
| Model weight / preservation weight | `1.0` / `0.0` | Preset |
| Reconstruction | Single-step | Preset and active branch |
| Gradient clipping | `1.0` | Shared legacy runner |
| Timesteps | Full scheduler range | Active branch ignores split-gating flags |
| Reference | Proximity anchor, threshold `0.2` | Preset |

Treat settings inferred from the current preset as provisional until checked against archived run configs. The old distributed batch size is global, not per GPU; preserve effective-batch semantics rather than copying per-rank values. Keep common generation/evaluation settings when comparing against PIU/SISS/UCE, and record that this is distinct from replaying every legacy evaluation setting.

## Small module boundaries

| Location | Responsibility |
| --- | --- |
| `data.py` | One paired-image dataset and loader using the existing split indices; reuse the seeded sampler pattern. |
| `prepare_data.py` | Opt-in image materialization and row-manifest validation, without changing existing embedding preparation. |
| `models/identity_encoder.py` | Load one verified frozen training recognizer, perform differentiable preprocessing, normalize outputs, and record provenance. Leave the ONNX evaluation extractor unchanged. |
| `methods/wid.py` | `WID.prepare_reference`, `WID.compute_loss`, and a small explicit `WIDContext` containing the shared noise context, VAE, training recognizer, and identity target. |
| `config.py`, `methods/__init__.py`, `main.py` | Add `WIDConfig`, explicit dispatch, dependency validation, and context/loader assembly. |
| `training/runner.py` | Reuse the loop; add optional clipping with default disabled for existing methods. Remove the assumption that every method has a negative-guidance checkpoint field. |
| `visualization.py` | Reuse chart helpers, adding model/identity loss series when present. Keep existing methods' plots unchanged. |
| `baselines/run_all.py` | Register WID only after integration passes; preserve the existing shared split, baseline images, conditions, and subprocess isolation. |

Return `LossOutput` with `model_loss`, `identity_loss`, weighted model/identity terms, `forget_loss`, and `preserve_loss`. The runner already logs arbitrary metric names. `forget_loss` remains the combined weighted forget objective for existing charts; retain loss defaults to zero. Store method-specific settings in the existing checkpoint config rather than giving WID a fake negative-guidance parameter; preserve existing PIU checkpoint compatibility and keep SISS gradient metadata separate.

Do not create another trainer, evaluation pipeline, registry framework, or generic dependency container. Keep WID dependencies in its context, not as nullable fields added to every method. Validate artifacts at setup boundaries rather than wrapping each tensor operation. Freeze modules explicitly; test gradient flow instead of surrounding every operation with defensive checks.

Initially omit iterative reconstruction, unused timestep gates, synthetic WID anchors, EMA flags without an active implementation, multi-GPU device routing, and per-step debug-image machinery. Do not expose embedding-mixture/centroid sampling options for image-paired WID training. Do not import code from the old repository at runtime. Verify source licenses before adapting a recognizer implementation.

## Implementation order and acceptance checks

1. Establish the manifest and recognizer/target contract. Check canonical ordering, missing paths, preprocessing, checkpoint provenance, and compatibility before expensive generation.
2. Add the dataset and recognizer modules. Test row pairing, training-only selection, RGB/range/resolution, frozen parameters, and gradients through preprocessing.
3. Implement the single-step WID loss independently. Test the reconstruction equation, VAE scaling, shared noisy inputs/timesteps, exact mean-MSE reduction, and loss/gradient parity against the old implementation under fixed tensors.
4. Integrate through the existing runner. Verify nonzero U-Net gradients from the identity term alone, no teacher/VAE/recognizer parameter gradients, clipping after accumulation, disabled-identity-loss behavior, checkpoint serialization, and separate component logs.
5. Add CLI, plots, and comparison-launcher support. Retain all existing regression tests and add a mocked end-to-end WID run. Verify identical evaluation manifests across methods, with no holdout images used to construct the identity target.
6. Run a short CUDA experiment with real images and verified weights. Check finite losses/gradients, memory use, target provenance, checkpoints, component curves, and before/after outputs. Only then run the selected full configuration and assess reproduction.

Keep the WID usage guide in this folder once implemented; the main README needs only the existing baseline links.
