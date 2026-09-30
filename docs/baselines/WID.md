# WID baseline

[Back to PIU](../../README.md)

WID trains on real forget images, matching an anchor-conditioned frozen teacher and adding an identity loss through the VAE decoder and a frozen PyTorch face recognizer. It reuses the shared optimizer loop, evaluation, logging, and comparison grids.

This is the single-step Arc2Face adaptation, not the original WID model. The default identity target corrects an unchecked cross-recognizer comparison in the old code, so it is not an exact replay of the reported legacy experiment.

## Prepare real images

Install the package and prepare embeddings as in the main README. Reinstall the editable package after updating to register the new command:

```bash
python -m pip install --no-deps -e .
```

For published embeddings, obtain the original **row-aligned** `image_paths.txt` used with those embeddings. A sorted list of image files is not a substitute. Point at a flat directory of the corresponding images:

```bash
piu-prepare-images --data-dir data/celebahq_512 \
  --image-paths /path/to/original/image_paths.txt \
  --image-root /path/to/celebahq_images --device cuda
```

The command uses the file names from that mapping in exactly the supplied order. It verifies every image against its stored ArcFace embedding (cosine at least `0.985`), then writes `image_manifest.json` with image, embedding, and label hashes. It does not replace embeddings or recluster identities. Verification is empirical: a low score can reflect extraction/backend differences rather than a wrong filename, and a passing score cannot distinguish all near-identical images. Retain the original mapping's provenance.

Checks are flushed to `image_verification.jsonl` after each row. Rerunning the command reuses passing checks only when the image hashes, data/mapping hashes, and recorded model/runtime settings still match; failed rows are retried. A completed scan saves all failures in `image_verification_report.json` and writes no new training manifest unless every row passes. Older versions did not save partial progress, so their interrupted scans cannot be resumed.

To diagnose particular zero-based rows without a full scan or download:

```bash
piu-prepare-images --data-dir data/celebahq_512 \
  --image-paths data/celebahq_512/file_names.txt \
  --image-root data/celebahq_512/images --device cpu --check-rows 1882 10753
```

Repeat with `--device cuda` to compare backends. This prints each result and saves `image_verification_check.json`; it never creates a training manifest. CPU and CUDA results are cached separately. Do not lower the cutoff just to bypass a failure; investigate extraction settings and the original image/mapping provenance first.

To download and materialize the named images instead, install the data extras and use:

```bash
python -m pip install -e '.[data]'
piu-prepare-images --data-dir data/celebahq_512 \
  --image-paths /path/to/original/image_paths.txt --download-images --device cuda
```

This uses the pinned dataset revision and saves images in `DATA_DIR/images`. For recomputed embeddings, the existing `file_names.txt` provides the row order:

```bash
piu-prepare-images --data-dir data/celebahq_512_recomputed --download-images --device cuda
```

Use the same dataset revision used for extraction if it was overridden. WID checks the manifest against the current embeddings/labels and rechecks all images it uses. `--image-root` on the demo can relocate an unchanged image directory. Existing embedding-only methods do not need this preparation.

## Run

Supply trusted pretrained **IR-SE50** recognition weights compatible with [InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch). The loader accepts a tensor state dict, optionally under `state_dict`, and strips a leading `module.` prefix. It checks the architecture strictly; ONNX files and iresnet100 checkpoints are not interchangeable with IR-SE50. No recognition weights are downloaded automatically; their licensing and provenance remain your responsibility.

```bash
piu-demo --method wid --identity-id 512 --data-dir data/celebahq_512 \
  --use-anchor-overrides --identity-checkpoint /path/to/backbone_ir_se_50.pth
```

Defaults follow the selected legacy final-run configuration where established: learning rate `5e-6`, identity weight `0.1`, and 100 optimizer steps. The accompanying preset/runner supplies AdamW, weight decay `0.01`, full U-Net training, micro-batch `4`, accumulation `16`, model weight `1`, preservation weight `0`, and gradient clipping at `1`. These supporting settings should still be checked against archived run configs before claiming exact reproduction.

Training uses paired individual images, not centroid/Dirichlet image mixtures. Images are resized/center-cropped to 512 pixels and scaled to [-1, 1]. Model and identity losses use the same noisy latent and timestep, sampled over the full scheduler range. Reconstruction is single-step. The identity loss is **mean MSE** between normalized embeddings, not cosine loss or a sum over features. The initial implementation omits iterative reconstruction and inactive legacy gating options.

## Identity targets

`--identity-target anchor_images` is the default: the target is the normalized mean of training-recognizer embeddings of the retained anchor's training images. The generated reconstruction uses that same recognizer and preprocessing. Diffusion conditioning and proximity selection still use the original ArcFace embeddings. This deliberately differs from the old code, which compared an IR-SE50 prediction directly with a stored ArcFace target without a compatibility check.

`--identity-target stored_arcface` requests the old target construction, but first requires cosine at least `0.99` between training-recognizer outputs and stored embeddings for every forget-training and selected anchor-training image. Incompatible models fail before Arc2Face loading. This is an empirical gate, not an assertion that arbitrary IR-SE50 weights match the canonical ArcFace space; the legacy checkpoint may fail it.

The default `--identity-channel-order bgr` retains the old preprocessing convention: differentiable full-image resize to 112 pixels, [-1, 1] values, and an RGB-to-BGR swap. `rgb` is available for weights whose documented input contract requires it. Neither path performs evaluation's face detection/alignment, and no pretrained checkpoint's end-to-end quality has been validated here. The target and prediction always use the same training preprocessing.

Setting `--identity-loss-weight 0` disables the recognizer and decoding branch and removes the checkpoint requirement, but still trains on real-image latents. This is an alignment-only ablation, not full WID. Retain distillation can be enabled with `--preservation-weight`; it is off by default.

## Outputs and comparison

Results go to `outputs/wid_demo` by default. The usual configuration, split, checkpoint (`checkpoints/wid_unet.pt`), before/after images, and ISM/AccU/AccR/SRK metrics are saved. Loss logs include raw model/identity losses and weighted components; training curves show those components alongside validation ISM.

`wid_identity.json` records the manifest/checkpoint hashes, training and anchor row indices, preprocessing, and target mode. `identity_target.pt` saves the actual training target; `reference_embedding.pt` remains the diffusion-conditioning anchor.

The [comparison launcher](../../baselines/README.md) includes WID and checks its dependencies before starting any method:

```bash
python baselines/run_all.py --identity-id 512 --data-dir data/celebahq_512 \
  --use-anchor-overrides --output-dir outputs/comparison_512 \
  --wid-args='--identity-checkpoint /path/to/backbone_ir_se_50.pth'
```

Use `--methods piu esd uce` if real images or recognition weights are unavailable. Use the same target mode/checkpoint/preprocessing when comparing WID experiments. Shared evaluation settings and deterministic splits do not by themselves reproduce archived experiments.

CPU tests cover image pairing/integrity, reconstruction, identity-only gradients, frozen modules, the real IR-SE50 architecture with synthetic weights, checkpointing, plots, and mocked demo/launcher execution. The loss and gradients match the old implementation under fixed test inputs and an identical target. Full Arc2Face/CUDA training and pretrained recognition quality remain unvalidated.
