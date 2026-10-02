# WID baseline

[Back to PIU](../../README.md)

WID trains on real forget images, matching an anchor-conditioned frozen teacher and adding an identity loss through the VAE decoder and a frozen PyTorch face recognizer. It reuses the shared optimizer loop, evaluation, logging, and comparison grids.

This is the single-step Arc2Face adaptation, not the original WID model. The default identity target corrects an unchecked cross-recognizer comparison in the old code, so it is not an exact replay of the reported legacy experiment.

## Prepare real images

Install the package and prepare embeddings as in the main README. Reinstall the editable package after updating to register the new command:

```bash
python -m pip install --no-deps -e .
```

The demo downloads the hosted images and prepares their manifest automatically when needed. To stage them in advance, download the images from the pinned dataset. The command creates `file_names.txt` in dataset row order and hashes the images in `image_manifest.json`:

```bash
piu-prepare-images --data-dir data/celebahq_512 --download-images
```

For recomputed embeddings, `piu-recompute-data` already writes `file_names.txt` alongside the embeddings. The same image command compares that list against the downloaded dataset row order:

```bash
piu-prepare-images --data-dir data/celebahq_512_recomputed --download-images
```

Use the same dataset revision used for extraction if it was overridden. The manifest checks row order, image integrity, and embedding/label file hashes without extracting faces again. For published embeddings, dataset order is provenance, not an independent proof that every stored embedding matches its image. It does not replace embeddings or recluster identities. WID rechecks the hashes and readability of images it uses before training. `--image-root` on the demo can relocate an unchanged image directory. Embedding-only methods do not need image preparation.

## Run

Supply trusted pretrained **IR-SE50** recognition weights compatible with [InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch). The loader accepts a tensor state dict, optionally under `state_dict`, and strips a leading `module.` prefix. It checks the architecture strictly; ONNX files and iresnet100 checkpoints are not interchangeable with IR-SE50. No recognition weights are downloaded automatically; their licensing and provenance remain your responsibility.

```bash
piu-demo --method wid --identity-id 512 --data-dir data/celebahq_512 \
  --use-anchor-overrides --identity-checkpoint /path/to/backbone_ir_se_50.pth
```

For a one-command WID run, including data/image preparation when missing, use the helper from the repository root:

```bash
baselines/run_wid.sh 512 /path/to/backbone_ir_se_50.pth --use-anchor-overrides
```

It reuses a complete data directory, downloads the paper artifacts if none are present, prepares the image manifest, and runs WID through the shared evaluation launcher. Set `CUDA_VISIBLE_DEVICES` before calling it to select a GPU. Use `--data-dir data/celebahq_512_recomputed` for recomputed data and omit `--use-anchor-overrides` in that case. `--output-dir PATH` changes the result location; the default is `outputs/wid_ID`.

Defaults follow the selected legacy final-run configuration where established: learning rate `5e-6`, identity weight `0.1`, and 100 optimizer steps. The accompanying preset/runner supplies AdamW, weight decay `0.01`, full U-Net training, micro-batch `4`, accumulation `16`, model weight `1`, preservation weight `0`, and gradient clipping at `1`. These supporting settings should still be checked against archived run configs before claiming exact reproduction.

Training uses paired individual images, not centroid/Dirichlet image mixtures. Images are resized/center-cropped to 512 pixels and scaled to [-1, 1]. Model and identity losses use the same noisy latent and timestep, sampled over the full scheduler range. Reconstruction is single-step. The identity loss is **mean MSE** between normalized embeddings, not cosine loss or a sum over features. The initial implementation omits iterative reconstruction and inactive legacy gating options.

## Identity targets

The identity target is the normalized mean of IR-SE50 embeddings from the retained anchor's training images. The generated reconstruction uses that same recognizer and preprocessing. Diffusion conditioning and proximity selection continue to use the dataset's ArcFace embeddings; the code does not compare vectors across those recognizers.

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

Use `--methods piu siss uce` when recognition weights are unavailable; use `--methods piu uce` when real images are unavailable. Shared evaluation settings and deterministic splits do not by themselves reproduce archived experiments.

CPU tests cover image pairing/integrity, reconstruction, identity-only gradients, frozen modules, the real IR-SE50 architecture with synthetic weights, checkpointing, plots, and mocked demo/launcher execution. The loss and gradients match the old implementation under fixed test inputs and an identical target. Full Arc2Face/CUDA training and pretrained recognition quality remain unvalidated.
