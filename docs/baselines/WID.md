# WID baseline

[Back to PIU](../../README.md)

WID trains on real forget images, matching an anchor-conditioned frozen teacher and adding an identity loss through the VAE decoder and a frozen PyTorch face recognizer. It reuses the shared optimizer loop, evaluation, logging, and comparison grids.

This is the single-step Arc2Face adaptation, not the original WID model. Unlike the old code's anchor-mean target, it compares each reconstruction with its own original image, as stated in the PIU appendix.

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

WID automatically downloads the pretrained **IR-SE50** [checkpoint hosted by AIRI-Institute](https://huggingface.co/AIRI-Institute/StyleFeatureEditor/blob/5a50cb1b78da946be8163b8b255d486996fd2c73/pretrained_models/model_ir_se50.pth) into the Hugging Face cache on first use (about 175 MB). The revision is pinned, and later runs reuse the cached weights. `HF_HOME` controls the cache location; `HF_HUB_OFFLINE=1` works once that cache is populated. The first download requires internet access.

To download it in advance from the terminal:

```bash
hf download AIRI-Institute/StyleFeatureEditor pretrained_models/model_ir_se50.pth --revision 5a50cb1b78da946be8163b8b255d486996fd2c73
```

To use your own compatible [InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch) weights, pass `--identity-checkpoint /path/to/backbone_ir_se_50.pth`. Local weights take precedence and skip downloading. The loader accepts a tensor state dict, optionally under `state_dict`, and strips a leading `module.` prefix. It checks the architecture strictly; ONNX files and iresnet100 checkpoints are not interchangeable with IR-SE50. Pretrained weights remain subject to their source's terms.

```bash
piu-demo --method wid --identity-id 512 --data-dir data/celebahq_512 \
  --use-anchor-overrides
```

For a one-command WID run, including data/image preparation when missing, use the helper from the repository root:

```bash
baselines/run_wid.sh 512 --use-anchor-overrides
```

An optional checkpoint path can still follow the identity ID to use local weights. It reuses a complete data directory, downloads the paper artifacts if none are present, prepares the image manifest, and runs WID through the shared evaluation launcher. Set `CUDA_VISIBLE_DEVICES` before calling it to select a GPU. Use `--data-dir data/celebahq_512_recomputed` for recomputed data and omit `--use-anchor-overrides` in that case. `--output-dir PATH` changes the result location; the default is `outputs/wid_ID`.

Defaults follow the PIU appendix: learning rate `5e-6`, identity weight `0.1`, 100 optimizer steps, AdamW with weight decay `0`, and effective batch `64` (micro-batch `4`, accumulation `16`). The code also trains the full U-Net, uses model weight `1`, preservation weight `0`, and clips gradients at `1`. The old preset used weight decay `0.01`; archived run configs would be needed to resolve that discrepancy.

Training uses paired individual images, not centroid/Dirichlet image mixtures. Images are resized/center-cropped to 512 pixels and scaled to [-1, 1]. Model and identity losses use the same noisy latent and timestep, sampled over the full scheduler range. Reconstruction is single-step. The identity loss is **mean MSE** between normalized embeddings, not cosine loss or a sum over features. The initial implementation omits iterative reconstruction and inactive legacy gating options.

## Identity targets

Each forget-training image has its own IR-SE50 target embedding, computed once before training and paired with that image in every batch. The generated reconstruction uses the same recognizer and preprocessing. Diffusion conditioning and proximity selection continue to use the dataset's ArcFace embeddings; the identity loss does not compare vectors across recognizers.

The default `--identity-channel-order bgr` retains the old preprocessing convention: differentiable full-image resize to 112 pixels, [-1, 1] values, and an RGB-to-BGR swap. `rgb` is available for weights whose documented input contract requires it. Neither path performs evaluation's face detection/alignment, and no pretrained checkpoint's end-to-end quality has been validated here. The target and prediction always use the same training preprocessing.

Setting `--identity-loss-weight 0` disables the recognizer and decoding branch and skips checkpoint loading and downloading, but still trains on real-image latents. This is an alignment-only ablation, not full WID. Retain distillation can be enabled with `--preservation-weight`; it is off by default.

## Outputs and comparison

Results go to `outputs/wid_demo` by default. The usual configuration, split, checkpoint (`checkpoints/wid_unet.pt`), before/after images, and ISM/AccU/AccR/SRK metrics are saved. Loss logs include raw model/identity losses and weighted components; training curves show those components alongside validation ISM.

`wid_identity.json` records the manifest/checkpoint hashes, forget-training row indices, preprocessing, and target mode. `identity_target.pt` saves the per-image targets in that row order; `reference_embedding.pt` remains the diffusion-conditioning anchor.

The [comparison launcher](../../baselines/README.md) includes WID and checks its dependencies before starting any method:

```bash
python baselines/run_all.py --identity-id 512 --data-dir data/celebahq_512 \
  --use-anchor-overrides --output-dir outputs/comparison_512
```

WID resolves the same cached recognition weights during preflight and worker execution; use `--methods piu uce` when real images are unavailable. Shared evaluation settings and deterministic splits do not by themselves reproduce archived experiments.

CPU tests cover image pairing/integrity, per-image targets, reconstruction, identity-only gradients, frozen modules, the real IR-SE50 architecture with synthetic weights, checkpointing, plots, and mocked demo/launcher execution. Full Arc2Face/CUDA training and pretrained recognition quality remain unvalidated.
