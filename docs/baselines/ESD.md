# ESD baseline

[Back to PIU](../../README.md)

This is an Arc2Face adaptation of ESD. Follow the [installation and data preparation instructions](../../README.md#install), then run from the repository root:

```bash
piu-demo --method esd --identity-id 512 --data-dir data/celebahq_512
```

## Settings

ESD uses the learning defaults from the [original paper, Section 5 and Appendix B.1](https://arxiv.org/html/2303.07345v3): 1,000 optimizer steps, learning rate `1e-5`, batch size `1`, no gradient accumulation, negative guidance `1.0`, and all cross-attention layers (`--train-mode x`). The optimizer is Adam with zero weight decay, as in the [authors' original training code](https://github.com/rohitgandikota/erasing/blob/a2189e9ae677aca22a00c361bde25d3d320d8a61/train-scripts/train-esd.py). Preservation loss is disabled by default. All learning settings can be overridden from the CLI, including `--optimizer adam|adamw`.

Its `zero_uncond` reference is a zero identity embedding passed through the Arc2Face encoder, not an empty text prompt. Use `--reference-mode gaussian_uncond` for a seeded, normalized Gaussian identity reference. Neither mode uses proximity anchors or `--use-anchor-overrides`.

Like PIU, ESD accepts `--train-mode surgical|x|full` and `--forget-sampling dirichlet|individual|centroid`. The centroid option averages only forget-training embeddings. PIU remains the default method with surgical layers, Dirichlet sampling, and preservation weight `10.0`. ESD can optionally use the same retain regularizer via `--preservation-weight`. See `piu-demo --method esd --help` for all options.

## Outputs

Results go to `outputs/esd_demo` unless `--output-dir` is set. ESD reuses PIU's training runner, logging, evaluation, and visualization: resolved configuration, data split, reference metadata and embedding, `checkpoints/esd_unet.pt`, before/after images, and metrics in `summary.json`.

Losses are recorded in `checkpoints/loss_history.jsonl`. Validation ISM is recorded every 50 optimizer steps in `checkpoints/ism_history.jsonl`; change the interval with `--evaluation-every`, or set it to `0` to disable intermediate evaluation. Training curves and the aligned before/after grid are saved under `summary/`.

## Reproduction notes

This matches the original ESD-x learning hyperparameters, not the complete original algorithm: our Arc2Face adaptation uses identity-embedding conditions and forward-noised random latents instead of ESD's partially denoised, model-generated training latents. Evaluation keeps the Arc2Face settings. Reproducing a particular PIU-paper experiment also requires its saved splits and settings.

The implementation has CPU loss, gradient, training, and checkpoint tests. Full Arc2Face/CUDA reproduction has not yet been validated.
