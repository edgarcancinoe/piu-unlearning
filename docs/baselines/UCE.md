# UCE baseline

[Back to PIU](../../README.md)

UCE edits cross-attention key/value projections directly, without gradient training. Follow the [installation and data preparation instructions](../../README.md#install), then run from the repository root:

```bash
piu-demo --method uce --identity-id 512 --data-dir data/celebahq_512 --use-anchor-overrides
```

Omit `--use-anchor-overrides` when using recomputed embeddings or selecting anchors automatically.

## Settings

The default weights are `--erase-scale 20 --preserve-scale 1 --lamb 0.7`, with `--edit-scope all`, matching the selected configuration in [PIU, Appendix C.2](https://arxiv.org/html/2605.22311v2#A3.SS2). These are the PIU-paper Arc2Face adaptation settings, not universal defaults from the original UCE paper.

Following the legacy final-run script, the edit uses raw forget-training embeddings, 20 retain-training identity centroids (including the anchor), and unnormalized branch weights. Use `--num-forget-combinations 32` to add seeded Dirichlet mixtures, `--num-preserve-ids 0` to preserve all training identities, or `--normalize-branch-weights` to average each branch's contribution. Validation samples never enter the edit.

`--edit-scope surgical` restricts edits to the configured blocks. UCE has no learning rate, optimizer, or training steps; use `piu-demo --method uce --help` for its options.

## Outputs

Results go to `outputs/uce_demo` unless `--output-dir` is set. UCE reuses PIU's before/after generation, evaluation, and comparison grid. Each run saves the resolved configuration, data split, anchor metadata and embedding, `checkpoints/uce_unet.pt`, before/after images, metrics in `summary.json`, and `summary/fixed_conditions.png`.

`checkpoints/edit.json` records the edited parameters, erase/preserve counts, preserve identity IDs, and effective branch weights. There are no training-loss logs or curves because the edit is closed-form.

## Reproduction notes

Conditioning is token-averaged as in the previous implementation. The appendix describes forget mixtures, but the final-run script sets their count to zero; this implementation follows the script. The new demo uses its shared deterministic splits and sampling, so matching weights alone does not guarantee identical published scores.

CPU tests cover the edit and demo integration, and the matrix update has been checked numerically against the old implementation. Full Arc2Face/CUDA reproduction has not yet been validated.
