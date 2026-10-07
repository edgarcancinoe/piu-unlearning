# UCE baseline

[Back to baselines](../baselines/README.md) | [PIU Appendix C.2](https://arxiv.org/html/2605.22311v2#A3.SS2)

UCE edits Arc2Face cross-attention weights directly; it has no optimizer or training steps.

```bash
piu-demo --method uce --identity-id 512 --data-dir data/celebahq_512 --use-anchor-overrides
```

Omit `--use-anchor-overrides` for recomputed data. Defaults from the PIU adaptation are `--erase-scale 20 --preserve-scale 1 --lamb 0.7 --edit-scope all`. The edit uses forget-training embeddings and 20 retain identity centroids, including the anchor. The legacy final run used no forget mixtures; add them with `--num-forget-combinations 32` if needed.

Results in `outputs/uce_demo` include the edited checkpoint, metrics, and before/after images. There are no training curves because the edit is closed-form. These settings follow the PIU final-run script, not universal defaults from the original UCE paper.
