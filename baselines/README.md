# Run baselines

After [installation](../README.md#install), run PIU, SISS, UCE, and WID in order from the repository root:

```bash
python baselines/run_all.py --identity-id 512 --data-dir data/celebahq_512 \
  --use-anchor-overrides --output-dir outputs/comparison_512
```

- `--identity-id ID`: identity to unlearn.
- `--methods piu siss uce wid`: select methods and order; all four run by default.
- `--data-dir DIR`: prepared paper or recomputed data.
- `--use-anchor-overrides`: use the paper comparison split; omit for recomputed data.
- `--output-dir DIR`: a new or empty directory for results.
- `--dry-run`: print resolved settings without running.
- `--METHOD-args='...'`: method options, for example `--siss-args='--beta 0.2'`. See [SISS](../docs/SISS.md), [UCE](../docs/UCE.md), and [WID](../docs/WID.md).

The output directory contains per-method runs, `comparison.csv`, and before/after comparison grids.

## Data

Use the [paper dataset](https://huggingface.co/datasets/edgarcancinoe/celebahq_512_id_clusters). SISS and WID also need images:

```bash
piu-prepare-data --output-dir data/celebahq_512
piu-prepare-images --data-dir data/celebahq_512 --download-images
```

Or recompute embeddings and clusters from the hosted images (requires `python -m pip install '.[data]'`):

```bash
piu-recompute-data --output-dir data/celebahq_512_recomputed --device cuda
piu-prepare-images --data-dir data/celebahq_512_recomputed --download-images
```

Choose an identity from the recomputed `labels.npy` and omit `--use-anchor-overrides`.
