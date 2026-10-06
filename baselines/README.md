# Run baselines

Run from the repository root. This compares PIU, SISS, UCE, and WID sequentially:

```bash
python baselines/run_all.py --identity-id 512 --data-dir data/celebahq_512 \
  --use-anchor-overrides --output-dir outputs/comparison_512
```

WID downloads and caches its pinned IR-SE50 weights automatically. Use `--wid-args='--identity-checkpoint /path/to/backbone_ir_se_50.pth'` to supply local weights.

## Options

- `--identity-id ID`: target identity.
- `--methods ...`: methods and run order. Defaults to all four.
- `--data-dir DIR`: paper or recomputed data.
- `--use-anchor-overrides`: use the paper comparison split; SISS does not use an anchor in training. Omit for recomputed data.
- `--output-dir DIR`: empty output directory.
- `--dry-run`: preview settings without running.
- `--METHOD-args='...'`: options for PIU, SISS, UCE, or WID. See [SISS](../docs/baselines/SISS.md), [UCE](../docs/baselines/UCE.md), and [WID](../docs/baselines/WID.md).

## Data

**Paper data:** `edgarcancinoe/celebahq_512_id_clusters`. `piu-prepare-data` writes `embeddings.npy`, `srk_labels_eps0.35.npy`, `centroids.npy`, and `centroid_labels.npy`. SISS and WID also require images:

```bash
piu-prepare-data --output-dir data/celebahq_512
piu-prepare-images --data-dir data/celebahq_512 --download-images
```

**Recompute embeddings and clusters:** extract ArcFace embeddings from the hosted images and run DBSCAN (`eps=0.35`, `min_samples=2`).

```bash
python -m pip install scikit-learn==1.7.2
piu-recompute-data --output-dir data/celebahq_512_recomputed --device cuda
piu-prepare-images --data-dir data/celebahq_512_recomputed --download-images
```

Choose an ID from the recomputed `labels.npy` and omit `--use-anchor-overrides`. Use `--device cpu` for CPU extraction.
