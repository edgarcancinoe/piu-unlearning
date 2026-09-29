# Compare methods

After [installing the package and preparing embeddings](../README.md#install), run from the repository root:

```bash
python baselines/run_all.py --identity-id 512 --data-dir data/celebahq_512 \
  --use-anchor-overrides --output-dir outputs/comparison_512
```

This runs **PIU, ESD, then UCE**, each in a fresh process loaded from the original model. WID is not implemented yet. Omit `--use-anchor-overrides` for recomputed embeddings.

All methods use the same train/validation split, evaluation conditions, and generation seeds. The original-model images are generated once and copied to subsequent runs. Recorded anchors stay in the shared training split even for ESD, although ESD still uses its synthetic reference rather than the anchor. Each method keeps its own learning defaults; this is not an equal-compute comparison.

## Options

Use `--methods esd uce` to run a subset in that order. Shared settings such as `--seed`, `--num-samples`, and `--num-inference-steps` apply to every method. Use `--dry-run` to inspect resolved settings without loading data/models or writing files.

Method-specific options use the existing demo parser. For a short plumbing check, not a reproduction run:

```bash
python baselines/run_all.py --identity-id 512 --data-dir data/celebahq_512 \
  --use-anchor-overrides --output-dir outputs/comparison_smoke \
  --num-samples 2 \
  --piu-args='--training-steps 2 --batch-size 1 --gradient-accumulation-steps 1 --evaluation-every 0' \
  --esd-args='--training-steps 2 --evaluation-every 0'
```

Use the `--METHOD-args='...'` form, including `=`, when the value starts with `--`. These overrides cannot change shared data, seeds, evaluation settings, or output paths. See the [ESD](../docs/baselines/ESD.md) and [UCE](../docs/baselines/UCE.md) notes for method details and limitations.

## Results

- `comparison.md`: metric table and aligned forget/retain image grids.
- `comparison.csv` and `comparison.json`: ISM, AccU, AccR, SRK, and changes from the original model.
- `comparison_forget.png` and `comparison_retain.png`: original, PIU, ESD, and UCE outputs for each fixed condition.
- `piu/`, `esd/`, `uce/`: normal demo artifacts, plus `run.log` and the replayable worker arguments in `job.json`.
- `runs.json`: completion/failure status and elapsed wall time for each method.

Elapsed time includes model loading, generation, and evaluation; it is not a training/editing speed benchmark. The first method also generates the original-model images. Training logs and curves remain method-specific; UCE has edit metadata instead.

The launcher stops if a method fails or saved evaluation conditions differ. Completed outputs and failure logs remain available. Choose an empty output directory for each invocation to avoid mixing experiments. CPU tests cover orchestration and reporting; full CUDA runs remain to be validated.
