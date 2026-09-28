# PIU

## Install

Use Linux x86-64 with Python 3.10, CUDA 12.4, and an NVIDIA GPU.

```bash
conda create --name piu python=3.10 pip -y
conda activate piu
export PYTHONNOUSERSITE=1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
```

## Prepare the paper data

Download the exact embeddings, identity labels, centroids, and face models used by the paper:

```bash
piu-prepare-data --output-dir data/celebahq_512
```

## Run the demo

```bash
piu-demo --identity-id 512 --data-dir data/celebahq_512 --use-anchor-overrides
```

Results are written to `outputs/demo`.
The resolved defaults and command-line overrides are recorded in `outputs/demo/config.json`.
ISM is evaluated every 50 optimizer steps by default; use `--evaluation-every 0` to disable it or pass another interval.

## Independently recompute the data

To reproduce data construction instead of downloading the paper artifacts, extract ArcFace embeddings and rerun DBSCAN:

```bash
piu-recompute-data --output-dir data/celebahq_512_recomputed --device cuda
```

The command writes its own labels and centroids and records the adjusted Rand index against the published partition in `metadata.json`. A difference does not stop preparation because independently generated cluster IDs are not the paper identity IDs. Run the demo on recomputed data without `--use-anchor-overrides`.
