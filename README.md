# PIU

## Install

Use Linux x86-64 with Python 3.10, CUDA 12.4, and an NVIDIA GPU.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
pip install --no-deps -e .
```

## Prepare the data

Run the complete preparation pipeline once. It downloads the published CelebA-HQ identity dataset and ArcFace models, extracts embeddings, recreates the DBSCAN identity labels, verifies them against the published labels, and writes the cluster centroids.

```bash
piu-prepare-data --output-dir data/celebahq_512 --device cuda
```

The two stages can also be run separately:

```bash
piu-prepare-data --stage embeddings --output-dir data/celebahq_512 --device cuda
piu-prepare-data --stage clusters --output-dir data/celebahq_512
```

## Run the demo

```bash
piu-demo --identity-id 512 --data-dir data/celebahq_512 --use-anchor-overrides
```

Results are written to `outputs/demo`.
