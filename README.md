# PIU

## Install

Use Linux x86-64 with Python 3.10, CUDA 12.4, and an NVIDIA GPU.

```bash
conda create --name piu python=3.10 pip -y
conda activate piu
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps .
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
