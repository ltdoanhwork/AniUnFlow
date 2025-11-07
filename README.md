# optical_flow

# 1) Create env (CUDA 12.x assumed; adapt to your machine)
conda create -n flow python=3.10 -y
conda activate flow


# 2) Install deps
pip install -r requirements.txt


# 3) Prepare AnimeRun manifests (edit paths inside config or pass via CLI)
python scripts/prepare_animerun_manifest.py \
--root /path/to/AnimeRun \
--out_csv data/animerun_train.csv --split train
python scripts/prepare_animerun_manifest.py \
--root /path/to/AnimeRun \
--out_csv data/animerun_val.csv --split val


# 4) Train baseline on AnimeRun
python train.py --config configs/animerun_baseline.yaml


# 5) Evaluate a checkpoint
python eval.py --config configs/animerun_baseline.yaml \
trainer.ckpt_path=/path/to/ckpt_best.pth

# Optical Flow Baseline (AnimeRun → LinkToAnime)

A minimal-yet-extensible PyTorch training stack for optical flow. Switch datasets/models via config only.

## Quick Start
- Prepare CSV manifests: `img1,img2,flow` (paths relative to dataset root are recommended).
- Edit `configs/animerun_baseline.yaml` (dataset root, csvs, resize).
- Train: `python train.py --config configs/animerun_baseline.yaml`
- Eval: `python eval.py --config configs/animerun_baseline.yaml trainer.ckpt_path=/path/to/best.pth`

## Dataset Conventions
We assume flow files are one of: `.flo`, `.pfm`, `.npy`, or KITTI-encoded `.png`. Resize logic scales u/v accordingly.

## Extend
- New dataset: add `dataio/<name>.py` and register in `dataio/registry.py`.
- New model: drop a file in `models/`, register it in `models/registry.py`, update config.
- New loss or metric: add under `losses/` or `metrics/` and call from trainer.

## Notes
- This baseline model is intentionally small to validate the pipeline. Swap for a stronger model once the data/IO path is stable.