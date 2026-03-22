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

```
models/
├── __init__.py
├── aniflowformer_t/
│ ├── __init__.py
│ ├── model.py # High-level model wrapper (AniFlowFormerT)
│ ├── encoder.py # PyramidEncoder + blocks
│ ├── tokenizer.py # CostTokenizer (local correlation → tokens)
│ ├── lcm.py # LatentCostMemory (temporal Transformer)
│ ├── gtr.py # GlobalTemporalRegressor (temporal aggregator)
│ ├── decoder.py # Multi-Scale Recurrent Decoder (coarse→fine)
│ ├── occlusion.py # Optional occlusion head
│ ├── sam_adapter.py # Optional SAM guidance adapter
│ ├── losses.py # Photometric, smoothness, temporal, cycle
│ └── utils.py # warp, SSIM, image gradients, grids
```


66: 
67: ## Global Matching V3 (New)
68: 
69: A significant upgrade introducing **SAM-2 structural guidance** for improved boundary handling and flat region consistency.
70: 
71: - **Documentation**: [docs/architecture_v3.md](docs/architecture_v3.md)
72: - **Config**: `configs/train_unsup_animerun_sam_v3.yaml`
73: - **Mask Precomputation**: `scripts/precompute_sam_masks.py`
74: 
75: ```bash
76: # Train V3
77: python scripts/train_unsup_animerun.py --config configs/train_unsup_animerun_sam_v3.yaml
78: ```
79: 
80: ---

## Parallel AnimeRun Research Branches

Two large unsupervised directions now live side by side:

- `V4 Hybrid SAM`: dense token matching first, SAM guidance and iterative refinement second.
- `V5 Object Memory`: SAM object slots first, affine/layered object motion first, dense residual correction second.
- `V5.1 Object Memory Dense`: V5 object memory plus dense correlation and multi-scale dense refinement for large motion.
- `V5.2 Object Memory Global`: V5.1 plus a lightweight coarse global matcher borrowed from the V6 analysis, without V6's heavier visibility and non-rigid branches.
- `V5.3 Deformable Object Memory Global`: V5.2 plus deformable slot motion so each object can express non-rigid motion on top of the global matcher.
- `V5.3b Deformable Ramp Global`: V5.3 plus delayed deformable slot motion, confidence-gated deformation, and temporal SAM support from neighboring frames.
- `V6 Global Slot Search`: object memory plus non-rigid slot flow, coarse global large-motion search, visibility-aware compositing, and staged dense refinement.

V5 documentation:

- [docs/architecture_v5_object_memory.md](docs/architecture_v5_object_memory.md)
- [docs/architecture_v5_1_object_memory_dense.md](docs/architecture_v5_1_object_memory_dense.md)
- [docs/architecture_v5_2_object_memory_global.md](docs/architecture_v5_2_object_memory_global.md)
- [docs/architecture_v5_3_object_memory_deformable_global.md](docs/architecture_v5_3_object_memory_deformable_global.md)
- [docs/architecture_v5_3b_object_memory_deformable_global.md](docs/architecture_v5_3b_object_memory_deformable_global.md)
- [docs/architecture_v6_global_slot_search.md](docs/architecture_v6_global_slot_search.md)
- Main config: `configs/v5_object_memory_sam_parallel.yaml`
- Dense follow-up config: `configs/v5_1_object_memory_dense_parallel.yaml`
- V6 main config: `configs/v6_global_slot_search_parallel.yaml`

```bash
# Train V5 object-memory branch
python scripts/train_unsup_animerun.py --config configs/v5_object_memory_sam_parallel.yaml

# Train V5.1 object-memory dense branch
python scripts/train_unsup_animerun.py --config configs/v5_1_object_memory_dense_parallel.yaml

# Train V5.2 object-memory global branch
python scripts/train_unsup_animerun.py --config configs/v5_2_object_memory_global.yaml

# Train V5.3 deformable object-memory global branch
python scripts/train_unsup_animerun.py --config configs/v5_3_object_memory_deformable_global.yaml

# Train V5.3b delayed deformable object-memory global branch
python scripts/train_unsup_animerun.py --config configs/v5_3b_object_memory_deformable_global.yaml

# Train V6 global-slot-search branch
python scripts/train_unsup_animerun.py --config configs/v6_global_slot_search_parallel.yaml

# Fine-tune V6 from a saved best checkpoint
python scripts/train_unsup_animerun.py \
  --config configs/v6_global_slot_search_finetune.yaml \
  --resume workspaces/v6_global_slot_search_parallel/best.pth

# Collect comparable CSV + Markdown reports
python scripts/collect_animerun_results.py
```

V5 ablations:

- `configs/v5_object_memory_sam_no_layer_order.yaml`
- `configs/v5_object_memory_sam_no_temporal_memory.yaml`
- `configs/v5_object_memory_sam_affine_only.yaml`

V6 ablations:

- `configs/v6_no_global_search.yaml`
- `configs/v6_affine_only_slots.yaml`
- `configs/v6_no_visibility_compositing.yaml`
- `configs/v6_no_hard_motion_reweight.yaml`
