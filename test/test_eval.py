# file: main.py
# Run:
#   python main.py --data_root data/AnimeRun_v2 --workspace outputs/aft_smoke --batch_size 2 --T 5
# Just check dataset shapes (no model forward):
#   python main.py --data_root data/AnimeRun_v2 --dry-run-dataset

from __future__ import annotations
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Adjust these imports to your project layout
from dataio import UnlabeledClipDataset  # the dataset you just wrote
from engine import UnsupervisedClipTrainer        # your trainer class

def build_cfg(args: argparse.Namespace) -> dict:
    """Minimal training/val config for the UnsupervisedClipTrainer."""
    cfg = {
        "model": {
            "name": "aniflowformer-t",
            # If your AFConfig has required fields, put them under args below.
            # Otherwise leave empty to use defaults.
            "args": {
                "use_sam": False
            },
        },
        "optim": {
            "seed": 1337,
            "epochs": 1,              # just one epoch for smoke test
            "lr": 2e-4,
            "weight_decay": 1e-4,
            "clip": 1.0,
            "accum_steps": 1,
            "scheduler": {"type": "cosine", "per_batch": True, "min_lr": 1e-6},
        },
        "loss": {
            "w_epe_sup": 0.0  # keep unsupervised; val uses GT only for metrics
        },
        "logging": {
            "use_tb": False,  # turn on if you want TensorBoard
            "log_every": 50,
            "tb_dir": "tb",
        },
        "viz": {
            "enable": False,
            "max_samples": 4,
            "save_dir": "val_vis",
        },
        "ckpt": {"save_every": 1},
    }
    return cfg

def build_val_loader(args: argparse.Namespace) -> DataLoader:
    """Validation loader that reads GT flows from test/Flow/..."""
    val_ds = UnlabeledClipDataset(
        root=args.data_root,
        T=args.T,
        is_test=True,           # critical: reads GT from test/Flow
        resize=not args.no_resize,
        keep_aspect=args.keep_aspect,
        pad_mode="reflect",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return val_loader

def dry_run_dataset(args: argparse.Namespace) -> None:
    """Print a couple of samples to verify shapes & GT wiring."""
    from itertools import islice
    val_loader = build_val_loader(args)
    print(f"[DryRun] B={args.batch_size}, T={args.T}, len(val)={len(val_loader.dataset)}")
    
    for b in islice(val_loader, 1):
        clip = b["clip"]             # (B, T, 3, H, W)
        flow_list = b.get("flow_list", None)  # list of length T-1; each tensor (B, 2, H, W)
        print("clip:", tuple(clip.shape))
        if flow_list is None:
            print("flow_list: None (unexpected for is_test=True)")
        else:
            print("flow_list length:", len(flow_list))
            print("flow_list[0] shape:", tuple(flow_list[0].shape))
    print("[DryRun] OK.")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True, help="Path to AnimeRun_v2 folder")
    p.add_argument("--workspace", type=str, default="outputs/aft_smoke", help="Folder for logs/ckpts")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--T", type=int, default=5, help="Odd number, e.g., 5 or 7")
    p.add_argument("--no_resize", action="store_true", help="Use original resolution (must be consistent)")
    p.add_argument("--keep_aspect", action="store_true", help="Keep aspect and pad instead of direct resize")
    p.add_argument("--dry-run-dataset", action="store_true", help="Only check dataset shapes, no model forward")
    args = p.parse_args()

    # Quick dataset sanity check
    if args.dry_run_dataset:
        dry_run_dataset(args)
        return

    cfg = build_cfg(args)
    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    # Build val loader (with GT)
    val_loader = build_val_loader(args)

    # Trainer init (model + optimizer + logging)
    # Note: if your AFConfig requires specific args, set them in build_cfg().
    trainer = UnsupervisedClipTrainer(args, cfg, workspace)

    # Single pass validation (your validate() currently stops after 1 batch).
    metrics = trainer.validate(val_loader)
    print("[VAL] metrics:", metrics)

    # If you want to ensure full val-set evaluation:
    #   - open your trainer code and REMOVE the `break` inside validate().

if __name__ == "__main__":
    # Make CUDA deterministic if you want reproducibility:
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    main()
