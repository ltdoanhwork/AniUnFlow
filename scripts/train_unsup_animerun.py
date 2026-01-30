#!/usr/bin/env python3
# file: scripts/train_unsup_animerun.py
"""
Segment-Aware Unsupervised Optical Flow Training on AnimeRun-2D
================================================================

Single entry point for training AniFlowFormer-T with SAM-2 segment guidance.

Usage:
    # Full training with default config
    python scripts/train_unsup_animerun.py --config configs/train_unsup_animerun_sam.yaml
    
    # Ablation: disable SAM guidance
    python scripts/train_unsup_animerun.py --config configs/train_unsup_animerun_sam.yaml \
        --set sam.enabled=false
    
    # Ablation: disable segment consistency loss
    python scripts/train_unsup_animerun.py --config configs/train_unsup_animerun_sam.yaml \
        --set loss.segment_consistency.enabled=false
    
    # Quick test run
    python scripts/train_unsup_animerun.py --config configs/train_unsup_animerun_sam.yaml \
        --set optim.epochs=2 data.batch_size=2 --debug

Features:
    - Modular loss components (all toggleable)
    - SAM-2 segment guidance (optional)
    - AnimeRun-compatible validation metrics
    - TensorBoard logging
    - Multi-GPU support
"""
from __future__ import annotations
import os
import sys
import argparse
import random
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Segment-Aware Unsupervised Optical Flow Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/train_unsup_animerun_sam.yaml",
        help="Path to config YAML file"
    )
    
    parser.add_argument(
        "--set", "-s",
        action="append",
        default=[],
        help="Override config values (e.g., --set optim.lr=1e-4)"
    )
    
    parser.add_argument(
        "--workspace", "-w",
        type=str,
        default=None,
        help="Override workspace directory"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (anomaly detection, verbose logging)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Initialize everything but don't train (for testing)"
    )
    
    # Distributed training
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def override_config(cfg: Dict[str, Any], overrides: list) -> Dict[str, Any]:
    """Apply command-line overrides to config."""
    for override in overrides:
        if "=" not in override:
            print(f"Warning: Invalid override format: {override}")
            continue
        
        key, value = override.split("=", 1)
        keys = key.split(".")
        
        # Parse value type
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        elif value.lower() == "null" or value.lower() == "none":
            value = None
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass  # Keep as string
        
        # Navigate to nested key
        d = cfg
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        
        d[keys[-1]] = value
        print(f"Config override: {key} = {value}")
    
    return cfg


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_distributed():
    """Setup distributed training if available."""
    if torch.cuda.is_available():
        # Check for distributed environment
        if "WORLD_SIZE" in os.environ:
            world_size = int(os.environ["WORLD_SIZE"])
            rank = int(os.environ.get("RANK", 0))
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            
            torch.distributed.init_process_group(backend="nccl")
            torch.cuda.set_device(local_rank)
            
            return world_size, rank, local_rank
    
    return 1, 0, 0


def main():
    """Main training entry point."""
    args = parse_args()
    
    # Load and merge config
    print(f"Loading config: {args.config}")
    cfg = load_config(args.config)
    cfg = override_config(cfg, args.set)
    
    # Override workspace if specified
    if args.workspace:
        cfg["workspace"] = args.workspace
    
    # Debug mode
    if args.debug:
        cfg["debug"] = cfg.get("debug", {})
        cfg["debug"]["detect_anomaly"] = True
        torch.autograd.set_detect_anomaly(True)
        print("Debug mode enabled")
    
    # Set seed
    seed = cfg.get("seed", 1337)
    set_seed(seed)
    print(f"Random seed: {seed}")
    
    # Setup CUDNN
    cudnn.benchmark = True
    
    # Setup workspace
    workspace = Path(cfg["workspace"])
    workspace.mkdir(parents=True, exist_ok=True)
    
    # Save config to workspace
    config_save_path = workspace / "config.yaml"
    with open(config_save_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"Config saved to: {config_save_path}")
    
    # Also save as JSON for easy parsing
    with open(workspace / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    
    # Import training components
    from dataio import UnlabeledClipDataset
    from torch.utils.data import DataLoader
    
    # Build datasets
    print("\n=== Building Datasets ===")
    data_cfg = cfg["data"]
    
    train_ds = UnlabeledClipDataset(
        root=data_cfg["train_root"],
        T=data_cfg.get("T", 5),
        stride_min=data_cfg.get("stride_min", 1),
        stride_max=data_cfg.get("stride_max", 2),
        crop_size=tuple(data_cfg.get("crop_size", [368, 768])),
        color_jitter=tuple(data_cfg.get("color_jitter", [0.3, 0.3, 0.3, 0.1])),
        do_flip=data_cfg.get("do_flip", True),
        grayscale_p=data_cfg.get("grayscale_p", 0.1),
        resize=data_cfg.get("resize", True),
        keep_aspect=data_cfg.get("keep_aspect", True),
        pad_mode=data_cfg.get("pad_mode", "reflect"),
        is_test=False,
        load_sam_masks=data_cfg.get("load_sam_masks", False),
        sam_mask_root=data_cfg.get("sam_mask_dir", None),
    )
    
    val_ds = UnlabeledClipDataset(
        root=data_cfg.get("val_root", data_cfg["train_root"]),
        T=data_cfg.get("T", 5),
        stride_min=1,
        stride_max=1,
        crop_size=tuple(data_cfg.get("crop_size", [368, 768])),
        color_jitter=None,
        do_flip=False,
        grayscale_p=0.0,
        resize=data_cfg.get("resize", True),
        keep_aspect=data_cfg.get("keep_aspect", True),
        pad_mode=data_cfg.get("pad_mode", "reflect"),
        is_test=True,
        load_sam_masks=False,  # Don't load masks for validation (computed on-the-fly if needed)
        sam_mask_root=None,
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg.get("batch_size", 4),
        shuffle=data_cfg.get("shuffle", True),
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=data_cfg.get("drop_last", True),
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, data_cfg.get("batch_size", 4) // 2),
        shuffle=False,
        num_workers=max(2, data_cfg.get("num_workers", 4) // 2),
        pin_memory=True,
        drop_last=False,
    )
    
    print(f"Train dataset: {len(train_ds)} clips")
    print(f"Val dataset: {len(val_ds)} clips")
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    
    # Dry run check
    if args.dry_run:
        print("\n=== Dry Run: Testing One Batch ===")
        batch = next(iter(train_loader))
        print(f"Batch clip shape: {batch['clip'].shape}")
        
        # Import and build trainer
        from engine.trainer_segment_aware import SegmentAwareTrainer
        trainer = SegmentAwareTrainer(cfg, workspace)
        
        # Test forward pass
        device = trainer.device
        clip = batch["clip"].to(device)
        if clip.dtype == torch.uint8:
            clip = clip.float() / 255.0
        
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            out = trainer.model(clip)
            print(f"Model output keys: {out.keys()}")
            print(f"Flows: {len(out['flows'])} predictions")
            if out['flows']:
                print(f"Flow shape: {out['flows'][0].shape}")
        
        print("\nDry run completed successfully!")
        return
    
    # Build trainer and start training
    print("\n=== Initializing Trainer ===")
    from engine.trainer_segment_aware import SegmentAwareTrainer
    
    trainer = SegmentAwareTrainer(cfg, workspace)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    print("\n=== Starting Training ===")
    print(f"Epochs: {cfg['optim']['epochs']}")
    print(f"Batch size: {data_cfg.get('batch_size', 4)}")
    print(f"Learning rate: {cfg['optim']['lr']}")
    print(f"Workspace: {workspace}")
    print(f"TensorBoard: {workspace / cfg.get('logging', {}).get('tb_dir', 'tb')}")
    print("=" * 50)
    
    trainer.fit(train_loader, val_loader)
    
    print("\n=== Training Complete ===")
    print(f"Best metric: {trainer.best_metric:.4f}")
    print(f"Checkpoints saved to: {workspace}")


if __name__ == "__main__":
    main()
