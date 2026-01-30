#!/usr/bin/env python3
# file: scripts/train_mdflow.py
"""
Single entry point for training MDFlow (FastFlowNet) on AnimeRun.
Uses dedicated engine/trainer_mdflow.py.
"""
import os
import sys
import argparse
import random
import yaml
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.trainer_mdflow import MDFlowTrainer
from dataio import UnlabeledClipDataset
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--workspace", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    cfg = load_config(args.config)
    
    if args.workspace:
        cfg["workspace"] = args.workspace
        
    workspace = Path(cfg["workspace"])
    
    # Seeding
    seed = cfg.get("seed", 1337)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Datasets
    data_cfg = cfg["data"]
    train_ds = UnlabeledClipDataset(
        root=data_cfg["train_root"],
        T=data_cfg.get("T", 3),
        stride_min=data_cfg.get("stride_min", 1),
        stride_max=data_cfg.get("stride_max", 2),
        crop_size=tuple(data_cfg.get("crop_size", [368, 768])),
        # ... simplified args for brevity, assumes config matches ...
    )
    
    # Basic validation dataset
    val_ds = UnlabeledClipDataset(
        root=data_cfg["val_root"],
        T=3,
        stride_min=1, stride_max=1,
        is_test=True
    )
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=data_cfg.get("batch_size", 4),
        shuffle=True, 
        num_workers=4, 
        drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=data_cfg.get("batch_size", 4), shuffle=False)
    
    # Dry run check
    if args.dry_run:
        print("Dry run: Checking dataset...")
        try:
            batch = next(iter(train_loader))
            print(f"Batch shape: {batch['clip'].shape}")
            trainer = MDFlowTrainer(cfg, workspace)
            # Test model forward
            clip = batch["clip"].to(trainer.device).float() / 255.0
            imgs = torch.cat([clip[:,0], clip[:,1]], dim=1)
            out = trainer.model(imgs)
            print("Model forward success")
        except Exception as e:
            print(f"Dry run failed: {e}")
            raise e
        return

    # Train
    trainer = MDFlowTrainer(cfg, workspace)
    trainer.fit(train_loader, val_loader)

if __name__ == "__main__":
    main()
