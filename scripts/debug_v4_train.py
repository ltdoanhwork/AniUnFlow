#!/usr/bin/env python3
"""
Debug script for V4 training loop.
Run 1 training step and 1 validation step.
"""
import sys
import os
import torch
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.aniunflow_v4.config import get_masks_only_config
from engine.trainer_segment_aware import SegmentAwareTrainer
from dataio.clip_dataset_unsup import UnlabeledClipDataset
from torch.utils.data import DataLoader

def collate_fn(batch):
    # Custom collate because clips can be different sizes? No, they should be cropped.
    # But let's use default collate for now.
    from torch.utils.data.dataloader import default_collate
    return default_collate(batch)

def main():
    print("=== Debug V4 Training Loop ===")
    
    # Setup config (masks only for speed, no SAM encoder needed for debug)
    cfg = get_masks_only_config()
    
    # Tiny dataset settings for debug
    cfg.data = {
        "batch_size": 2,
        "num_workers": 0,
        "train_root": "data/AnimeRun_v2",
        "val_root": "data/AnimeRun_v2",
        "load_sam_masks": True,
        "sam_mask_dir": "data/AnimeRun_v2/SAM_Masks", # Using existing masks
        "T": 3,
        "crop_size": [368, 768],
    }
    
    # Disable heavy logging
    cfg.logging = {"use_tb": False}
    cfg.viz = {"enable": False}
    
    # Initialize Trainer
    print("Initializing Trainer...")
    # Convert V4Config to dict because Trainer expects dict
    cfg_dict = cfg.to_dict()
    trainer = SegmentAwareTrainer(cfg_dict, Path("workspaces/debug_v4"))
    
    # Create dummy dataloader
    print("Creating DataLoader...")
    dataset = UnlabeledClipDataset(
        root=cfg.data["train_root"],
        split="train",
        clip_len=cfg.data["T"],
        load_sam_masks=True,
        sam_mask_root=cfg.data["sam_mask_dir"]
    )
    
    # Take subset for speed
    dataset.train_seqs = dataset.train_seqs[:2]
    dataset.samples = dataset.samples[:4]
    
    loader = DataLoader(
        dataset, 
        batch_size=2, 
        shuffle=False, 
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # Run 1 step
    print("\nRunning 1 training step...")
    accum = trainer._train_one_epoch(loader)
    print(f"Train result: {accum}")
    
    print("\n✅ Debug complete!")

if __name__ == "__main__":
    main()
