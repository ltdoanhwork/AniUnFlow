import argparse
import yaml
from pathlib import Path
import sys
import os
# Add root to path for dataio
sys.path.append(os.getcwd())
# Add models/UnSAMFlow to path so its internal imports work
sys.path.append(os.path.join(os.getcwd(), "models/UnSAMFlow"))

import torch
from torch.utils.data import DataLoader
from dataio.clip_dataset_unsup import UnlabeledClipDataset
from engine.trainer_unsamflow import UnSAMFlowTrainer

def build_dataset(cfg, is_test=False):
    # Pass 'sam_mask' specific args if present in cfg
    load_sam = cfg.get("load_sam_masks", False)
    sam_root = cfg.get("sam_mask_root", None)
    
    return UnlabeledClipDataset(
        root=cfg["root"],
        T=cfg.get("clip_len", 2),
        stride_min=1,
        stride_max=1,
        crop_size=tuple(cfg.get("crop_size", [384, 768])),
        color_jitter=cfg.get("color_jitter", None),
        do_flip=cfg.get("do_flip", False),
        grayscale_p=cfg.get("grayscale_p", 0.0),
        resize=cfg.get("resize", True),
        keep_aspect=cfg.get("keep_aspect", True),
        pad_mode=cfg.get("pad_mode", "reflect"),
        is_test=is_test,
        load_sam_masks=load_sam,
        sam_mask_root=sam_root,
    )

def build_loader(ds, cfg):
    return DataLoader(
        ds,
        batch_size=cfg.get("batch_size", 4),
        shuffle=cfg.get("shuffle", True),
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=cfg.get("drop_last", True),
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
        
    trainer = UnSAMFlowTrainer(cfg, workspace=Path(cfg["workspace"]))
    
    if args.dry_run:
        print("Dry run: Checking dataset...")
        t_ds = build_dataset(cfg["data"]["train"], is_test=False)
        print(f"[Train] Found {len(t_ds)} sequences")
        
        t_loader = build_loader(t_ds, cfg["data"]["train"])
        batch = next(iter(t_loader))
        clip = batch["clip"]
        print(f"Batch shape: {clip.shape}")
        
        if "sam_masks" in batch:
             print(f"SAM Masks shape: {batch['sam_masks'].shape}")
             print(f"SAM Masks min: {batch['sam_masks'].min().item():.4f}, max: {batch['sam_masks'].max().item():.4f}")
        else:
             print("WARNING: No SAM masks found in batch. UnSAMFlow requires masks.")
        
        img1 = clip[:, 0].to(trainer.device)
        img2 = clip[:, 1].to(trainer.device)
        if img1.dtype == torch.uint8:
            img1 = img1.float() / 255.0
            img2 = img2.float() / 255.0

        print(f"Image min: {img1.min().item():.4f}, max: {img1.max().item():.4f}, mean: {img1.mean().item():.4f}")
        
        # Test Forward
        trainer.model.eval()
        
        seg1, seg2 = None, None
        if "sam_masks" in batch:
            masks = batch["sam_masks"].to(trainer.device)
            # Adapt shape if needed
            if masks.ndim == 5 and masks.shape[2]==1: masks = masks.squeeze(2)
            seg1 = masks[:, 0].unsqueeze(1).float()
            seg2 = masks[:, 1].unsqueeze(1).float()
            
        with torch.no_grad():
             res = trainer.model(img1, img2, full_seg1=seg1, full_seg2=seg2, with_bk=True)
             
             flows_12 = res["flows_12"]
             flows_21 = res["flows_21"]
             
             combined_flows = [torch.cat([f12, f21], dim=1) for f12, f21 in zip(flows_12, flows_21)]
             loss_pack = trainer.criterion(combined_flows, img1, img2, full_seg1=seg1, full_seg2=seg2, occ_aware=False)
             
             print(f"Dry run loss (occ_aware=False): {loss_pack[0].item():.6f}")
             
        print("Model forward success.")
        if "flows_12" in res:
             print(f"Flow output shape: {res['flows_12'][0].shape}")
        return

    # Train
    train_ds = build_dataset(cfg["data"]["train"], is_test=False)
    val_ds = build_dataset(cfg["data"]["val"], is_test=True)
    
    train_loader = build_loader(train_ds, cfg["data"]["train"])
    val_loader = build_loader(val_ds, cfg["data"]["val"])
    
    trainer.fit(train_loader, val_loader)

if __name__ == "__main__":
    main()
