#!/usr/bin/env python
"""
Debug script to verify UnSAMFlow eval pipeline works end-to-end.
Run this BEFORE starting training to ensure eval will work.

Usage:
    python scripts/debug_unsamflow_eval.py --config configs/train_unsup_animerun_unsamflow.yaml
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataio.clip_dataset_unsup import UnlabeledClipDataset
from torch.utils.data import DataLoader

from models.UnSAMFlow.models.pwclite import PWCLite


def print_sep(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check_dataset(cfg):
    """Check dataset loading for both train and test modes."""
    print_sep("1. Dataset Loading Check")
    
    data_cfg = cfg.get("data", {})
    root = data_cfg.get("train_root", "data/AnimeRun_v2/")
    
    # Test dataset (for eval)
    print("\n[Test Dataset]")
    crop_size = data_cfg.get("crop_size", [384, 768])
    test_ds = UnlabeledClipDataset(
        root=root,
        T=3,
        crop_size=(crop_size[0], crop_size[-1]),
        load_sam_masks=data_cfg.get("load_sam_masks", False),
        sam_mask_root=data_cfg.get("sam_mask_dir", None),
        is_test=True,
    )


    print(f"  Sequences: {len(test_ds.test_seqs) if hasattr(test_ds, 'test_seqs') else 'N/A'}")
    print(f"  Total samples: {len(test_ds)}")
    print(f"  is_test: {test_ds.is_test}")
    
    # Get one sample
    sample = test_ds[0]
    print(f"\n[Sample Contents]")
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: Tensor shape={v.shape}, dtype={v.dtype}")
        elif isinstance(v, list):
            if len(v) > 0 and isinstance(v[0], torch.Tensor):
                print(f"  {k}: List[Tensor] len={len(v)}, each shape={v[0].shape}")
            else:
                print(f"  {k}: List len={len(v)}")
        else:
            print(f"  {k}: {type(v).__name__}")
    
    # Check if flow_list exists for test
    if "flow_list" in sample:
        print(f"\n  ✅ GT flow found! Length: {len(sample['flow_list'])}")
        for i, f in enumerate(sample["flow_list"]):
            print(f"     flow[{i}]: shape={f.shape}, mean={f.abs().mean():.4f}, max={f.abs().max():.4f}")
    else:
        print(f"\n  ❌ ERROR: flow_list not found in test sample!")
        return False
    
    return True


def check_model(cfg):
    """Check model can forward pass."""
    print_sep("2. Model Forward Pass Check")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    
    # Build model with config that supports 'in' operator
    from argparse import Namespace
    mcfg = cfg.get("model", {}).get("args", {})
    
    # Create namespace with __contains__ to support 'in' checks
    class ConfigNamespace:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def __contains__(self, key):
            return hasattr(self, key)
    
    # PWCLite requires many config attrs - add all with defaults
    model_cfg = ConfigNamespace(
        reduce_dense=mcfg.get("reduce_dense", True),
        input_boundary=mcfg.get("input_boundary", False),
        input_adj_map=mcfg.get("input_adj_map", False),
        add_mask_corr=mcfg.get("add_mask_corr", False),
        mask_corr_agg=mcfg.get("mask_corr_agg", "simple"),
        learned_upsampler=mcfg.get("learned_upsampler", False),
        aggregation_type=mcfg.get("aggregation_type", "residual"),
    )
    model = PWCLite(model_cfg).to(device)


    model.eval()
    print(f"  Model: PWCLite loaded")
    
    # Dummy input - use H=384 to align with PWCLite pyramid (368/16=23 doesn't work)
    B, H, W = 2, 384, 768
    img1 = torch.randn(B, 3, H, W, device=device)
    img2 = torch.randn(B, 3, H, W, device=device)

    
    # If input_boundary is True, we need segment masks
    seg1, seg2 = None, None
    if model_cfg.input_boundary:
        seg1 = torch.randint(0, 10, (B, 1, H, W), device=device)
        seg2 = torch.randint(0, 10, (B, 1, H, W), device=device)
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            res = model(img1, img2, full_seg1=seg1, full_seg2=seg2, with_bk=False)

    
    print(f"\n[Model Output]")
    for k, v in res.items():
        if isinstance(v, list):
            print(f"  {k}: List len={len(v)}")
            for i, x in enumerate(v[:2]):  # Show first 2
                print(f"    [{i}]: shape={x.shape}, mean={x.abs().mean():.4f}")
        elif isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}")
    
    # Check flows_12 exists
    if "flows_12" in res:
        flow = res["flows_12"][0]
        print(f"\n  ✅ Forward flow found! shape={flow.shape}")
    else:
        print(f"\n  ❌ ERROR: flows_12 not in model output!")
        return False
    
    return True


def check_eval_loop(cfg):
    """Check full eval loop with real data."""
    print_sep("3. Full Eval Loop Check")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = cfg.get("data", {})
    
    # Build test dataset directly
    print("\n[Building Test Loader]")
    root = data_cfg.get("train_root", "data/AnimeRun_v2/")
    crop_size = data_cfg.get("crop_size", [384, 768])  # Must match PWCLite pyramid (divisible by 16)

    
    test_ds = UnlabeledClipDataset(
        root=root,
        T=3,
        crop_size=(crop_size[0], crop_size[-1]),
        load_sam_masks=data_cfg.get("load_sam_masks", False),
        sam_mask_root=data_cfg.get("sam_mask_dir", None),
        is_test=True,
    )

    
    val_loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=0)
    print(f"  Val loader batches: {len(val_loader)}")

    
    # Build model
    class ConfigNamespace:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def __contains__(self, key):
            return hasattr(self, key)
    
    mcfg = cfg.get("model", {}).get("args", {})
    model_cfg = ConfigNamespace(
        reduce_dense=mcfg.get("reduce_dense", True),
        input_boundary=mcfg.get("input_boundary", False),
        input_adj_map=mcfg.get("input_adj_map", False),
        add_mask_corr=mcfg.get("add_mask_corr", False),
        mask_corr_agg=mcfg.get("mask_corr_agg", "simple"),
        learned_upsampler=mcfg.get("learned_upsampler", False),
        aggregation_type=mcfg.get("aggregation_type", "residual"),
    )

    model = PWCLite(model_cfg).to(device)
    model.eval()

    
    # Run one batch
    print("\n[Running Eval on One Batch]")
    batch = next(iter(val_loader))
    
    # Move to device - dataset returns 'clip' not 'frames'
    frames = batch["clip"].to(device)
    print(f"  frames: {frames.shape}")
    
    img1 = frames[:, 0]
    img2 = frames[:, 1]
    
    # Get segment masks if available and needed
    seg1, seg2 = None, None
    if model_cfg.input_boundary:
        if "sam_masks" in batch:
            masks = batch["sam_masks"].to(device)
            seg1 = masks[:, 0].unsqueeze(1)
            seg2 = masks[:, 1].unsqueeze(1)
        else:
            # Create dummy segment masks
            print("  Warning: input_boundary=True but no SAM masks, using dummy")
            B, _, H, W = img1.shape
            seg1 = torch.randint(0, 10, (B, 1, H, W), device=device)
            seg2 = torch.randint(0, 10, (B, 1, H, W), device=device)
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            res = model(img1, img2, full_seg1=seg1, full_seg2=seg2, with_bk=False)
    
    flow_pred = res["flows_12"][0]
    print(f"  flow_pred: {flow_pred.shape}, mean={flow_pred.abs().mean():.4f}")

    
    # Check GT
    if "flow_list" not in batch:
        print("  ❌ ERROR: flow_list not in batch!")
        return False
    
    gt = batch["flow_list"][0].to(device)
    print(f"  GT flow: {gt.shape}, mean={gt.abs().mean():.4f}")
    
    # Resize and compute EPE
    if flow_pred.shape[-2:] != gt.shape[-2:]:
        flow_pred_up = F.interpolate(flow_pred, size=gt.shape[-2:], mode='bilinear', align_corners=True)
        # Scale flow
        scale_x = gt.shape[-1] / img1.shape[-1]
        scale_y = gt.shape[-2] / img1.shape[-2]
        flow_pred_up[:, 0] *= scale_x
        flow_pred_up[:, 1] *= scale_y
        print(f"  Upsampled flow: {flow_pred_up.shape}, scale=({scale_x:.2f}, {scale_y:.2f})")
    else:
        flow_pred_up = flow_pred
    
    # EPE
    epe = torch.norm(flow_pred_up - gt, dim=1).mean()
    print(f"\n  ✅ EPE computed: {epe.item():.4f}")
    
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_unsup_animerun_unsamflow.yaml")
    args = parser.parse_args()
    
    print(f"\n🔍 UnSAMFlow Eval Pipeline Debug")
    print(f"   Config: {args.config}")
    
    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    success = True
    
    # Run checks
    if not check_dataset(cfg):
        success = False
        print("\n❌ Dataset check FAILED!")
    
    if not check_model(cfg):
        success = False
        print("\n❌ Model check FAILED!")
    
    if not check_eval_loop(cfg):
        success = False
        print("\n❌ Eval loop check FAILED!")
    
    print_sep("Summary")
    if success:
        print("✅ All checks passed! You can start training.")
    else:
        print("❌ Some checks failed. Fix issues before training.")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
