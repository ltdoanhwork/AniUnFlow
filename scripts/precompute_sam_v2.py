#!/usr/bin/env python3
"""
Precompute SAM-2 Multi-Segment Masks for V4
============================================
Uses SAM-2's Automatic Mask Generator to produce proper multi-segment
integer label maps (not binary foreground/background).

Output: per-frame .pt files with shape (H, W) dtype uint8
        where each pixel has a segment label 0..N (0=background)

Usage:
    python scripts/precompute_sam_v2.py \
        --data_root data/AnimeRun_v2 \
        --out_dir data/AnimeRun_v2/SAM_Masks_v2 \
        --split train
"""
import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
import cv2
from tqdm import tqdm

# Add project root and SAM2 to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models" / "sam2"))

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def parse_args():
    p = argparse.ArgumentParser(description="Precompute SAM-2 multi-segment masks")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--split", type=str, default="train", choices=["train", "test"])
    p.add_argument("--checkpoint", type=str,
                   default="models/sam2/checkpoints/sam2.1_hiera_tiny.pt")
    p.add_argument("--config", type=str,
                   default="configs/sam2.1/sam2.1_hiera_t.yaml")
    p.add_argument("--max_segments", type=int, default=32,
                   help="Maximum number of segments to keep per frame")
    p.add_argument("--min_area", type=float, default=100,
                   help="Minimum mask area in pixels")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def find_frames(data_root: Path, split: str):
    """Find all frame images for given split."""
    base = data_root / split / "Frame_Anime"
    if not base.exists():
        base = data_root / "Frame_Anime"
    
    frames = []
    for scene in sorted(base.iterdir()):
        if not scene.is_dir():
            continue
        # Only process 'original' pass (masks are pass-independent)
        original_dir = scene / "original"
        if original_dir.exists():
            for f in sorted(original_dir.glob("*.png")):
                frames.append(f)
        else:
            # Flat structure
            for f in sorted(scene.glob("*.png")):
                frames.append(f)
    
    return frames


def masks_to_labels(masks_data, max_segments=32, min_area=100):
    """
    Convert SAM2 mask generator output to integer label map.
    
    Args:
        masks_data: list of dicts from SAM2AutomaticMaskGenerator.generate()
        max_segments: max segments to keep
        min_area: minimum mask area
    
    Returns:
        labels: (H, W) uint8 array, 0=background, 1..N=segments
    """
    if not masks_data:
        return None
    
    H, W = masks_data[0]["segmentation"].shape
    labels = np.zeros((H, W), dtype=np.uint8)
    
    # Sort by area (largest first) for consistent layering
    sorted_masks = sorted(masks_data, key=lambda x: x["area"], reverse=True)
    
    seg_id = 1
    for m in sorted_masks:
        if seg_id > max_segments:
            break
        if m["area"] < min_area:
            continue
        
        mask = m["segmentation"]  # (H, W) bool
        labels[mask] = seg_id
        seg_id += 1
    
    return labels


def main():
    args = parse_args()
    
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir) / args.split / "Frame_Anime"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Building SAM2 model from {args.checkpoint}...")
    sam2 = build_sam2(args.config, args.checkpoint, device=args.device)
    generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=32,
        points_per_batch=64,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.92,
        crop_n_layers=0,           # No multi-crop for speed
        min_mask_region_area=args.min_area,
    )
    print("SAM2 mask generator ready.")
    
    frames = find_frames(data_root, args.split)
    print(f"Found {len(frames)} frames to process.")
    
    if len(frames) == 0:
        print("No frames found! Check data_root and split.")
        return
    
    skipped = 0
    processed = 0
    
    for frame_path in tqdm(frames, desc="Generating masks"):
        # Compute output path
        # frame_path: .../Frame_Anime/scene/original/XXXX.png
        try:
            parts = list(frame_path.parts)
            idx = parts.index("Frame_Anime")
            rel_parts = parts[idx+1:]  # scene/original/XXXX.png
            out_path = out_dir / Path(*rel_parts).with_suffix(".pt")
        except (ValueError, IndexError):
            continue
        
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load image
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Generate masks
        with torch.inference_mode():
            masks_data = generator.generate(img_rgb)
        
        # Convert to label map
        labels = masks_to_labels(masks_data, args.max_segments, args.min_area)
        if labels is None:
            labels = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
        
        # Save
        torch.save(torch.from_numpy(labels), str(out_path))
        processed += 1
    
    print(f"\nDone! Processed: {processed}, Skipped: {skipped}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
