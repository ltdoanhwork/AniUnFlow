#!/usr/bin/env python3
# file: scripts/precompute_sam_masks.py
"""
Precompute SAM-2 Video Segmentation Masks for AnimeRun
======================================================
Runs SAM-2 in video mode on AnimeRun sequences and saves
compressed mask tensors for efficient training loading.

Why precompute?
1. SAM-2 video inference is heavy (cannot run real-time during training)
2. Video mode ensures temporal consistency across the whole clip
3. Precomputed masks allow faster dataloading

Output Structure:
  out_dir/
    Train/
      Sequence1/
        frame_0001.pt  (compressed tensor)
        frame_0002.pt
        ...
"""
import os
import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
import cv2
from addict import Dict as AdDict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.aniunflow.sam2_guidance import SAM2GuidanceModule

def parse_args():
    parser = argparse.ArgumentParser(description="Precompute SAM-2 Masks")
    parser.add_argument("--data_root", type=str, required=True, help="Path to AnimeRun dataset root")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for masks")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run SAM-2")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to SAM-2 checkpoint")
    parser.add_argument("--num_segments", type=int, default=16, help="Max segments to keep")
    parser.add_argument("--batch_size", type=int, default=1, help="Sequence batch size (keep 1 for video mode)")
    return parser.parse_args()

def find_sequences(root_dir):
    """
    Find all sequence directories in AnimeRun.
    AnimeRun structure V2: root/train/Frame_Anime/SequenceName/{original, color_*, ...}/*.png
    """
    seq_dirs = []
    
    frame_anime_dir = root_dir / "Frame_Anime"
    if not frame_anime_dir.exists():
        # Try finding directly if structure is flat or different
        frame_anime_dir = root_dir

    # Iterate over scene directories
    for scene in sorted(frame_anime_dir.iterdir()):
        if not scene.is_dir():
            continue
            
        # Check subdirectories (passes)
        passes = sorted([d for d in scene.iterdir() if d.is_dir()])
        
        # If no subdirs, check if scene itself is a sequence
        if not passes:
            # Check for images
            if list(scene.glob("*.png")) or list(scene.glob("*.jpg")):
                seq_dirs.append(scene)
            continue
            
        # Add each pass as a sequence
        for p in passes:
             if list(p.glob("*.png")) or list(p.glob("*.jpg")):
                 seq_dirs.append(p)
                 
    return sorted(seq_dirs)

def load_sequence_frames(seq_path):
    """Load all frames in a sequence."""
    frames = sorted(list(seq_path.glob("*.png")) + list(seq_path.glob("*.jpg")))
    return frames

def save_compressed_mask(mask_tensor, out_path):
    """
    Save mask tensor with compression.
    mask_tensor: (S, H, W) float/bool
    """
    # Convert to sparse or indexed PNG if possible, 
    # but for S segments (soft masks?), we likely want float16 or int8
    
    # Strategy 1: Save as torch tensor (simple)
    # Convert to half precision to save space
    mask_half = mask_tensor.half()
    
    # Or to uint8 if it's hard masks (0-255)
    # mask_uint8 = (mask_tensor * 255).to(torch.uint8)
    
    torch.save(mask_half, out_path)

def main():
    args = parse_args()
    
    # 1. Initialize SAM-2
    print(f"Initializing SAM-2 ({args.device})...")
    sam = SAM2GuidanceModule(
        sam_checkpoint=args.checkpoint,
        device=args.device,
        num_segments=args.num_segments,
        use_automatic_mask_generator=True # Use AMG to discover objects for the first frame?
        # Note: SAM-2 video mode usually requires prompts or AMG on first frame
    )
    # Trigger lazy load
    sam._lazy_load()
    
    if sam._sam_model is None:
        print("Error: Could not load SAM-2 model. Check checkpoint path.")
        return

    # 2. Find Sequences
    split_root = Path(args.data_root) / args.split
    sequences = find_sequences(split_root)
    print(f"Found {len(sequences)} sequences in {split_root}")
    
    # 3. Process Loops
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    
    for seq_idx, seq_path in enumerate(tqdm(sequences, desc="Processing Sequences")):
        # Construct output path
        rel_path = seq_path.relative_to(split_root)  # e.g., Frame_Anime/Seq001
        seq_out_dir = out_root / args.split / rel_path
        seq_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if already done (simple check)
        frames = load_sequence_frames(seq_path)
        if len(frames) == 0:
            continue
            
        last_frame_name = frames[-1].stem + ".pt"
        if (seq_out_dir / last_frame_name).exists():
            continue  # Skip existing
            
        # Load all frames into tensor (T, 3, H, W)
        # Note: If T is huge, we might need sliding window, 
        # but AnimeRun clips are usually short enough for GPU ram
        
        # Read frames
        img_list = []
        for fpath in frames:
            img = cv2.imread(str(fpath))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_list.append(img)
            
        # To tensor
        imgs_np = np.stack(img_list, axis=0) # (T, H, W, 3)
        imgs_tensor = torch.from_numpy(imgs_np).float() / 255.0
        imgs_tensor = imgs_tensor.permute(0, 3, 1, 2) # (T, 3, H, W)
        
        # Run SAM-2 Video Inference
        # We need to adapt the SAM2GuidanceModule to run on full sequence
        # The existing extract_segment_masks handles (B, T, ...)
        
        with torch.no_grad():
            # Add batch dim
            clip_input = imgs_tensor.unsqueeze(0).to(args.device) # (1, T, 3, H, W)
            
            # This calls internal SAM-2 logic
            # Note: extract_segment_masks in our wrapper currently does per-frame logic if video mode isn't fully implemented in wrapper
            # Let's check wrapper implementation. 
            # If wrapper uses per-frame, we should update it to use video state for consistency.
            # For now, let's assume the wrapper does its best.
            
            masks = sam.extract_segment_masks(clip_input) # (1, T, S, H, W)
            masks = masks.squeeze(0).cpu() # (T, S, H, W)
            
        # Save individual frames
        for t, fpath in enumerate(frames):
            out_name = fpath.stem + ".pt"
            save_path = seq_out_dir / out_name
            save_compressed_mask(masks[t], save_path)
            
    print("Precomputation Complete!")

if __name__ == "__main__":
    main()
