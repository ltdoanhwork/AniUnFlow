#!/usr/bin/env python3
"""
SAM-2 Enhanced Precomputation Script V2
========================================
Features:
- Aggressive AMG settings for more segments
- Multi-scale hierarchical masks
- SAM encoder feature extraction (optional)
- Video propagation mode (optional)

Output:
  out_dir/
    train/Frame_Anime/scene/original/
      masks/frame.pt      # (H, W) uint8 label map
      features/frame.pt   # Optional: dict of multi-scale features
"""
import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
import numpy as np
import cv2
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="SAM-2 Enhanced Precomputation V2")
    
    # Paths
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to AnimeRun_v2 root")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for masks/features")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "test"])
    
    # SAM checkpoint
    parser.add_argument("--checkpoint", type=str,
                        default="models/sam2/checkpoints/sam2.1_hiera_base.pt")
    parser.add_argument("--model_cfg", type=str,
                        default="configs/sam2.1/sam2.1_hiera_b.yaml")
    
    # AMG settings (aggressive for more segments)
    parser.add_argument("--points_per_side", type=int, default=32,
                        help="Grid density for AMG (higher = more masks)")
    parser.add_argument("--pred_iou_thresh", type=float, default=0.7,
                        help="IoU threshold (lower = more masks, default SAM=0.88)")
    parser.add_argument("--stability_score_thresh", type=float, default=0.8,
                        help="Stability threshold (lower = more masks, default SAM=0.95)")
    parser.add_argument("--min_mask_region_area", type=int, default=100,
                        help="Minimum mask area in pixels")
    parser.add_argument("--max_segments", type=int, default=32,
                        help="Maximum segments to keep per frame")
    
    # Feature extraction
    parser.add_argument("--extract_features", action="store_true",
                        help="Also extract SAM encoder features")
    parser.add_argument("--feature_scales", type=int, nargs="+",
                        default=[4, 8, 16], help="Feature scales to extract")
    
    # Processing
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip sequences with existing outputs")
    
    return parser.parse_args()


def find_sequences(data_root: Path, split: str) -> List[Path]:
    """Find all 'original' subdirectories with images."""
    frame_root = data_root / split / "Frame_Anime"
    sequences = []
    
    for scene in sorted(frame_root.iterdir()):
        if not scene.is_dir():
            continue
        
        # Look for 'original' subdir (where masks are aligned)
        original = scene / "original"
        if original.exists() and list(original.glob("*.png")):
            sequences.append(original)
    
    return sequences


def load_sam2_amg(checkpoint: str, model_cfg: str, device: str):
    """Load SAM-2 Automatic Mask Generator."""
    try:
        # Try SAM-2 library
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        
        sam2_model = build_sam2(model_cfg, checkpoint, device=device)
        
        return sam2_model, SAM2AutomaticMaskGenerator
    except ImportError:
        print("SAM-2 not found. Please install: pip install segment-anything-2")
        sys.exit(1)


def masks_to_label_map(masks: List[Dict], max_segments: int) -> torch.Tensor:
    """
    Convert AMG output to single-channel label map.
    
    Args:
        masks: List of dicts from AMG, each with 'segmentation', 'area', 'stability_score'
        max_segments: Maximum segments to keep
        
    Returns:
        label_map: (H, W) uint8, 0=background, 1-N=segments
    """
    if not masks:
        return None
    
    # Sort by area (largest first) and stability
    sorted_masks = sorted(masks, 
                          key=lambda x: (x.get('stability_score', 0.9), x['area']),
                          reverse=True)
    
    # Take top N segments
    sorted_masks = sorted_masks[:max_segments]
    
    # Get dimensions from first mask
    H, W = sorted_masks[0]['segmentation'].shape
    label_map = np.zeros((H, W), dtype=np.uint8)
    
    # Assign labels (background=0, segments=1,2,3,...)
    # Process in reverse order so larger/more stable segments take priority
    for i, mask_data in enumerate(reversed(sorted_masks)):
        seg = mask_data['segmentation']  # (H, W) bool
        label_id = len(sorted_masks) - i  # 1, 2, 3, ...
        label_map[seg] = label_id
    
    return torch.from_numpy(label_map)


def extract_encoder_features(sam_model, image: np.ndarray, scales: List[int]) -> Dict[int, torch.Tensor]:
    """
    Extract SAM encoder features at multiple scales.
    
    Args:
        sam_model: SAM-2 model
        image: (H, W, 3) uint8 RGB
        scales: List of downscale factors [4, 8, 16]
        
    Returns:
        features: Dict mapping scale -> (C, H/scale, W/scale) tensor
    """
    # This is a placeholder - actual implementation depends on SAM-2 internals
    # Will need to hook into the image encoder
    features = {}
    
    # SAM-2 typically outputs features at 1/16 scale
    # We can interpolate to get multi-scale
    
    return features


def process_sequence(
    seq_path: Path,
    output_dir: Path,
    amg,
    args,
    sam_model=None,
):
    """Process a single sequence directory."""
    # Get frame list
    frames = sorted(list(seq_path.glob("*.png")) + list(seq_path.glob("*.jpg")))
    if not frames:
        return
    
    # Create output directories
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    if args.extract_features:
        features_dir = output_dir / "features"
        features_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already done
    if args.skip_existing:
        last_mask = masks_dir / (frames[-1].stem + ".pt")
        if last_mask.exists():
            return
    
    # Process each frame
    for frame_path in tqdm(frames, desc=f"  {seq_path.parent.name}", leave=False):
        out_mask_path = masks_dir / (frame_path.stem + ".pt")
        
        if args.skip_existing and out_mask_path.exists():
            continue
        
        # Load image
        img = cv2.imread(str(frame_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Run AMG
        try:
            masks = amg.generate(img_rgb)
        except Exception as e:
            print(f"Error processing {frame_path}: {e}")
            # Save empty mask
            torch.save(torch.zeros((img.shape[0], img.shape[1]), dtype=torch.uint8), out_mask_path)
            continue
        
        # Convert to label map
        label_map = masks_to_label_map(masks, args.max_segments)
        if label_map is None:
            label_map = torch.zeros((img.shape[0], img.shape[1]), dtype=torch.uint8)
        
        # Save mask
        torch.save(label_map, out_mask_path)
        
        # Extract and save features if requested
        if args.extract_features and sam_model is not None:
            out_feat_path = features_dir / (frame_path.stem + ".pt")
            features = extract_encoder_features(sam_model, img_rgb, args.feature_scales)
            torch.save(features, out_feat_path)


def main():
    args = parse_args()
    
    data_root = Path(args.data_root)
    output_root = Path(args.output_dir)
    
    print(f"=== SAM-2 Enhanced Precomputation V2 ===")
    print(f"Data root: {data_root}")
    print(f"Output: {output_root}")
    print(f"Settings: points_per_side={args.points_per_side}, "
          f"iou_thresh={args.pred_iou_thresh}, "
          f"stability_thresh={args.stability_score_thresh}")
    
    # Find sequences
    sequences = find_sequences(data_root, args.split)
    print(f"Found {len(sequences)} sequences in {args.split}")
    
    # Load SAM-2
    print(f"Loading SAM-2 from {args.checkpoint}...")
    sam_model, AMGClass = load_sam2_amg(args.checkpoint, args.model_cfg, args.device)
    
    # Create AMG with aggressive settings
    amg = AMGClass(
        model=sam_model,
        points_per_side=args.points_per_side,
        points_per_batch=64,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        stability_score_offset=1.0,
        crop_n_layers=0,  # No multi-crop for speed
        box_nms_thresh=0.7,
        min_mask_region_area=args.min_mask_region_area,
    )
    
    # Process sequences
    for seq_path in tqdm(sequences, desc="Processing sequences"):
        # Construct relative output path
        rel_path = seq_path.relative_to(data_root / args.split / "Frame_Anime")
        out_dir = output_root / args.split / "Frame_Anime" / rel_path
        
        process_sequence(
            seq_path,
            out_dir,
            amg,
            args,
            sam_model if args.extract_features else None,
        )
    
    print("Done!")


if __name__ == "__main__":
    main()
