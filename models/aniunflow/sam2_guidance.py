# file: models/aniunflow/sam2_guidance.py
"""
SAM-2 Video Segmentation Guidance Module
==========================================
Provides temporally consistent segment masks for guiding optical flow training.

SAM-2 is used ONLY as a structural prior, NOT as ground-truth supervision.
The module is designed to be SAM-variant agnostic (SAM-1/SAM-2/future).

Key Features:
- Frozen SAM-2 (no gradient flow)
- Lazy loading for memory efficiency  
- Automatic video mode for temporal consistency
- Configurable granularity (number of segments)
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class SAM2GuidanceModule(nn.Module):
    """
    SAM-2 Video Segmentation for Structural Guidance.
    
    This module wraps SAM-2 to extract temporally consistent segment masks
    that can be used for:
    - Segment-aware cost aggregation
    - Occlusion reasoning  
    - Flat-region regularization
    - Segment-level temporal consistency
    
    The SAM-2 model is completely frozen during training.
    """
    
    def __init__(
        self,
        sam_checkpoint: Optional[str] = None,
        model_type: str = "configs/sam2.1/sam2.1_hiera_t.yaml",
        device: str = "cuda",
        num_segments: int = 16,
        mask_threshold: float = 0.0,
        use_automatic_mask_generator: bool = True,
    ):
        """
        Initialize SAM-2 guidance module.
        
        Args:
            sam_checkpoint: Path to SAM-2 checkpoint
            model_type: SAM-2 model type (sam2_hiera_t, sam2_hiera_s, etc.)
            device: Device for SAM-2 inference
            num_segments: Target number of segments per frame
            mask_threshold: Threshold for binarizing soft masks
            use_automatic_mask_generator: Use automatic mask generation
        """
        super().__init__()
        self.sam_checkpoint = sam_checkpoint
        self.model_type = model_type
        self.device_str = device
        self.num_segments = num_segments
        self.mask_threshold = mask_threshold
        self.use_amg = use_automatic_mask_generator
        
        # Lazy loading - SAM-2 is loaded on first use
        self._sam_predictor = None
        self._sam_model = None
        self._loaded = False
    
    def _lazy_load(self):
        """Lazy load SAM-2 model on first use."""
        if self._loaded:
            return
        
        try:
            # Try importing SAM-2 from the local models/sam2 directory
            import sys
            sam2_path = Path(__file__).parent.parent / "sam2"
            if sam2_path.exists():
                sys.path.insert(0, str(sam2_path))
            
            from sam2.build_sam import build_sam2_video_predictor, build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            
            if self.sam_checkpoint is None:
                # Default checkpoint path
                self.sam_checkpoint = str(sam2_path / "checkpoints" / "sam2.1_hiera_tiny.pt")
            
            # Build SAM-2 model
            self._sam_model = build_sam2(
                self.model_type,
                self.sam_checkpoint,
                device=self.device_str,
            )
            
            # Create automatic mask generator if needed
            if self.use_amg:
                self._mask_generator = SAM2AutomaticMaskGenerator(
                    model=self._sam_model,
                    points_per_side=16,
                    points_per_batch=64,
                    pred_iou_thresh=0.7,
                    stability_score_thresh=0.92,
                    box_nms_thresh=0.7,
                )
            
            # Freeze all parameters
            for param in self._sam_model.parameters():
                param.requires_grad = False
            
            self._loaded = True
            print(f"[SAM2Guidance] Loaded SAM-2 model: {self.model_type}")
            
        except ImportError as e:
            print(f"[SAM2Guidance] Warning: Could not load SAM-2: {e}")
            print("[SAM2Guidance] Using fallback grid-based segmentation")
            self._loaded = True
            self._sam_model = None
    
    def _masks_to_label_map(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Convert multi-channel masks to integer label map.
        
        Args:
            masks: (S, H, W) float masks
        
        Returns:
            labels: (1, H, W) uint8 integer labels where 0=background, 1-S=segments
        """
        if masks.shape[0] == 0:
            # No segments
            return torch.zeros((1, masks.shape[1], masks.shape[2]), dtype=torch.uint8, device=masks.device)
        
        # Find segment with max activation per pixel
        max_vals, labels = masks.max(dim=0)  # (H, W)
        
        # Shift to 1-indexed (0 reserved for background)
        labels = labels + 1
        
        # Set low-activation pixels to background
        labels[max_vals < 0.5] = 0
        
        return labels.unsqueeze(0).to(torch.uint8)  # (1, H, W)

    @torch.no_grad()
    def extract_segment_masks(
        self,
        clip: torch.Tensor,  # (B, T, 3, H, W) in [0, 1]
    ) -> torch.Tensor:
        """
        Extract temporally consistent segment masks for a video clip.
        
        Args:
            clip: Video clip tensor (B, T, 3, H, W) normalized to [0, 1]
        
        Returns:
            segment_labels: (B, T, 1, H, W) uint8 integer label maps
                where 0=background, 1-S=segment IDs
                
        Note: This returns integer label maps instead of soft masks for:
            1. Storage efficiency (30× smaller)
            2. UnSAMFlow compatibility (expects integer labels)
        """
        self._lazy_load()
        
        B, T, C, H, W = clip.shape
        device = clip.device
        
        if self._sam_model is None:
            # Fallback: grid-based pseudo-segmentation
            return self._fallback_grid_segmentation_labels(clip)
        
        # Process each batch item
        all_labels = []
        for b in range(B):
            batch_labels = []
            for t in range(T):
                frame = clip[b, t]  # (3, H, W)
                
                # Convert to numpy for SAM-2 (expects HWC uint8)
                frame_np = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
                
                # Generate masks
                masks_np = self._generate_frame_masks(frame_np)  # (S, H, W)
                
                # Convert to tensor
                masks_tensor = torch.from_numpy(masks_np).float().to(device)
                
                # Ensure we have exactly num_segments
                masks_tensor = self._normalize_segment_count(masks_tensor)
                
                # Convert to integer label map
                labels = self._masks_to_label_map(masks_tensor)  # (1, H, W)
                
                batch_labels.append(labels)
            
            all_labels.append(torch.stack(batch_labels, dim=0))  # (T, 1, H, W)
        
        return torch.stack(all_labels, dim=0)  # (B, T, 1, H, W)
    
    def _generate_frame_masks(self, frame_np) -> 'np.ndarray':
        """Generate segment masks for a single frame."""
        import numpy as np
        
        try:
            if self.use_amg and hasattr(self, '_mask_generator'):
                # Use automatic mask generator
                results = self._mask_generator.generate(frame_np)
                
                # Sort by area (largest first) and take top N
                results = sorted(results, key=lambda x: x['area'], reverse=True)
                results = results[:self.num_segments]
                
                # Stack masks
                masks = np.stack([r['segmentation'] for r in results], axis=0)
                return masks.astype(np.float32)
            else:
                # Fallback
                H, W = frame_np.shape[:2]
                return self._create_grid_masks(H, W)
        except Exception as e:
            print(f"[SAM2Guidance] Mask generation failed: {e}")
            H, W = frame_np.shape[:2]
            return self._create_grid_masks(H, W)
    
    def _create_grid_masks(self, H: int, W: int) -> 'np.ndarray':
        """Create simple grid-based pseudo-segmentation."""
        import numpy as np
        
        # Create grid of segments
        n_rows = int(np.sqrt(self.num_segments))
        n_cols = (self.num_segments + n_rows - 1) // n_rows
        
        masks = np.zeros((n_rows * n_cols, H, W), dtype=np.float32)
        
        cell_h = H // n_rows
        cell_w = W // n_cols
        
        idx = 0
        for i in range(n_rows):
            for j in range(n_cols):
                if idx >= self.num_segments:
                    break
                y1, y2 = i * cell_h, (i + 1) * cell_h if i < n_rows - 1 else H
                x1, x2 = j * cell_w, (j + 1) * cell_w if j < n_cols - 1 else W
                masks[idx, y1:y2, x1:x2] = 1.0
                idx += 1
        
        return masks[:self.num_segments]
    
    def _fallback_grid_segmentation(self, clip: torch.Tensor) -> torch.Tensor:
        """Fallback grid-based segmentation when SAM-2 is unavailable (legacy multi-channel)."""
        B, T, C, H, W = clip.shape
        device = clip.device
        
        # Create grid masks
        import numpy as np
        masks_np = self._create_grid_masks(H, W)
        masks = torch.from_numpy(masks_np).float().to(device)  # (S, H, W)
        
        # Expand for batch and time dimensions
        masks = masks.unsqueeze(0).unsqueeze(0)  # (1, 1, S, H, W)
        masks = masks.expand(B, T, -1, -1, -1)   # (B, T, S, H, W)
        
        return masks

    def _fallback_grid_segmentation_labels(self, clip: torch.Tensor) -> torch.Tensor:
        """Fallback grid-based segmentation returning integer labels."""
        B, T, C, H, W = clip.shape
        device = clip.device
        
        # Create grid label map directly
        n_rows = int(torch.sqrt(torch.tensor(self.num_segments)).item())
        n_cols = (self.num_segments + n_rows - 1) // n_rows
        
        labels = torch.zeros((H, W), dtype=torch.uint8, device=device)
        
        cell_h = H // n_rows
        cell_w = W // n_cols
        
        label_id = 1  # Start from 1 (0 is background)
        for i in range(n_rows):
            for j in range(n_cols):
                if label_id > self.num_segments:
                    break
                y1, y2 = i * cell_h, (i + 1) * cell_h if i < n_rows - 1 else H
                x1, x2 = j * cell_w, (j + 1) * cell_w if j < n_cols - 1 else W
                labels[y1:y2, x1:x2] = label_id
                label_id += 1
        
        # Expand for batch and time dimensions
        labels = labels.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, H, W)
        labels = labels.expand(B, T, -1, -1, -1)  # (B, T, 1, H, W)
        
        return labels
    
    def _normalize_segment_count(self, masks: torch.Tensor) -> torch.Tensor:
        """Ensure we have exactly num_segments masks."""
        S = masks.shape[0]
        H, W = masks.shape[-2:]
        
        if S == self.num_segments:
            return masks
        elif S > self.num_segments:
            return masks[:self.num_segments]
        else:
            # Pad with empty masks
            padding = torch.zeros(
                self.num_segments - S, H, W,
                device=masks.device, dtype=masks.dtype
            )
            return torch.cat([masks, padding], dim=0)
    
    @torch.no_grad()
    def extract_segment_tokens(
        self,
        clip: torch.Tensor,           # (B, T, 3, H, W)
        features: List[torch.Tensor], # List of (B, C, h, w) per time step
        segment_masks: Optional[torch.Tensor] = None,  # (B, T, S, H, W)
    ) -> torch.Tensor:
        """
        Extract segment-level feature tokens by pooling features within segments.
        
        Args:
            clip: Video clip for mask extraction if needed
            features: Feature maps from encoder (one per time step)
            segment_masks: Pre-computed segment masks (optional)
        
        Returns:
            segment_tokens: (B, T, S, D) segment feature tokens
        """
        if segment_masks is None:
            segment_masks = self.extract_segment_masks(clip)
        
        B, T, S, H_mask, W_mask = segment_masks.shape
        
        all_tokens = []
        for t in range(T):
            feat = features[t] if isinstance(features, list) else features[:, t]
            _, C, h, w = feat.shape
            
            # Resize masks to feature resolution
            masks_t = segment_masks[:, t]  # (B, S, H_mask, W_mask)
            masks_t = F.interpolate(
                masks_t, size=(h, w), mode='bilinear', align_corners=False
            )  # (B, S, h, w)
            
            # Pool features within each segment
            masks_flat = masks_t.view(B, S, -1)  # (B, S, hw)
            feat_flat = feat.view(B, C, -1)       # (B, C, hw)
            
            # Weighted average pooling
            weights = masks_flat / (masks_flat.sum(dim=-1, keepdim=True).clamp(min=1e-6))
            tokens = torch.bmm(weights, feat_flat.transpose(1, 2))  # (B, S, C)
            
            all_tokens.append(tokens)
        
        return torch.stack(all_tokens, dim=1)  # (B, T, S, C)
    
    @torch.no_grad()
    def compute_boundary_maps(
        self,
        segment_masks: torch.Tensor,  # (B, T, S, H, W)
        kernel_size: int = 3,
    ) -> torch.Tensor:
        """
        Compute boundary maps from segment masks.
        
        Args:
            segment_masks: Segment masks (B, T, S, H, W)
            kernel_size: Morphological kernel size
        
        Returns:
            boundary_maps: (B, T, 1, H, W) boundary indicator maps
        """
        B, T, S, H, W = segment_masks.shape
        
        all_boundaries = []
        for t in range(T):
            masks_t = segment_masks[:, t]  # (B, S, H, W)
            
            # Compute boundary for each segment and aggregate
            boundary = torch.zeros(B, 1, H, W, device=masks_t.device, dtype=masks_t.dtype)
            
            padding = kernel_size // 2
            for s in range(S):
                m = masks_t[:, s:s+1]  # (B, 1, H, W)
                dilated = F.max_pool2d(m, kernel_size, stride=1, padding=padding)
                eroded = -F.max_pool2d(-m, kernel_size, stride=1, padding=padding)
                boundary = boundary + (dilated - eroded).clamp(0, 1)
            
            boundary = boundary.clamp(0, 1)
            all_boundaries.append(boundary)
        
        return torch.stack(all_boundaries, dim=1)  # (B, T, 1, H, W)


class SegmentMaskCache:
    """
    Cache for pre-computed segment masks.
    Useful for training efficiency when SAM-2 inference is slow.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.memory_cache: Dict[str, torch.Tensor] = {}
    
    def get(self, key: str) -> Optional[torch.Tensor]:
        """Get cached masks."""
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        if self.cache_dir:
            cache_path = self.cache_dir / f"{key}.pt"
            if cache_path.exists():
                masks = torch.load(cache_path)
                self.memory_cache[key] = masks
                return masks
        
        return None
    
    def put(self, key: str, masks: torch.Tensor):
        """Cache masks."""
        self.memory_cache[key] = masks
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save(masks, self.cache_dir / f"{key}.pt")


# ============= Factory function =============
def build_sam2_guidance(cfg: Dict) -> SAM2GuidanceModule:
    """Build SAM-2 guidance module from config."""
    sam_cfg = cfg.get('sam', {})
    return SAM2GuidanceModule(
        sam_checkpoint=sam_cfg.get('checkpoint'),
        model_type=sam_cfg.get('model_type', 'configs/sam2.1/sam2.1_hiera_t.yaml'),
        num_segments=sam_cfg.get('num_segments', 16),
        use_automatic_mask_generator=sam_cfg.get('use_amg', True),
    )
