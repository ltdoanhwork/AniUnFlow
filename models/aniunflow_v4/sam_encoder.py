"""
SAM Encoder Feature Extractor
=============================
Extracts frozen ViT features from SAM-2 encoder.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys
import os

# Add SAM2 to python path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # AniUnFlow root
SAM2_ROOT = ROOT / "models" / "sam2"
if str(SAM2_ROOT) not in sys.path:
    sys.path.append(str(SAM2_ROOT))


class SAMEncoderWrapper(nn.Module):
    """
    Wraps SAM-2 image encoder for feature extraction.
    
    Features can be extracted at multiple scales and fused with
    the optical flow encoder pyramid.
    """
    
    def __init__(
        self,
        checkpoint: str = "models/sam2/checkpoints/sam2.1_hiera_base_plus.pt",
        config: str = "configs/sam2.1/sam2.1_hiera_b+.yaml",
        freeze: bool = True,
        feature_scales: List[int] = [8, 16],
        device: str = "cuda",
    ):
        super().__init__()
        
        self.checkpoint = checkpoint
        self.config = config
        self.freeze = freeze
        self.feature_scales = feature_scales
        self.device = device
        
        # Lazy load to avoid loading SAM at import time
        self._encoder = None
        self._loaded = False
        
    def _lazy_load(self):
        """Load SAM encoder on first forward pass."""
        if self._loaded:
            return
            
        from sam2.build_sam import build_sam2
        
        # Build SAM model
        sam_model = build_sam2(self.config, self.checkpoint, device=self.device)
        
        # Extract image encoder
        self._encoder = sam_model.image_encoder
        
        if self.freeze:
            for param in self._encoder.parameters():
                param.requires_grad = False
            self._encoder.eval()
        
        self._loaded = True
        print(f"[SAMEncoder] Loaded from {self.checkpoint}")
    
    def forward(
        self,
        images: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        """
        Extract SAM encoder features.
        
        Args:
            images: (B, 3, H, W) normalized RGB images
            
        Returns:
            features: Dict mapping scale -> (B, C, H/scale, W/scale)
        """
        self._lazy_load()
        
        if self._encoder is None:
            return {}
        
        B, C, H, W = images.shape
        
        with torch.set_grad_enabled(not self.freeze):
            # SAM expects images in [0, 255] range typically
            # Adjust if your images are [0, 1]
            if images.max() <= 1.0:
                images = images * 255.0
            
            # Get encoder output
            # SAM-2 Hiera encoder returns multi-scale features
            encoder_out = self._encoder(images)
            
        # Extract features at requested scales
        features = {}
        
        if isinstance(encoder_out, dict):
            # SAM-2 returns dict with 'vision_features', etc.
            for scale in self.feature_scales:
                key = f"scale_{scale}"
                if key in encoder_out:
                    features[scale] = encoder_out[key]
        elif isinstance(encoder_out, (list, tuple)):
            # Multi-scale list output
            for i, scale in enumerate(self.feature_scales):
                if i < len(encoder_out):
                    features[scale] = encoder_out[i]
        else:
            # Single tensor output - interpolate to get multi-scale
            feat = encoder_out
            for scale in self.feature_scales:
                target_h, target_w = H // scale, W // scale
                if feat.shape[2:] != (target_h, target_w):
                    features[scale] = nn.functional.interpolate(
                        feat, size=(target_h, target_w), mode='bilinear', align_corners=False
                    )
                else:
                    features[scale] = feat
        
        return features
    
    def get_feature_dim(self, scale: int) -> int:
        """Get feature dimension at given scale."""
        # SAM-2 Hiera base: 768 dim at final scale
        # This may vary by model size
        return 768 if scale >= 16 else 384


class SAMFeatureCache:
    """
    Loads precomputed SAM features from disk.
    Used during training for efficiency.
    """
    
    def __init__(self, cache_dir: str, device: str = "cuda"):
        self.cache_dir = Path(cache_dir)
        self.device = device
        self._cache = {}
    
    def get_features(
        self,
        frame_paths: List[str],
    ) -> Optional[Dict[int, torch.Tensor]]:
        """
        Load cached features for given frames.
        
        Args:
            frame_paths: List of frame file paths
            
        Returns:
            features: Dict[scale, (T, C, H, W)] or None if not cached
        """
        features_list = []
        
        for path in frame_paths:
            path = Path(path)
            cache_path = self.cache_dir / path.parent.name / "features" / (path.stem + ".pt")
            
            if not cache_path.exists():
                return None
            
            try:
                feat = torch.load(cache_path, map_location=self.device)
                features_list.append(feat)
            except Exception:
                return None
        
        if not features_list:
            return None
        
        # Stack along time dimension
        scales = features_list[0].keys()
        stacked = {}
        for scale in scales:
            stacked[scale] = torch.stack([f[scale] for f in features_list], dim=0)
        
        return stacked
