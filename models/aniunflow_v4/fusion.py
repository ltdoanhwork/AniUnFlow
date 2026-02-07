"""
Feature Fusion Modules
======================
Fuse SAM features with optical flow encoder features.
Based on SAMFlow Context Fusion Module (CFM).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class ContextFusionModule(nn.Module):
    """
    Context Fusion Module (CFM) from SAMFlow.
    
    Fuses SAM encoder features with optical flow features
    using adaptive gating and multi-scale alignment.
    """
    
    def __init__(
        self,
        flow_dim: int = 64,
        sam_dim: int = 256,
        out_dim: int = 64,
        num_scales: int = 2,
    ):
        super().__init__()
        
        self.flow_dim = flow_dim
        self.sam_dim = sam_dim
        self.out_dim = out_dim
        
        # Project SAM features to match flow dimension
        self.sam_proj = nn.Conv2d(sam_dim, out_dim, 1, bias=False)
        
        # Gating mechanism: learn how much SAM info to use
        self.gate = nn.Sequential(
            nn.Conv2d(flow_dim + out_dim, out_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
            nn.Sigmoid(),
        )
        
        # Final fusion
        self.fusion = nn.Conv2d(flow_dim + out_dim, out_dim, 1)
        
    def forward(
        self,
        flow_feat: torch.Tensor,     # (B, C_flow, H, W)
        sam_feat: torch.Tensor,      # (B, C_sam, H', W')
    ) -> torch.Tensor:
        """
        Fuse flow and SAM features.
        
        Returns:
            fused: (B, out_dim, H, W)
        """
        B, _, H, W = flow_feat.shape
        
        # Align SAM features to flow resolution
        if sam_feat.shape[2:] != (H, W):
            sam_feat = F.interpolate(sam_feat, size=(H, W), mode='bilinear', align_corners=False)
        
        # Project SAM features
        sam_proj = self.sam_proj(sam_feat)
        
        # Compute gate (how much SAM info to use)
        concat = torch.cat([flow_feat, sam_proj], dim=1)
        gate = self.gate(concat)
        
        # Gated SAM contribution
        gated_sam = gate * sam_proj
        
        # Final fusion
        fused = self.fusion(torch.cat([flow_feat, gated_sam], dim=1))
        
        return fused


class MultiScaleFusion(nn.Module):
    """
    Multi-scale feature fusion for pyramid encoder.
    
    Fuses SAM features at each pyramid level.
    """
    
    def __init__(
        self,
        flow_dims: List[int] = [32, 64, 128],
        sam_dim: int = 256,
        scales: List[int] = [4, 8, 16],
    ):
        super().__init__()
        
        self.scales = scales
        self.fusers = nn.ModuleDict()
        
        for i, (scale, flow_dim) in enumerate(zip(scales, flow_dims)):
            self.fusers[str(scale)] = ContextFusionModule(
                flow_dim=flow_dim,
                sam_dim=sam_dim,
                out_dim=flow_dim,
            )
    
    def forward(
        self,
        flow_feats: Dict[int, torch.Tensor],   # scale -> (B, C, H, W)
        sam_feats: Dict[int, torch.Tensor],    # scale -> (B, C_sam, H, W)
    ) -> Dict[int, torch.Tensor]:
        """
        Fuse at each scale.
        
        Returns:
            fused: Dict[scale, (B, C, H, W)]
        """
        fused = {}
        
        for scale in self.scales:
            scale_key = str(scale)
            
            if scale not in flow_feats:
                continue
            
            flow_feat = flow_feats[scale]
            
            # Get nearest SAM feature scale
            sam_feat = None
            for sam_scale in sorted(sam_feats.keys(), reverse=True):
                if sam_scale <= scale:
                    sam_feat = sam_feats[sam_scale]
                    break
            
            if sam_feat is None and sam_feats:
                sam_feat = list(sam_feats.values())[0]
            
            if sam_feat is not None and scale_key in self.fusers:
                fused[scale] = self.fusers[scale_key](flow_feat, sam_feat)
            else:
                fused[scale] = flow_feat
        
        return fused


class BoundaryAwareConcat(nn.Module):
    """
    Simple boundary-aware feature concatenation.
    
    Concatenates boundary maps and mask features with flow features.
    """
    
    def __init__(
        self,
        flow_dim: int = 64,
        out_dim: int = 64,
        num_segments: int = 16,
    ):
        super().__init__()
        
        # Boundary encoding (+1 for boundary channel)
        # Segment encoding (+num_segments for one-hot, or just use label)
        extra_channels = 1 + 1  # boundary + segment label (normalized)
        
        self.proj = nn.Conv2d(flow_dim + extra_channels, out_dim, 1)
    
    def forward(
        self,
        flow_feat: torch.Tensor,    # (B, C, H, W)
        boundary: torch.Tensor,     # (B, 1, H, W) 
        segment_labels: torch.Tensor,  # (B, 1, H, W) normalized labels
    ) -> torch.Tensor:
        """
        Concatenate boundary and segment info with flow features.
        """
        # Ensure same resolution
        if boundary.shape[2:] != flow_feat.shape[2:]:
            boundary = F.interpolate(boundary, size=flow_feat.shape[2:], mode='nearest')
        if segment_labels.shape[2:] != flow_feat.shape[2:]:
            segment_labels = F.interpolate(segment_labels, size=flow_feat.shape[2:], mode='nearest')
        
        concat = torch.cat([flow_feat, boundary, segment_labels], dim=1)
        return self.proj(concat)
