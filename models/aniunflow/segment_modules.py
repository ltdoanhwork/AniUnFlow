# file: models/aniunflow/segment_modules.py
"""
Segment-Aware Model Extension Modules
======================================
Drop-in components for AniFlowFormer-T to enable segment-aware processing:

1. SegmentAwareCostModulation - Modulates cost volume based on segment affinity
2. SegmentAwareAttentionMask - Creates attention bias from segment structure  
3. SegmentAwareRefinementHead - Refines flow using segment boundaries

All modules are independently toggleable for ablation studies.
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SegmentAwareCostModulation(nn.Module):
    """
    Segment-Aware Cost Modulation
    ==============================
    Modulates the cost volume based on segment affinity between frames.
    
    Pixels within the same segment across frames have their matching costs
    boosted, while cross-segment matches are suppressed.
    
    This helps flow estimation by leveraging segment structure as a prior.
    """
    
    def __init__(
        self,
        cost_dim: int = 1,
        segment_dim: int = 64,
        modulation_strength: float = 0.5,
        learnable: bool = True,
    ):
        """
        Args:
            cost_dim: Number of cost volume channels
            segment_dim: Dimension for segment embedding projection
            modulation_strength: Strength of modulation (0 = no effect, 1 = full)
            learnable: Whether to use learnable projection layers
        """
        super().__init__()
        self.modulation_strength = modulation_strength
        self.learnable = learnable
        
        if learnable:
            # Project segment features to modulation weights
            self.segment_proj = nn.Sequential(
                nn.Conv2d(segment_dim, segment_dim // 2, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(segment_dim // 2, 1, 1),
                nn.Sigmoid(),
            )
    
    def forward(
        self,
        cost_volume: torch.Tensor,      # (B, C, H, W) or (B, H1*W1, H2, W2)
        segment_masks_1: torch.Tensor,  # (B, S, H, W) source frame segments
        segment_masks_2: torch.Tensor,  # (B, S, H, W) target frame segments
    ) -> torch.Tensor:
        """
        Modulate cost volume using segment affinity.
        
        Args:
            cost_volume: Cost volume to modulate
            segment_masks_1: Segment masks for frame 1
            segment_masks_2: Segment masks for frame 2
        
        Returns:
            Modulated cost volume (same shape as input)
        """
        B = cost_volume.shape[0]
        
        # Compute segment affinity matrix
        # For each pixel pair, compute how likely they belong to same segment
        affinity = self._compute_segment_affinity(segment_masks_1, segment_masks_2)
        
        # Resize affinity to match cost volume spatial dimensions
        if cost_volume.dim() == 4:
            # Standard cost volume (B, C, H, W)
            H, W = cost_volume.shape[-2:]
            affinity = F.interpolate(
                affinity.unsqueeze(1), size=(H, W),
                mode='bilinear', align_corners=False
            ).squeeze(1)  # (B, H, W)
            
            # Apply modulation
            modulation = 1.0 + self.modulation_strength * (affinity.unsqueeze(1) - 0.5)
            cost_volume = cost_volume * modulation
        
        return cost_volume
    
    def _compute_segment_affinity(
        self,
        masks_1: torch.Tensor,
        masks_2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-pixel segment affinity between two frames.
        
        Returns affinity map where high values indicate pixels likely
        belong to the same semantic segment.
        """
        B, S, H, W = masks_1.shape
        
        # Ensure same resolution
        if masks_2.shape[-2:] != (H, W):
            masks_2 = F.interpolate(
                masks_2, size=(H, W), mode='bilinear', align_corners=False
            )
        
        # Compute per-pixel segment assignment (soft)
        # For each pixel, we have a distribution over S segments
        # Affinity = dot product of segment distributions
        
        # Normalize to get probability distributions
        masks_1_norm = masks_1 / (masks_1.sum(dim=1, keepdim=True).clamp(min=1e-6))
        masks_2_norm = masks_2 / (masks_2.sum(dim=1, keepdim=True).clamp(min=1e-6))
        
        # Element-wise product and sum over segments
        affinity = (masks_1_norm * masks_2_norm).sum(dim=1)  # (B, H, W)
        
        return affinity


class SegmentAwareAttentionMask(nn.Module):
    """
    Segment-Aware Attention Mask Generator
    ========================================
    Creates attention bias from segment structure for transformer layers.
    
    Tokens from the same segment attend more strongly to each other,
    while cross-segment attention is relatively suppressed.
    """
    
    def __init__(
        self,
        temperature: float = 1.0,
        bias_scale: float = 2.0,
    ):
        """
        Args:
            temperature: Temperature for softmax (higher = softer attention)
            bias_scale: Scale factor for attention bias
        """
        super().__init__()
        self.temperature = temperature
        self.bias_scale = bias_scale
    
    def forward(
        self,
        segment_masks: torch.Tensor,  # (B, S, H, W)
        query_shape: Tuple[int, int],  # (h, w) for query tokens
        key_shape: Optional[Tuple[int, int]] = None,  # (h, w) for key tokens
    ) -> torch.Tensor:
        """
        Generate attention bias from segment masks.
        
        Args:
            segment_masks: Segment masks (B, S, H, W)
            query_shape: Spatial shape of query tokens
            key_shape: Spatial shape of key tokens (default: same as query)
        
        Returns:
            attention_bias: (B, N_q, N_k) bias to add to attention logits
        """
        B, S, H, W = segment_masks.shape
        h_q, w_q = query_shape
        h_k, w_k = key_shape if key_shape else query_shape
        
        # Resize masks to query/key resolutions
        masks_q = F.interpolate(
            segment_masks, size=(h_q, w_q), mode='bilinear', align_corners=False
        )  # (B, S, h_q, w_q)
        masks_k = F.interpolate(
            segment_masks, size=(h_k, w_k), mode='bilinear', align_corners=False
        )  # (B, S, h_k, w_k)
        
        # Flatten to token sequences
        masks_q = rearrange(masks_q, 'b s h w -> b (h w) s')  # (B, N_q, S)
        masks_k = rearrange(masks_k, 'b s h w -> b s (h w)')  # (B, S, N_k)
        
        # Normalize to get segment probability
        masks_q = masks_q / (masks_q.sum(dim=-1, keepdim=True).clamp(min=1e-6))
        masks_k = masks_k / (masks_k.sum(dim=-1, keepdim=True).clamp(min=1e-6))
        
        # Compute segment affinity as attention bias
        # Tokens in same segment should have higher affinity
        # affinity[i,j] = sum_s p(s|q_i) * p(s|k_j)
        affinity = torch.bmm(masks_q, masks_k)  # (B, N_q, N_k)
        
        # Scale to attention bias range
        attention_bias = self.bias_scale * (affinity - 0.5) / self.temperature
        
        return attention_bias


class SegmentAwareRefinementHead(nn.Module):
    """
    Segment-Aware Flow Refinement Head
    ====================================
    Post-processing refinement that uses segment boundaries to improve
    flow predictions, especially at object boundaries.
    
    The refinement encourages:
    - Smooth flow within segments
    - Accurate flow discontinuities at segment boundaries
    """
    
    def __init__(
        self,
        flow_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 2,
        use_boundary_features: bool = True,
    ):
        """
        Args:
            flow_dim: Flow channels (typically 2)
            hidden_dim: Hidden dimension for refinement network
            num_layers: Number of refinement layers
            use_boundary_features: Whether to use explicit boundary features
        """
        super().__init__()
        self.use_boundary_features = use_boundary_features
        
        # Input: flow + optional boundary
        in_dim = flow_dim + (1 if use_boundary_features else 0)
        
        layers = []
        for i in range(num_layers):
            dim_in = in_dim if i == 0 else hidden_dim
            dim_out = hidden_dim if i < num_layers - 1 else flow_dim
            layers.extend([
                nn.Conv2d(dim_in, dim_out, 3, padding=1),
                nn.ReLU(inplace=True) if i < num_layers - 1 else nn.Identity(),
            ])
        
        self.refine_net = nn.Sequential(*layers)
        
        # Residual weight (learnable)
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(
        self,
        flow: torch.Tensor,           # (B, 2, H, W)
        boundary_map: torch.Tensor,   # (B, 1, H, W)
    ) -> torch.Tensor:
        """
        Refine flow using segment boundary information.
        
        Args:
            flow: Initial flow prediction
            boundary_map: Segment boundary map (1 at boundaries)
        
        Returns:
            Refined flow (same shape as input)
        """
        # Resize boundary if needed
        if boundary_map.shape[-2:] != flow.shape[-2:]:
            boundary_map = F.interpolate(
                boundary_map, size=flow.shape[-2:],
                mode='bilinear', align_corners=False
            )
        
        if self.use_boundary_features:
            x = torch.cat([flow, boundary_map], dim=1)
        else:
            x = flow
        
        # Compute residual refinement
        residual = self.refine_net(x)
        
        # Apply residual with learnable weight
        refined_flow = flow + self.residual_weight * residual
        
        return refined_flow


class SegmentGuidedCorrelation(nn.Module):
    """
    Segment-Guided Correlation Layer
    ==================================
    Computes correlation/cost volume with segment-aware weighting.
    
    This replaces standard correlation with one that considers
    segment structure when computing matching costs.
    """
    
    def __init__(
        self,
        feature_dim: int,
        segment_weight: float = 0.3,
        radius: int = 4,
    ):
        """
        Args:
            feature_dim: Feature dimension
            segment_weight: Weight for segment-based modulation
            radius: Correlation search radius
        """
        super().__init__()
        self.segment_weight = segment_weight
        self.radius = radius
        
        # Segment feature projection
        self.seg_proj = nn.Conv2d(feature_dim, feature_dim // 4, 1)
    
    def forward(
        self,
        fmap1: torch.Tensor,          # (B, C, H, W)
        fmap2: torch.Tensor,          # (B, C, H, W)
        segment_masks_1: torch.Tensor, # (B, S, H, W)
        segment_masks_2: torch.Tensor, # (B, S, H, W)
    ) -> torch.Tensor:
        """
        Compute segment-guided correlation volume.
        
        Args:
            fmap1, fmap2: Feature maps from two frames
            segment_masks_1, segment_masks_2: Corresponding segment masks
        
        Returns:
            corr: Correlation volume with segment modulation
        """
        B, C, H, W = fmap1.shape
        
        # Standard correlation (all pairs within radius)
        # For simplicity, using global correlation here
        fmap1_flat = fmap1.view(B, C, -1)  # (B, C, HW)
        fmap2_flat = fmap2.view(B, C, -1)
        
        corr = torch.bmm(fmap1_flat.transpose(1, 2), fmap2_flat)  # (B, HW, HW)
        corr = corr / (C ** 0.5)  # Scale by sqrt(d)
        
        # Segment affinity modulation
        # Resize masks to feature resolution
        masks_1 = F.interpolate(segment_masks_1, size=(H, W), mode='bilinear', align_corners=False)
        masks_2 = F.interpolate(segment_masks_2, size=(H, W), mode='bilinear', align_corners=False)
        
        # Compute segment affinity
        masks_1_flat = masks_1.view(B, -1, H * W)  # (B, S, HW)
        masks_2_flat = masks_2.view(B, -1, H * W)
        
        # Normalize
        masks_1_norm = masks_1_flat / (masks_1_flat.sum(dim=1, keepdim=True).clamp(min=1e-6))
        masks_2_norm = masks_2_flat / (masks_2_flat.sum(dim=1, keepdim=True).clamp(min=1e-6))
        
        # Segment affinity
        seg_affinity = torch.bmm(masks_1_norm.transpose(1, 2), masks_2_norm)  # (B, HW, HW)
        
        # Modulate correlation
        corr = corr + self.segment_weight * seg_affinity
        
        # Reshape to spatial
        corr = corr.view(B, H, W, H, W)
        
        return corr


class SegmentAwareModuleBundle(nn.Module):
    """
    Bundle of all segment-aware modules for easy integration.
    
    This provides a unified interface to all segment-aware extensions,
    with each component independently toggleable.
    """
    
    def __init__(
        self,
        feature_dim: int = 128,
        use_cost_modulation: bool = True,
        use_attention_mask: bool = True,
        use_refinement: bool = False,
    ):
        super().__init__()
        
        self.use_cost_modulation = use_cost_modulation
        self.use_attention_mask = use_attention_mask
        self.use_refinement = use_refinement
        
        if use_cost_modulation:
            self.cost_modulation = SegmentAwareCostModulation(
                segment_dim=feature_dim
            )
        
        if use_attention_mask:
            self.attention_mask = SegmentAwareAttentionMask()
        
        if use_refinement:
            self.refinement = SegmentAwareRefinementHead(
                hidden_dim=feature_dim
            )
    
    def modulate_cost(
        self,
        cost_volume: torch.Tensor,
        masks_1: torch.Tensor,
        masks_2: torch.Tensor,
    ) -> torch.Tensor:
        """Apply cost modulation if enabled."""
        if self.use_cost_modulation:
            return self.cost_modulation(cost_volume, masks_1, masks_2)
        return cost_volume
    
    def get_attention_bias(
        self,
        segment_masks: torch.Tensor,
        query_shape: Tuple[int, int],
        key_shape: Optional[Tuple[int, int]] = None,
    ) -> Optional[torch.Tensor]:
        """Get attention bias if enabled."""
        if self.use_attention_mask:
            return self.attention_mask(segment_masks, query_shape, key_shape)
        return None
    
    def refine_flow(
        self,
        flow: torch.Tensor,
        boundary_map: torch.Tensor,
    ) -> torch.Tensor:
        """Apply flow refinement if enabled."""
        if self.use_refinement:
            return self.refinement(flow, boundary_map)
        return flow


# ============= Factory function =============
def build_segment_modules(cfg: Dict) -> SegmentAwareModuleBundle:
    """Build segment-aware module bundle from config."""
    model_cfg = cfg.get('model', {}).get('args', {})
    return SegmentAwareModuleBundle(
        feature_dim=model_cfg.get('enc_channels', 64) * 2,
        use_cost_modulation=model_cfg.get('use_segment_cost_modulation', False),
        use_attention_mask=model_cfg.get('use_segment_attention_mask', False),
        use_refinement=model_cfg.get('use_segment_refinement', False),
    )
