"""
SAM Guidance Adapter V3
========================
Enhanced SAM integration for optical flow estimation.

Based on:
- SAMFlow (AAAI 2024): Feature concatenation + residual mixing
- UnSAMFlow (CVPR 2024): Object-level guidance + boundary awareness
- FlowI-SAM / FlowP-SAM: Flow-as-input and flow-as-prompt paradigms

4 Integration Modes:
1. Feature Concatenation: Concat SAM boundary features with encoder features
2. Attention Bias Generation: Generate attention bias for LCM from segment masks
3. Cost Modulation: Modulate matching cost at segment boundaries
4. Object Pooling: Pool features per segment for segment tokens

This is V3 - does not modify existing sam_adapter.py
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def compute_boundary_map(masks: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    Compute segment boundaries from segment masks using morphological gradient.
    
    Args:
        masks: (B, S, H, W) segment masks
        kernel_size: Dilation kernel size
    
    Returns:
        boundary: (B, 1, H, W) boundary map in [0, 1]
    """
    B, S, H, W = masks.shape
    padding = kernel_size // 2
    
    # Ensure float dtype for pooling operations
    masks = masks.float()
    
    # Combine boundaries from all segments
    boundary = torch.zeros(B, 1, H, W, device=masks.device, dtype=masks.dtype)
    
    for s in range(S):
        m = masks[:, s:s+1]  # (B, 1, H, W)
        # Morphological gradient = dilation - erosion
        dilated = F.max_pool2d(m, kernel_size, stride=1, padding=padding)
        eroded = -F.max_pool2d(-m, kernel_size, stride=1, padding=padding)
        boundary = boundary + (dilated - eroded).clamp(0, 1)
    
    return boundary.clamp(0, 1)


class ResidualMixBlock(nn.Module):
    """
    Residual block for mixing SAM and encoder features (SAMFlow style).
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.GELU()
        
        # Residual projection if dimensions differ
        self.proj = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class SegmentTokenEncoder(nn.Module):
    """
    Encode segment masks into compact segment tokens via masked pooling.
    """
    def __init__(self, feat_dim: int, token_dim: int, num_segments: int = 16):
        super().__init__()
        self.num_segments = num_segments
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
    
    def forward(
        self,
        masks: torch.Tensor,  # (B, S, H, W)
        features: torch.Tensor,  # (B, C, H, W)
    ) -> torch.Tensor:
        """
        Pool features within each segment and project to tokens.
        
        Returns:
            seg_tokens: (B, S, token_dim)
        """
        B, S, H, W = masks.shape
        C = features.shape[1]
        
        # Resize masks to feature resolution if needed
        if masks.shape[-2:] != features.shape[-2:]:
            masks = F.interpolate(masks.float(), size=features.shape[-2:], mode='bilinear', align_corners=False)
        
        # Masked average pooling per segment
        # masks: (B, S, H, W) -> (B, S, 1, H, W)
        # features: (B, C, H, W) -> (B, 1, C, H, W)
        masks_exp = masks.unsqueeze(2)  # (B, S, 1, H, W)
        feats_exp = features.unsqueeze(1)  # (B, 1, C, H, W)
        
        # Weighted sum: (B, S, C, H, W) -> sum over H, W -> (B, S, C)
        weighted = (masks_exp * feats_exp).sum(dim=(-2, -1))  # (B, S, C)
        mask_sum = masks.sum(dim=(-2, -1)).unsqueeze(-1).clamp(min=1e-6)  # (B, S) -> (B, S, 1)
        pooled = weighted / mask_sum  # (B, S, C) / (B, S, 1) = (B, S, C)
        
        # Project to token dimension
        seg_tokens = self.proj(pooled)  # (B, S, token_dim)
        
        return seg_tokens


class AttentionBiasGenerator(nn.Module):
    """
    Generate attention bias from segment masks.
    Encourages attention within same segment, discourages across boundaries.
    """
    def __init__(self, num_heads: int = 4, temperature: float = 1.0):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = temperature
    
    def forward(
        self,
        masks: torch.Tensor,  # (B, S, H, W)
        target_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Generate attention bias matrix.
        
        Args:
            masks: Segment masks (B, S, H, W)
            target_hw: Target (H, W) for the attention, if different from masks
        
        Returns:
            attn_bias: (B, num_heads, H*W, H*W) attention bias
        """
        B, S, H, W = masks.shape
        
        if target_hw is not None and target_hw != (H, W):
            masks = F.interpolate(masks.float(), size=target_hw, mode='bilinear', align_corners=False)
            H, W = target_hw
        
        N = H * W
        
        # Flatten masks: (B, S, N)
        masks_flat = masks.view(B, S, N)
        
        # Compute segment affinity: (B, N, N)
        # High affinity if two positions belong to same segment
        # affinity[i,j] = sum_s(mask_s[i] * mask_s[j])
        affinity = torch.einsum('bsn,bsm->bnm', masks_flat, masks_flat)  # (B, N, N)
        
        # Normalize to [0, 1]
        affinity = affinity / (S + 1e-6)
        
        # Convert to attention bias (log-space for additive bias)
        # Same segment -> positive bias, different segment -> negative bias
        attn_bias = (affinity - 0.5) * 2 * self.temperature  # (B, N, N)
        
        # Expand for multi-head attention
        attn_bias = attn_bias.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # (B, H, N, N)
        
        return attn_bias


class CostModulator(nn.Module):
    """
    Modulate matching cost based on segment boundaries.
    Reduces cost at boundaries to allow for discontinuities.
    """
    def __init__(self, feat_dim: int, boundary_boost: float = 2.0):
        super().__init__()
        self.boundary_boost = boundary_boost
        
        # Learnable boundary-aware modulation
        self.boundary_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(32, feat_dim, 3, 1, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        features: torch.Tensor,  # (B, C, H, W)
        boundary_map: torch.Tensor,  # (B, 1, H, W)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Modulate features based on boundary map.
        
        Returns:
            modulated_features: Features with boundary information
            modulation_weights: Per-pixel modulation weights
        """
        # Resize boundary to feature resolution
        if boundary_map.shape[-2:] != features.shape[-2:]:
            boundary_map = F.interpolate(
                boundary_map, size=features.shape[-2:], 
                mode='bilinear', align_corners=False
            )
        
        # Generate modulation weights
        modulation = self.boundary_encoder(boundary_map)  # (B, C, H, W)
        
        # At boundaries: boost features for distinctiveness
        # Inside segments: normal features
        weights = 1.0 + self.boundary_boost * boundary_map
        
        modulated = features * modulation * weights
        
        return modulated, weights.mean(dim=1, keepdim=True)


class SAMGuidanceAdapterV3(nn.Module):
    """
    Enhanced SAM Guidance Adapter V3.
    
    Integrates SAM masks into optical flow estimation through 4 modes:
    1. feature_concat: Concatenate boundary features with encoder features
    2. attention_bias: Generate attention bias for transformer modules
    3. cost_modulation: Modulate matching cost at boundaries
    4. object_pooling: Pool features per segment for segment tokens
    
    Args:
        feat_dim: Feature dimension from encoder
        token_dim: Token dimension for transformer
        num_segments: Maximum number of segments
        num_heads: Number of attention heads for bias generation
        
        # Mode toggles
        use_feature_concat: Enable feature concatenation mode
        use_attention_bias: Enable attention bias generation
        use_cost_modulation: Enable cost modulation mode
        use_object_pooling: Enable object-level pooling
    """
    
    def __init__(
        self,
        feat_dim: int = 128,
        token_dim: int = 128,
        num_segments: int = 16,
        num_heads: int = 4,
        use_feature_concat: bool = True,
        use_attention_bias: bool = True,
        use_cost_modulation: bool = True,
        use_object_pooling: bool = True,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.token_dim = token_dim
        self.num_segments = num_segments
        
        # Mode flags
        self.use_feature_concat = use_feature_concat
        self.use_attention_bias = use_attention_bias
        self.use_cost_modulation = use_cost_modulation
        self.use_object_pooling = use_object_pooling
        
        # Mode 1: Feature Concatenation (SAMFlow style)
        if use_feature_concat:
            # Boundary feature encoder
            self.boundary_encoder = nn.Sequential(
                nn.Conv2d(1, 32, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(32, feat_dim // 2, 3, 1, 1),
                nn.GELU(),
            )
            # Residual mixing block
            self.mix_block = ResidualMixBlock(feat_dim + feat_dim // 2, feat_dim)
        
        # Mode 2: Attention Bias
        if use_attention_bias:
            self.attn_bias_gen = AttentionBiasGenerator(num_heads=num_heads)
        
        # Mode 3: Cost Modulation
        if use_cost_modulation:
            self.cost_modulator = CostModulator(feat_dim=feat_dim)
        
        # Mode 4: Object Pooling for Segment Tokens
        if use_object_pooling:
            self.seg_token_encoder = SegmentTokenEncoder(
                feat_dim=feat_dim,
                token_dim=token_dim,
                num_segments=num_segments,
            )
    
    def forward(
        self,
        masks: torch.Tensor,  # (B, T, S, H, W) or (B, S, H, W)
        features_per_time: List[torch.Tensor],  # List of T tensors, each (B, C, H, W)
        return_all: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Process SAM masks and generate guidance signals.
        
        Args:
            masks: SAM segment masks
            features_per_time: Encoder features for each frame
            return_all: If True, return all outputs; otherwise only essential ones
        
        Returns:
            Dictionary containing:
            - enhanced_features: List of enhanced features per time (if feature_concat)
            - boundary_maps: (B, T, 1, H, W) boundary maps
            - attn_bias: List of attention bias matrices per time (if attention_bias)
            - seg_tokens: (B, T, S, token_dim) segment tokens (if object_pooling)
            - modulation_weights: List of modulation weights per time (if cost_modulation)
        """
        # Handle 4D masks (single time step)
        if masks.dim() == 4:
            masks = masks.unsqueeze(1)  # Add time dimension
        
        B, T, S, H, W = masks.shape
        outputs = {}
        
        # Compute boundary maps for all frames
        boundary_maps = []
        for t in range(T):
            boundary = compute_boundary_map(masks[:, t])  # (B, 1, H, W)
            boundary_maps.append(boundary)
        boundary_maps_tensor = torch.stack(boundary_maps, dim=1)  # (B, T, 1, H, W)
        outputs['boundary_maps'] = boundary_maps_tensor
        
        # Mode 1: Feature Concatenation
        if self.use_feature_concat:
            enhanced_features = []
            for t, feat in enumerate(features_per_time):
                # Get boundary at feature resolution
                boundary_t = boundary_maps[t]
                if boundary_t.shape[-2:] != feat.shape[-2:]:
                    boundary_t = F.interpolate(
                        boundary_t, size=feat.shape[-2:],
                        mode='bilinear', align_corners=False
                    )
                
                # Encode boundary
                boundary_feat = self.boundary_encoder(boundary_t)  # (B, feat_dim//2, H, W)
                
                # Concatenate and mix
                concat = torch.cat([feat, boundary_feat], dim=1)  # (B, feat_dim + feat_dim//2, H, W)
                enhanced = self.mix_block(concat)  # (B, feat_dim, H, W)
                enhanced_features.append(enhanced)
            
            outputs['enhanced_features'] = enhanced_features
        
        # Mode 2: Attention Bias
        if self.use_attention_bias:
            attn_biases = []
            for t in range(min(T, len(features_per_time))):
                feat_hw = features_per_time[t].shape[-2:]
                bias = self.attn_bias_gen(masks[:, t], target_hw=feat_hw)
                attn_biases.append(bias)
            outputs['attn_bias'] = attn_biases
        
        # Mode 3: Cost Modulation
        if self.use_cost_modulation:
            modulated_features = []
            modulation_weights = []
            for t, feat in enumerate(features_per_time):
                mod_feat, weights = self.cost_modulator(feat, boundary_maps[t])
                modulated_features.append(mod_feat)
                modulation_weights.append(weights)
            outputs['modulated_features'] = modulated_features
            outputs['modulation_weights'] = modulation_weights
        
        # Mode 4: Object Pooling
        if self.use_object_pooling:
            seg_tokens = []
            for t, feat in enumerate(features_per_time):
                tokens = self.seg_token_encoder(masks[:, t], feat)  # (B, S, token_dim)
                seg_tokens.append(tokens)
            outputs['seg_tokens'] = torch.stack(seg_tokens, dim=1)  # (B, T, S, token_dim)
        
        # Compute edge map (legacy compatibility with V2)
        edge_maps = []
        for t in range(T):
            m_sum = masks[:, t].float().sum(dim=1, keepdim=True)  # (B, 1, H, W)
            edge = (m_sum - F.max_pool2d(m_sum, 3, 1, 1)).abs() > 0
            edge_maps.append(edge.float())
        outputs['edge_maps'] = torch.stack(edge_maps, dim=1)
        
        return outputs
    
    def get_legacy_output(
        self,
        masks: torch.Tensor,
        features_per_time: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Legacy output format compatible with V2 SamGuidanceAdapter.
        
        Returns:
            seg_tokens: (B, T, S, token_dim)
            edge_maps: (B, T, 1, H, W)
        """
        out = self.forward(masks, features_per_time, return_all=False)
        seg_tokens = out.get('seg_tokens', torch.zeros(1))
        edge_maps = out.get('edge_maps', torch.zeros(1))
        return seg_tokens, edge_maps


# ============= Convenience builder function =============
def build_sam_adapter_v3(cfg: Dict) -> SAMGuidanceAdapterV3:
    """Build SAM Guidance Adapter V3 from config."""
    sam_cfg = cfg.get('sam_guidance', {})
    model_cfg = cfg.get('model', {}).get('args', {})
    
    return SAMGuidanceAdapterV3(
        feat_dim=model_cfg.get('enc_channels', 64) * 2,  # Level 1/8 has 2x channels
        token_dim=model_cfg.get('token_dim', 128),
        num_segments=cfg.get('sam', {}).get('num_segments', 16),
        num_heads=model_cfg.get('lcm_heads', 4),
        use_feature_concat=sam_cfg.get('feature_concat', True),
        use_attention_bias=sam_cfg.get('attention_bias', True),
        use_cost_modulation=sam_cfg.get('cost_modulation', True),
        use_object_pooling=sam_cfg.get('object_pooling', True),
    )
