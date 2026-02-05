"""
SAM-Guided Global Matcher V3
============================
Global matching with SAM-aware cost volume modulation.

Key improvements over V1 GlobalMatchingTokenizer:
1. Segment-aware attention: Boost attention within same segment
2. Boundary penalty: Reduce matching confidence at boundaries
3. Motion consistency prior: Encourage similar flow within segments

Based on:
- GMFlow (ICCV 2022): Global matching with attention
- SAMFlow (AAAI 2024): SAM feature integration
- FlowFormer++ (2024): Efficient token-based matching
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Import from existing global_matcher for reuse
from .global_matcher import GlobalMatcher, EfficientGlobalMatcher


class SAMGuidedGlobalMatcher(nn.Module):
    """
    Global matching with SAM-guided cost modulation.
    
    Extends EfficientGlobalMatcher with:
    1. Boundary-aware matching: Reduce cost at segment boundaries
    2. Segment affinity integration: Boost matches within segments
    3. Optional attention bias injection
    
    Args:
        dim: Input feature dimension
        token_dim: Output token dimension
        num_heads: Number of attention heads
        topk: Top-k correspondences for efficiency
        use_boundary_modulation: Modulate cost at boundaries
        use_segment_affinity: Boost within-segment matches
        boundary_penalty: How much to reduce cost at boundaries
    """
    
    def __init__(
        self,
        dim: int = 128,
        token_dim: int = 192,
        num_heads: int = 4,
        topk: int = 256,
        use_position_encoding: bool = True,
        use_boundary_modulation: bool = True,
        use_segment_affinity: bool = True,
        boundary_penalty: float = 0.5,
        affinity_boost: float = 0.3,
    ):
        super().__init__()
        self.dim = dim
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads
        self.topk = topk
        
        # Modulation options
        self.use_boundary_modulation = use_boundary_modulation
        self.use_segment_affinity = use_segment_affinity
        self.boundary_penalty = boundary_penalty
        self.affinity_boost = affinity_boost
        
        assert token_dim % num_heads == 0
        
        # QKV projections
        self.q_proj = nn.Linear(dim, token_dim)
        self.k_proj = nn.Linear(dim, token_dim)
        self.v_proj = nn.Linear(dim, token_dim)
        self.out_proj = nn.Linear(token_dim, token_dim)
        
        # Position encoding
        self.use_position_encoding = use_position_encoding
        if use_position_encoding:
            self.register_buffer("pos_enc", None)
        
        # Boundary feature integration
        if use_boundary_modulation:
            self.boundary_proj = nn.Sequential(
                nn.Conv2d(1, 32, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(32, num_heads, 1),  # Per-head boundary weight
            )
        
        # Segment affinity integration
        if use_segment_affinity:
            self.affinity_scale = nn.Parameter(torch.ones(num_heads) * affinity_boost)
    
    def _get_position_encoding(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Generate 2D sinusoidal position encoding."""
        if self.pos_enc is None or self.pos_enc.shape[0] != H * W:
            y_pos = torch.arange(H, device=device).unsqueeze(1).repeat(1, W)
            x_pos = torch.arange(W, device=device).unsqueeze(0).repeat(H, 1)
            
            y_pos = y_pos.flatten()
            x_pos = x_pos.flatten()
            
            dim_half = self.token_dim // 2
            div_term = torch.exp(
                torch.arange(0, dim_half, 1, device=device).float() *
                (-math.log(10000.0) / dim_half)
            )
            
            pe = torch.zeros(H * W, self.token_dim, device=device)
            pe[:, 0:dim_half] = torch.sin(x_pos.unsqueeze(1) * div_term)
            pe[:, dim_half:] = torch.cos(y_pos.unsqueeze(1) * div_term[:self.token_dim - dim_half])
            
            self.pos_enc = pe
        
        return self.pos_enc
    
    def forward(
        self,
        feat1: torch.Tensor,  # (B, C, H, W)
        feat2: torch.Tensor,  # (B, C, H, W)
        boundary_map: Optional[torch.Tensor] = None,  # (B, 1, H, W)
        segment_affinity: Optional[torch.Tensor] = None,  # (B, N, N) or (B, H, N, N)
        attn_bias: Optional[torch.Tensor] = None,  # (B, H, N, N)
    ) -> torch.Tensor:
        """
        Compute SAM-guided global matching.
        
        Args:
            feat1: Source frame features
            feat2: Target frame features
            boundary_map: Segment boundary map (1 at boundaries)
            segment_affinity: Pre-computed segment affinity matrix
            attn_bias: External attention bias from SAM adapter
        
        Returns:
            match_tokens: (B, token_dim, H, W)
        """
        B, C, H, W = feat1.shape
        N = H * W
        
        # Flatten features
        feat1_flat = rearrange(feat1, 'b c h w -> b (h w) c')
        feat2_flat = rearrange(feat2, 'b c h w -> b (h w) c')
        
        # Project to QKV
        Q = self.q_proj(feat1_flat)  # (B, N, token_dim)
        K = self.k_proj(feat2_flat)
        V = self.v_proj(feat2_flat)
        
        # Add position encoding
        if self.use_position_encoding:
            pos_enc = self._get_position_encoding(H, W, feat1.device)
            Q = Q + pos_enc.unsqueeze(0)
            K = K + pos_enc.unsqueeze(0)
        
        # Reshape for multi-head attention
        Q = rearrange(Q, 'b n (h d) -> b h n d', h=self.num_heads)
        K = rearrange(K, 'b n (h d) -> b h n d', h=self.num_heads)
        V = rearrange(V, 'b n (h d) -> b h n d', h=self.num_heads)
        
        scale = 1.0 / math.sqrt(self.head_dim)
        
        # Chunked processing for memory efficiency
        chunk_size = 512
        out_chunks = []
        
        for i in range(0, N, chunk_size):
            end_i = min(i + chunk_size, N)
            Q_chunk = Q[:, :, i:end_i]  # (B, H, chunk, D)
            
            # Compute attention scores
            scores = torch.matmul(Q_chunk, K.transpose(-2, -1)) * scale  # (B, H, chunk, N)
            
            # Apply boundary modulation
            if self.use_boundary_modulation and boundary_map is not None:
                boundary_weights = self._get_boundary_weights(boundary_map, H, W, i, end_i)
                scores = scores - self.boundary_penalty * boundary_weights
            
            # Apply segment affinity
            if self.use_segment_affinity and segment_affinity is not None:
                affinity_chunk = self._get_affinity_chunk(segment_affinity, i, end_i)
                scores = scores + self.affinity_scale.view(1, -1, 1, 1) * affinity_chunk
            
            # Apply external attention bias
            if attn_bias is not None:
                bias_chunk = attn_bias[:, :, i:end_i, :]  # (B, H, chunk, N)
                scores = scores + bias_chunk
            
            # Top-k sampling for efficiency
            topk = min(self.topk, scores.shape[-1])
            topk_scores, topk_indices = torch.topk(scores, k=topk, dim=-1)
            
            # Softmax over top-k
            attn = F.softmax(topk_scores, dim=-1)  # (B, H, chunk, k)
            
            # Gather top-k values
            topk_indices_exp = topk_indices.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim)
            V_expanded = V.unsqueeze(2).expand(-1, -1, end_i - i, -1, -1)
            V_topk = torch.gather(V_expanded, 3, topk_indices_exp)
            
            # Weighted sum
            out_chunk = (attn.unsqueeze(-1) * V_topk).sum(dim=-2)  # (B, H, chunk, D)
            out_chunks.append(out_chunk)
        
        # Concatenate chunks
        out = torch.cat(out_chunks, dim=2)  # (B, H, N, D)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.out_proj(out)
        
        # Reshape to spatial
        match_tokens = rearrange(out, 'b (h w) c -> b c h w', h=H, w=W)
        
        return match_tokens
    
    def _get_boundary_weights(
        self,
        boundary_map: torch.Tensor,
        H: int, W: int,
        start: int, end: int,
    ) -> torch.Tensor:
        """Get boundary weights for attention chunk."""
        B = boundary_map.shape[0]
        
        # Resize boundary map if needed
        if boundary_map.shape[-2:] != (H, W):
            boundary_map = F.interpolate(
                boundary_map, size=(H, W),
                mode='bilinear', align_corners=False
            )
        
        # Project boundary to per-head weights
        boundary_weights = self.boundary_proj(boundary_map)  # (B, num_heads, H, W)
        boundary_flat = rearrange(boundary_weights, 'b h hh ww -> b h (hh ww)')  # (B, H, N)
        
        # Get chunk
        boundary_chunk = boundary_flat[:, :, start:end]  # (B, H, chunk)
        
        # Create pairwise boundary cost (additive in both directions)
        # High cost if either source or target is at boundary
        boundary_cost = boundary_chunk.unsqueeze(-1) + boundary_flat.unsqueeze(2)  # (B, H, chunk, N)
        
        return boundary_cost
    
    def _get_affinity_chunk(
        self,
        segment_affinity: torch.Tensor,
        start: int, end: int,
    ) -> torch.Tensor:
        """Get segment affinity for attention chunk."""
        if segment_affinity.dim() == 3:
            # (B, N, N) -> expand to (B, H, N, N)
            affinity = segment_affinity.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        else:
            affinity = segment_affinity
        
        return affinity[:, :, start:end, :]  # (B, H, chunk, N)


class SAMGuidedGlobalMatchingTokenizer(nn.Module):
    """
    Multi-scale SAM-guided global matching tokenizer.
    
    Replacement for GlobalMatchingTokenizer with SAM integration.
    Processes 3 resolution levels (1/4, 1/8, 1/16).
    """
    
    def __init__(
        self,
        dims: List[int] = [64, 128, 192],
        token_dim: int = 192,
        num_heads: int = 4,
        topk: int = 128,
        use_boundary_modulation: bool = True,
        use_segment_affinity: bool = True,
    ):
        super().__init__()
        self.dims = dims
        self.token_dim = token_dim
        
        self.matchers = nn.ModuleList([
            SAMGuidedGlobalMatcher(
                dim=d,
                token_dim=token_dim,
                num_heads=num_heads,
                topk=topk,
                use_boundary_modulation=use_boundary_modulation,
                use_segment_affinity=use_segment_affinity,
            )
            for d in dims
        ])
    
    def forward(
        self,
        feats_levels: List[List[torch.Tensor]],
        boundary_maps: Optional[List[torch.Tensor]] = None,
        attn_biases: Optional[List[torch.Tensor]] = None,
    ) -> List[List[torch.Tensor]]:
        """
        Args:
            feats_levels: 3 levels, each with T frames
            boundary_maps: T-1 boundary maps (for consecutive pairs)
            attn_biases: T-1 attention biases
        
        Returns:
            tokens: 3 levels, each with T-1 matching tokens
        """
        l1, l2, l3 = feats_levels
        
        tokens = []
        for lvl_idx, (level_feats, matcher) in enumerate(zip([l1, l2, l3], self.matchers)):
            level_tokens = []
            for t in range(len(level_feats) - 1):
                # Get boundary and bias for this pair
                boundary = None
                if boundary_maps is not None and t < len(boundary_maps):
                    boundary = boundary_maps[t]
                    # Resize to this level's resolution
                    feat_hw = level_feats[t].shape[-2:]
                    if boundary.shape[-2:] != feat_hw:
                        boundary = F.interpolate(
                            boundary, size=feat_hw,
                            mode='bilinear', align_corners=False
                        )
                
                attn_bias = None
                if attn_biases is not None and t < len(attn_biases):
                    attn_bias = attn_biases[t]
                    # May need to resize based on level
                
                match_tok = matcher(
                    level_feats[t],
                    level_feats[t + 1],
                    boundary_map=boundary,
                    attn_bias=attn_bias,
                )
                level_tokens.append(match_tok)
            
            tokens.append(level_tokens)
        
        return tokens


# ============= Convenience builder =============
def build_sam_guided_matcher(cfg: Dict) -> SAMGuidedGlobalMatchingTokenizer:
    """Build SAM-guided global matching tokenizer from config."""
    model_cfg = cfg.get('model', {}).get('args', {})
    sam_cfg = cfg.get('sam_guidance', {})
    
    c = model_cfg.get('enc_channels', 64)
    dims = [c, c * 2, c * 3]
    
    return SAMGuidedGlobalMatchingTokenizer(
        dims=dims,
        token_dim=model_cfg.get('token_dim', 192),
        num_heads=4,
        topk=128,
        use_boundary_modulation=sam_cfg.get('cost_modulation', True),
        use_segment_affinity=sam_cfg.get('attention_bias', True),
    )
