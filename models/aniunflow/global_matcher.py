"""
Global Matching Module for Optical Flow
=========================================
Replaces local correlation with global all-pairs matching using Transformer.

Based on GMFlow (ICCV 2022) and FlowFormer++ (2024).

Key improvements over local correlation:
- Handles large displacements (>10 pixels)
- Global receptive field
- Learnable matching weights
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class GlobalMatcher(nn.Module):
    """
    Global Matching using Transformer-style attention.
    
    Instead of local correlation within a fixed window (±3px),
    computes all-pairs correlation globally across the entire feature map.
    
    Architecture:
        1. Project feat1 → Q, feat2 → K,V
        2. Compute similarity: S = softmax(Q @ K^T / sqrt(d))
        3. Aggregate: M = S @ V
        4. Project to tokens
    
    Args:
        dim: Feature dimension (C)
        token_dim: Output token dimension
        num_heads: Number of attention heads
        use_position_encoding: Whether to add positional encoding
    """
    
    def __init__(
        self,
        dim: int = 128,
        token_dim: int = 192,
        num_heads: int = 4,
        use_position_encoding: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads
        
        assert token_dim % num_heads == 0, "token_dim must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(dim, token_dim)
        self.k_proj = nn.Linear(dim, token_dim)
        self.v_proj = nn.Linear(dim, token_dim)
        
        # Output projection
        self.out_proj = nn.Linear(token_dim, token_dim)
        
        # Position encoding (learnable)
        self.use_position_encoding = use_position_encoding
        if use_position_encoding:
            # Will be initialized dynamically based on input size
            self.register_buffer("pos_enc", None)
            self.max_size = 128  # Maximum H or W
    
    def _get_position_encoding(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """
        Generate 2D sinusoidal position encoding.
        Returns: [H*W, token_dim]
        """
        if self.pos_enc is None or self.pos_enc.shape[0] != H*W:
            # Generate position encoding
            y_pos = torch.arange(H, device=device).unsqueeze(1).repeat(1, W)  # [H, W]
            x_pos = torch.arange(W, device=device).unsqueeze(0).repeat(H, 1)  # [H, W]
            
            y_pos = y_pos.flatten()  # [H*W]
            x_pos = x_pos.flatten()  # [H*W]
            
            # Sinusoidal encoding
            dim_half = self.token_dim // 2
            div_term = torch.exp(torch.arange(0, dim_half, 1, device=device).float() * 
                                 (-math.log(10000.0) / dim_half))
            
            pe = torch.zeros(H*W, self.token_dim, device=device)
            # Alternate sin/cos for x and y positions
            pe[:, 0:dim_half] = torch.sin(x_pos.unsqueeze(1) * div_term)
            pe[:, dim_half:] = torch.cos(y_pos.unsqueeze(1) * div_term[:self.token_dim-dim_half])
            
            self.pos_enc = pe
        
        return self.pos_enc
    
    def forward(
        self,
        feat1: torch.Tensor,  # [B, C, H, W]
        feat2: torch.Tensor,  # [B, C, H, W]
    ) -> torch.Tensor:
        """
        Compute global all-pairs matching between feat1 and feat2.
        
        Returns:
            match_tokens: [B, token_dim, H, W]
        """
        B, C, H, W = feat1.shape
        
        # Flatten spatial dimensions
        feat1_flat = rearrange(feat1, 'b c h w -> b (h w) c')  # [B, HW, C]
        feat2_flat = rearrange(feat2, 'b c h w -> b (h w) c')  # [B, HW, C]
        
        # Add position encoding if enabled
        if self.use_position_encoding:
            pos_enc = self._get_position_encoding(H, W, feat1.device)  # [HW, token_dim]
            # We add it after projection to Q,K
        
        # Project to Q, K, V
        Q = self.q_proj(feat1_flat)  # [B, HW, token_dim]
        K = self.k_proj(feat2_flat)  # [B, HW, token_dim]
        V = self.v_proj(feat2_flat)  # [B, HW, token_dim]
        
        # Add position encoding
        if self.use_position_encoding:
            Q = Q + pos_enc.unsqueeze(0)
            K = K + pos_enc.unsqueeze(0)
        
        # Reshape for multi-head attention
        Q = rearrange(Q, 'b n (h d) -> b h n d', h=self.num_heads)  # [B, H, HW, D]
        K = rearrange(K, 'b n (h d) -> b h n d', h=self.num_heads)
        V = rearrange(V, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # Compute attention scores
        # S = Q @ K^T / sqrt(d)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B, H, HW, HW]
        
        # Softmax to get attention weights
        attn = F.softmax(scores, dim=-1)  # [B, H, HW, HW]
        
        # Aggregate values
        out = torch.matmul(attn, V)  # [B, H, HW, D]
        
        # Merge heads
        out = rearrange(out, 'b h n d -> b n (h d)')  # [B, HW, token_dim]
        
        # Output projection
        out = self.out_proj(out)  # [B, HW, token_dim]
        
        # Reshape back to spatial
        match_tokens = rearrange(out, 'b (h w) c -> b c h w', h=H, w=W)  # [B, token_dim, H, W]
        
        return match_tokens


class GlobalMatchingTokenizer(nn.Module):
    """
    Multi-scale global matching tokenizer.
    
    Replacement for CostTokenizer that uses global matching instead of local correlation.
    Processes multiple resolution levels (1/4, 1/8, 1/16) from pyramid encoder.
    
    Args:
        dims: Feature dimensions at each level [64, 128, 192]
        token_dim: Output token dimension (default: 192)
        num_heads: Number of attention heads for matching
    """
    
    def __init__(
        self,
        dims=[64, 128, 192],
        token_dim=192,
        num_heads=4,
        topk=128,  # Top-k correspondences for efficiency
    ):
        super().__init__()
        self.dims = dims
        self.token_dim = token_dim
        
        # Use EfficientGlobalMatcher for memory efficiency
        self.matchers = nn.ModuleList([
            EfficientGlobalMatcher(dim=d, token_dim=token_dim, num_heads=num_heads, topk=topk)
            for d in dims
        ])
    
    def forward(self, feats_levels):
        """
        Args:
            feats_levels: List of 3 levels, each is a list of T frames
                          [[f0_l1, f1_l1, ...], [f0_l2, f1_l2, ...], [f0_l3, f1_l3, ...]]
        
        Returns:
            tokens: List of 3 levels, each is a list of T-1 matching tokens
                    [[tok_01_l1, tok_12_l1, ...], [tok_01_l2, ...], [tok_01_l3, ...]]
        """
        l1, l2, l3 = feats_levels
        
        tokens = []
        for level_feats, matcher in zip([l1, l2, l3], self.matchers):
            level_tokens = []
            for t in range(len(level_feats) - 1):
                # Match consecutive frames
                match_tok = matcher(level_feats[t], level_feats[t+1])
                level_tokens.append(match_tok)
            tokens.append(level_tokens)
        
        return tokens


# ============= Efficiency variant with sparse sampling =============
class EfficientGlobalMatcher(GlobalMatcher):
    """
    Memory-efficient variant using sparse sampling.
    
    Instead of full HW x HW attention, samples top-k correspondences.
    Based on "Efficient Optical Flow via Edge-Preserving Cascading" (2024).
    
    Reduces memory from O(H^2*W^2) to O(H*W*k) where k << HW.
    """
    
    def __init__(
        self,
        dim: int = 128,
        token_dim: int = 192,
        num_heads: int = 4,
        topk: int = 256,  # Number of top correspondences to keep
        use_position_encoding: bool = True,
    ):
        super().__init__(dim, token_dim, num_heads, use_position_encoding)
        self.topk = topk
    
    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """
        Compute global matching with top-k sparse sampling.
        Uses chunked processing to avoid large memory allocation.
        """
        B, C, H, W = feat1.shape
        
        feat1_flat = rearrange(feat1, 'b c h w -> b (h w) c')
        feat2_flat = rearrange(feat2, 'b c h w -> b (h w) c')
        
        Q = self.q_proj(feat1_flat)
        K = self.k_proj(feat2_flat)
        V = self.v_proj(feat2_flat)
        
        if self.use_position_encoding:
            pos_enc = self._get_position_encoding(H, W, feat1.device)
            Q = Q + pos_enc.unsqueeze(0)
            K = K + pos_enc.unsqueeze(0)
        
        Q = rearrange(Q, 'b n (h d) -> b h n d', h=self.num_heads)
        K = rearrange(K, 'b n (h d) -> b h n d', h=self.num_heads)
        V = rearrange(V, 'b n (h d) -> b h n d', h=self.num_heads)
        
        scale = 1.0 / math.sqrt(self.head_dim)
        
        # Process in chunks to save memory
        chunk_size = 512  # Process 512 queries at a time
        N = H * W
        out_chunks = []
        
        for i in range(0, N, chunk_size):
            end_i = min(i + chunk_size, N)
            Q_chunk = Q[:, :, i:end_i]  # [B, H, chunk, D]
            
            # Compute scores for this chunk
            scores_chunk = torch.matmul(Q_chunk, K.transpose(-2, -1)) * scale  # [B, H, chunk, HW]
            
            # Top-k sampling
            topk = min(self.topk, scores_chunk.shape[-1])
            topk_scores, topk_indices = torch.topk(scores_chunk, k=topk, dim=-1)  # [B, H, chunk, k]
            
            # Softmax over top-k
            attn_chunk = F.softmax(topk_scores, dim=-1)  # [B, H, chunk, k]
            
            # Gather top-k values
            # Expand V for gathering: [B, H, 1, HW, D] -> [B, H, chunk, k, D]
            topk_indices_exp = topk_indices.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim)
            V_expanded = V.unsqueeze(2).expand(-1, -1, end_i - i, -1, -1)
            V_topk = torch.gather(V_expanded, 3, topk_indices_exp)  # [B, H, chunk, k, D]
            
            # Weighted sum
            out_chunk = (attn_chunk.unsqueeze(-1) * V_topk).sum(dim=-2)  # [B, H, chunk, D]
            out_chunks.append(out_chunk)
        
        # Concatenate chunks
        out = torch.cat(out_chunks, dim=2)  # [B, H, N, D]
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.out_proj(out)
        match_tokens = rearrange(out, 'b (h w) c -> b c h w', h=H, w=W)
        
        return match_tokens
