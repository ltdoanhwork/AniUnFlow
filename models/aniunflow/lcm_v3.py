"""
Latent Cost Memory V3
======================
Enhanced LCM with segment cross-attention.

Key improvements over V1:
1. Actually USES seg_tokens (V1 ignores them!)
2. Cross-attention between cost tokens and segment tokens
3. SAM-conditioned memory updates
4. Optional attention bias injection
"""
from __future__ import annotations
from typing import List, Optional
import torch
import torch.nn as nn
from einops import rearrange


class TransformerBlock(nn.Module):
    """Standard transformer block with pre-norm."""
    def __init__(self, dim: int, heads: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_norm = self.n1(x)
        y, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask, need_weights=False)
        x = x + y
        return x + self.mlp(self.n2(x))


class SegmentCrossAttention(nn.Module):
    """
    Cross-attention between cost tokens and segment tokens.
    Allows cost tokens to attend to segment information.
    """
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_out = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
        
        # Gate to control segment influence
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        cost_tokens: torch.Tensor,  # (B, N, D)
        seg_tokens: torch.Tensor,   # (B, S, D)
    ) -> torch.Tensor:
        """
        Cross-attention from cost tokens to segment tokens.
        
        Returns:
            updated_tokens: (B, N, D)
        """
        Q = self.norm_q(cost_tokens)
        KV = self.norm_kv(seg_tokens)
        
        # Cross-attention: cost tokens query segment tokens
        attended, _ = self.cross_attn(Q, KV, KV, need_weights=False)
        
        # Gated residual connection
        gate = self.gate(attended)
        attended = self.proj(self.norm_out(attended))
        
        return cost_tokens + gate * attended


class LatentCostMemoryV3(nn.Module):
    """
    Latent Cost Memory V3 with segment cross-attention.
    
    Key improvements:
    1. Uses seg_tokens via cross-attention (V1 ignored them!)
    2. Supports attention bias from SAM boundaries
    3. Improved memory mechanism with momentum
    4. Per-level processing with skip connections
    
    Args:
        token_dim: Token dimension
        depth: Number of transformer blocks
        heads: Number of attention heads
        use_segment_cross_attn: Enable segment cross-attention
        memory_momentum: Momentum for memory update (0.8 = 80% current, 20% memory)
    """
    
    def __init__(
        self,
        token_dim: int = 192,
        depth: int = 6,
        heads: int = 4,
        use_segment_cross_attn: bool = True,
        memory_momentum: float = 0.8,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.depth = depth
        self.memory_momentum = memory_momentum
        self.use_segment_cross_attn = use_segment_cross_attn
        
        # Main transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(token_dim, heads) for _ in range(depth)
        ])
        
        # Segment cross-attention (after every 2 blocks)
        if use_segment_cross_attn:
            self.seg_cross_attn = nn.ModuleList([
                SegmentCrossAttention(token_dim, heads)
                for _ in range(depth // 2)
            ])
        
        # Memory state (detached, not trainable)
        self.register_buffer('memory_state', None, persistent=False)
    
    def reset_memory(self):
        """Reset memory state."""
        self.memory_state = None
    
    def forward(
        self,
        tokens_per_level: List[List[torch.Tensor]],
        seg_tokens: Optional[torch.Tensor] = None,  # (B, T, S, D) or (B, S, D)
        attn_bias: Optional[List[torch.Tensor]] = None,  # Per-time attention biases
    ) -> List[List[torch.Tensor]]:
        """
        Process cost tokens with segment guidance.
        
        Args:
            tokens_per_level: 3 levels, each with T-1 tokens (B, D, H, W)
            seg_tokens: Segment tokens from SAM adapter
            attn_bias: Attention biases from SAM adapter
        
        Returns:
            out_levels: Same structure as input, processed tokens (B, D, N)
        """
        out_levels = []
        
        for lvl_idx, level_tokens in enumerate(tokens_per_level):
            lvl_out = []
            
            for t, tok in enumerate(level_tokens):
                # Reshape: (B, D, H, W) -> (B, N, D)
                B, D, H, W = tok.shape
                x = rearrange(tok, 'b d h w -> b (h w) d')
                
                # Get segment tokens for this time step
                seg_tok_t = None
                if seg_tokens is not None:
                    if seg_tokens.dim() == 4:  # (B, T, S, D)
                        if t < seg_tokens.shape[1]:
                            seg_tok_t = seg_tokens[:, t]  # (B, S, D)
                    elif seg_tokens.dim() == 3:  # (B, S, D)
                        seg_tok_t = seg_tokens
                
                # Get attention bias for this time step
                bias_t = None
                if attn_bias is not None and t < len(attn_bias):
                    bias_t = attn_bias[t]
                    # Convert to attention mask format if needed
                    # (B, H, N, N) -> (B*H, N, N) for nn.MultiheadAttention
                    if bias_t is not None and bias_t.dim() == 4:
                        # Average over heads for compatibility
                        bias_t = bias_t.mean(dim=1)  # (B, N, N)
                
                # Process through transformer blocks
                seg_attn_idx = 0
                for block_idx, block in enumerate(self.blocks):
                    x = block(x, attn_mask=bias_t)
                    
                    # Apply segment cross-attention every 2 blocks
                    if self.use_segment_cross_attn and seg_tok_t is not None:
                        if (block_idx + 1) % 2 == 0 and seg_attn_idx < len(self.seg_cross_attn):
                            x = self.seg_cross_attn[seg_attn_idx](x, seg_tok_t)
                            seg_attn_idx += 1
                
                # Memory update with momentum
                if self.memory_state is None or self.memory_state.shape != x.shape:
                    self.memory_state = x.detach()
                else:
                    # Fuse with memory
                    x = self.memory_momentum * x + (1 - self.memory_momentum) * self.memory_state
                    self.memory_state = x.detach()
                
                # Output: (B, N, D) -> (B, D, N)
                out = rearrange(x, 'b n d -> b d n')
                lvl_out.append(out)
            
            out_levels.append(lvl_out)
        
        return out_levels


# ============= Builder =============
def build_lcm_v3(cfg: dict) -> LatentCostMemoryV3:
    """Build LCM V3 from config."""
    model_cfg = cfg.get('model', {}).get('args', {})
    sam_cfg = cfg.get('sam_guidance', {})
    
    return LatentCostMemoryV3(
        token_dim=model_cfg.get('token_dim', 192),
        depth=model_cfg.get('lcm_depth', 6),
        heads=model_cfg.get('lcm_heads', 4),
        use_segment_cross_attn=sam_cfg.get('attention_bias', True),
    )
