from __future__ import annotations
import torch
import torch.nn as nn
from einops import rearrange


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=4.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim*mlp_ratio)), nn.GELU(), nn.Linear(int(dim*mlp_ratio), dim))
    
    def forward(self, x):
        y, _ = self.attn(self.n1(x), self.n1(x), self.n1(x), need_weights=False)
        x = x + y
        return x + self.mlp(self.n2(x))


class LatentCostMemory(nn.Module):
    def __init__(self, token_dim=192, depth=6, heads=4):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(token_dim, heads) for _ in range(depth)])
        self.state = None
    
    def forward(self, tokens_per_level, seg_tokens=None, attn_bias=None):
        out_levels = []
        for lvl in tokens_per_level:
            lvl_lat = []
            for tok in lvl: # B D H W
                x = rearrange(tok, 'b d h w -> b (h w) d')
                for blk in self.blocks:
                    x = blk(x)
                if self.state is None or self.state.shape[1] != x.shape[1]:
                    self.state = x.detach()
                x = 0.8*x + 0.2*self.state
                self.state = x.detach()
                lvl_lat.append(rearrange(x, 'b n d -> b d n'))
            out_levels.append(lvl_lat)
        return out_levels