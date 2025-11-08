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


class GlobalTemporalRegressor(nn.Module):
    def __init__(self, token_dim=192, heads=4, depth=2):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(token_dim, heads) for _ in range(depth)])
        self.head = nn.Sequential(nn.Linear(token_dim, token_dim), nn.GELU(), nn.Linear(token_dim, 2))
    
    def forward(self, latent_levels, feats_levels):
        lvl = 1
        per_pair = latent_levels[lvl]
        flows = []
        for k, lat in enumerate(per_pair):
            x = rearrange(lat, 'b d n -> b n d')
            for blk in self.blocks:
                x = blk(x)
            vec = self.head(x)
            H, W = feats_levels[lvl][k].shape[-2:]
            flow = rearrange(vec, 'b (h w) c -> b c h w', h=H, w=W)
            flows.append(flow)
        return flows