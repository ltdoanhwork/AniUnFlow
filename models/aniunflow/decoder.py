from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import warp

class MSRecurrentDecoder(nn.Module):
    def __init__(self, chs=[64,128,192], iters_per_level=4):
        super().__init__()
        self.iters = iters_per_level
        self.refine1 = nn.Sequential(nn.Conv2d(chs[1]+2+chs[1], chs[1], 3,1,1), nn.GELU(), nn.Conv2d(chs[1], 2, 3,1,1))
        self.refine0 = nn.Sequential(nn.Conv2d(chs[0]+2+chs[0], chs[0], 3,1,1), nn.GELU(), nn.Conv2d(chs[0], 2, 3,1,1))
    def forward(self, coarse_flows, feats_levels, latent_levels, attn_bias=None):
        lvl1, lvl2, lvl3 = feats_levels
        outs = []
        for k, f8 in enumerate(coarse_flows):
            cur = f8
            # refine at 1/8
            for _ in range(self.iters):
                x = torch.cat([lvl2[k], cur, warp(lvl2[k+1], cur)], dim=1)
                cur = cur + self.refine1(x)
            # upsample to 1/4 and refine
            cur2 = F.interpolate(cur, scale_factor=2.0, mode='bilinear', align_corners=True)
            for _ in range(self.iters):
                x = torch.cat([lvl1[k], cur2, warp(lvl1[k+1], cur2)], dim=1)
                cur2 = cur2 + self.refine0(x)
            outs.append(cur2)
        return outs