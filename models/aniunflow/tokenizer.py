from __future__ import annotations
import math
import torch
import torch.nn as nn


class CostTokenizer(nn.Module):
    def __init__(self, dims=[64,128,192], token_dim=192):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d((2*3+1)**2, token_dim, 1) for _ in dims])

    @staticmethod
    def local_correlation(f1, f2, r=3):
        B, C, H, W = f1.shape
        vols = []
        for dy in range(-r, r+1):
            for dx in range(-r, r+1):
                vols.append((f1 * torch.roll(f2, (dy,dx), (2,3))).sum(1, keepdim=True)/math.sqrt(C))
        return torch.cat(vols, 1)

    def _level(self, feats, proj, r=3):
        toks = []
        for t in range(len(feats)-1):
            corr = self.local_correlation(feats[t], feats[t+1], r)
            toks.append(proj(corr))
        return toks

    def forward(self, feats_levels):
        l1, l2, l3 = feats_levels
        return [self._level(l1, self.proj[0]), self._level(l2, self.proj[1]), self._level(l3, self.proj[2])]
