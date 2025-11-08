from __future__ import annotations
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, 1, 1), nn.GroupNorm(8, c_out), nn.GELU(),
            nn.Conv2d(c_out, c_out, 3, 1, 1), nn.GroupNorm(8, c_out), nn.GELU(),
        )
    def forward(self, x):
        return self.net(x)


class PyramidEncoder(nn.Module):
    def __init__(self, c=64):
        super().__init__()
        self.lvl1 = nn.Sequential(
            nn.Conv2d(3, c, 3, 2, 1), nn.GELU(),
            nn.Conv2d(c, c, 3, 2, 1), nn.GELU(),
            ConvBlock(c, c)
        )
        self.lvl2 = nn.Sequential(nn.Conv2d(c, c*2, 3, 2, 1), nn.GELU(), ConvBlock(c*2, c*2))
        self.lvl3 = nn.Sequential(nn.Conv2d(c*2, c*3, 3, 2, 1), nn.GELU(), ConvBlock(c*3, c*3))
        self.cache = {}


    def forward(self, frames, use_cache=True):
        B, T, C, H, W = frames.shape
        f1s, f2s, f3s = [], [], []
        for t in range(T):
            x = frames[:, t]
            f1 = self.lvl1(x)
            f2 = self.lvl2(f1)
            key = f"lvl3_{t-1}"
            if use_cache and key in self.cache:
                f3 = self.lvl3[0](f2); f3 = self.lvl3[1](f3 + 0.0*self.cache[key])
            else:
                f3 = self.lvl3(f2)
            self.cache[f"lvl3_{t}"] = f3.detach()
            f1s.append(f1); f2s.append(f2); f3s.append(f3)
        return [f1s, f2s, f3s]