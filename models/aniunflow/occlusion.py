from __future__ import annotations
import torch.nn as nn


class OcclusionHead(nn.Module):
    def __init__(self, in_ch=128):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, in_ch, 3,1,1), nn.GELU(), nn.Conv2d(in_ch, 1, 1))
    
    def forward(self, feat):
        return self.net(feat)