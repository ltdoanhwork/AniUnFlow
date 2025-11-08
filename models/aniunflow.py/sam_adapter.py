from __future__ import annotations
import torch
import torch.nn as nn
from einops import rearrange


class SamGuidanceAdapter(nn.Module):
    def __init__(self, feat_dim=128, token_dim=192):
        super().__init__()
        self.proj = nn.Linear(feat_dim, token_dim)

    def forward(self, masks, lvl8_feats):
        B = lvl8_feats[0].shape[0]
        T = len(lvl8_feats)
        S = masks.shape[2]
        tokens, edges = [], []
        for t in range(T):
            f = lvl8_feats[t]; C, H, W = f.shape[1:]
            m = torch.nn.functional.interpolate(masks[:, t].float(), size=(H, W), mode='nearest')
            m = rearrange(m, 'b s h w -> b s (h w)')
            f_flat = rearrange(f, 'b c h w -> b (h w) c')
            num = m.sum(-1, keepdim=True).clamp_min(1.0)
            seg = (m @ f_flat) / num # B S C
            tokens.append(self.proj(seg))
            m_img = m.sum(1).view(B, 1, H, W)
            edge = (m_img - torch.nn.functional.max_pool2d(m_img, 3, 1, 1)).abs() > 0
            edges.append(edge.float())
        return torch.stack(tokens, 1), torch.stack(edges, 1)