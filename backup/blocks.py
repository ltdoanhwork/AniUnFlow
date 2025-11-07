import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

class Encoder(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.c1 = ConvBlock(in_ch, base)
        self.c2 = ConvBlock(base, base*2, s=2)  # /2
        self.c3 = ConvBlock(base*2, base*4, s=2) # /4
    def forward(self, x):
        f1 = self.c1(x)
        f2 = self.c2(f1)
        f3 = self.c3(f2)
        return [f1, f2, f3]

class Decoder(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.u2 = ConvBlock(base*4 + 2, base*2)   # concat corr or flow hints
        self.u1 = ConvBlock(base*2 + 2, base)
        # self.out = nn.Conv2d(base, 2, 3, 1, 1)
        self.out = nn.Conv2d(base + 2, 2, 3, 1, 1)
    def forward(self, feats, hints):
        # feats: [f1, f2, f3], hints: [h3,h2,h1] (2-ch flow hints per scale)
        f1, f2, f3 = feats
        h3, h2, h1 = hints
        x = torch.cat([f3, h3], dim=1)
        x = F.interpolate(self.u2(x), scale_factor=2.0, mode='bilinear', align_corners=False)
        x = torch.cat([x, h2], dim=1)
        x = F.interpolate(self.u1(x), scale_factor=2.0, mode='bilinear', align_corners=False)
        x = torch.cat([x, h1], dim=1)

        flow = self.out(x)
        return flow

class CostVolume(nn.Module):
    """Naive correlation via local window (no custom CUDA)."""
    def __init__(self, radius: int = 4):
        super().__init__()
        self.radius = radius
    def forward(self, f1, f2):
        B, C, H, W = f1.shape
        r = self.radius
        vols = []
        for dy in range(-r, r+1):
            for dx in range(-r, r+1):
                f2s = torch.roll(f2, shifts=(dy, dx), dims=(2, 3))
                vols.append((f1 * f2s).sum(1, keepdim=True))
        vol = torch.cat(vols, dim=1)  # [B, (2r+1)^2, H, W]
        return vol