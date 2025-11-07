import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import Encoder, Decoder, CostVolume

class BaselineFNet(nn.Module):
    def __init__(self, hidden_dim=128, pyramid_levels=2, corr_radius=4):
        super().__init__()
        assert pyramid_levels in (2,)
        self.enc = Encoder(in_ch=3, base=hidden_dim // 4)
        self.cv = CostVolume(radius=corr_radius)
        self.refine = Decoder(base=hidden_dim // 4)

        # 81 channels đầu vào, 2 kênh đầu ra (u,v flow)
        self.h3_conv = nn.Conv2d(81, 2, kernel_size=1)
        self.h2_conv = nn.Conv2d(2, 2, kernel_size=1)
        self.h1_conv = nn.Conv2d(2, 2, kernel_size=1)

        self.register_buffer("mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std",  torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1), persistent=False)

    def forward(self, im1, im2):
        x1 = (im1 - self.mean) / self.std
        x2 = (im2 - self.mean) / self.std
        f1 = self.enc(x1)
        f2 = self.enc(x2)

        cv3 = self.cv(f1[2], f2[2])                # [8, 81, 92, 160]
        h3 = torch.tanh(self.h3_conv(cv3))         # [8, 2, 92, 160]

        h2 = F.interpolate(h3, scale_factor=2.0, mode='bilinear', align_corners=False)
        h2 = torch.tanh(self.h2_conv(h2))          # [8, 2, 184, 320]

        h1 = F.interpolate(h2, scale_factor=2.0, mode='bilinear', align_corners=False)
        h1 = torch.tanh(self.h1_conv(h1))          # [8, 2, 368, 640]

        flow = self.refine(f1, [h3, h2, h1])
        print(f"flow.shape: {flow.shape}")
        return {"flow": flow, "pyramid": [h3, h2, flow]}
