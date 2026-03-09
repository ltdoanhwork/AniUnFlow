from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _coords_grid(batch: int, ht: int, wd: int, device: torch.device) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(ht, device=device),
        torch.arange(wd, device=device),
        indexing="ij",
    )
    coords = torch.stack([x, y], dim=0).float()
    return coords.unsqueeze(0).repeat(batch, 1, 1, 1)


def _bilinear_sampler(img: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Sample img using pixel-space coordinates."""
    h, w = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / max(w - 1, 1) - 1
    ygrid = 2 * ygrid / max(h - 1, 1) - 1
    grid = torch.cat([xgrid, ygrid], dim=-1)
    return F.grid_sample(img, grid, align_corners=True)


class CorrLookup:
    """RAFT-style local lookup on an all-pairs correlation pyramid."""

    def __init__(
        self,
        fmap1: torch.Tensor,
        fmap2: torch.Tensor,
        num_levels: int = 4,
        radius: int = 4,
    ):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        corr = CorrLookup._corr(fmap1, fmap2)
        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)
        self.corr_pyramid.append(corr)

        for _ in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    @staticmethod
    def _corr(fmap1: torch.Tensor, fmap2: torch.Tensor) -> torch.Tensor:
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)
        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / math.sqrt(max(dim, 1))

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), axis=-1)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / (2**i)
            coords_lvl = centroid_lvl + delta.view(1, 2 * r + 1, 2 * r + 1, 2)

            corr = _bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()


class FlowHead(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(F.relu(self.conv1(x)))


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim: int = 128, input_dim: int = 256):
        super().__init__()
        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))

        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        return (1 - z) * h + z * q


class BasicMotionEncoder(nn.Module):
    def __init__(self, corr_levels: int = 4, corr_radius: int = 4):
        super().__init__()
        cor_planes = corr_levels * (2 * corr_radius + 1) ** 2
        self.convc1 = nn.Conv2d(cor_planes, 192, 1, padding=0)
        self.convc2 = nn.Conv2d(192, 128, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = nn.Conv2d(128 + 32, 126, 3, padding=1)

    def forward(self, flow: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        out = F.relu(self.conv(torch.cat([cor, flo], dim=1)))
        return torch.cat([out, flow], dim=1)


class SAMGatedMotionEncoder(nn.Module):
    def __init__(self, corr_levels: int, corr_radius: int, boundary_gate_strength: float = 0.3):
        super().__init__()
        self.encoder = BasicMotionEncoder(corr_levels, corr_radius)
        self.boundary_gate = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
        )
        self.boundary_gate_strength = float(boundary_gate_strength)

    def forward(
        self,
        flow: torch.Tensor,
        corr: torch.Tensor,
        boundary: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if boundary is not None:
            if boundary.dim() == 3:
                boundary = boundary.unsqueeze(1)
            if boundary.shape[2:] != corr.shape[2:]:
                boundary = F.interpolate(boundary.float(), size=corr.shape[2:], mode="bilinear", align_corners=False)
            gate = torch.sigmoid(self.boundary_gate(boundary.float()))
            corr = corr * (1.0 - self.boundary_gate_strength * gate)
        return self.encoder(flow, corr)


class HybridUpdateBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        context_dim: int = 128,
        corr_levels: int = 4,
        corr_radius: int = 4,
        boundary_gate_strength: float = 0.3,
    ):
        super().__init__()
        self.motion_encoder = SAMGatedMotionEncoder(
            corr_levels=corr_levels,
            corr_radius=corr_radius,
            boundary_gate_strength=boundary_gate_strength,
        )
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=context_dim + 128)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.mask_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 4 * 9, 1, padding=0),  # 2x convex upsampling
        )
        self.segment_gate = nn.Conv2d(1, context_dim + 128, 3, padding=1)
        self.segment_gate_strength = float(boundary_gate_strength)

    def forward(
        self,
        net: torch.Tensor,
        inp: torch.Tensor,
        corr: torch.Tensor,
        flow: torch.Tensor,
        boundary: Optional[torch.Tensor] = None,
        segment: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        motion_features = self.motion_encoder(flow, corr, boundary=boundary)
        gru_in = torch.cat([inp, motion_features], dim=1)

        if segment is not None:
            if segment.dim() == 3:
                segment = segment.unsqueeze(1)
            if segment.shape[2:] != gru_in.shape[2:]:
                segment = F.interpolate(segment.float(), size=gru_in.shape[2:], mode="nearest")
            gate = torch.sigmoid(self.segment_gate(segment.float()))
            gru_in = gru_in * (1.0 + self.segment_gate_strength * (gate - 0.5))

        net = self.gru(net, gru_in)
        delta_flow = self.flow_head(net)
        up_mask = 0.25 * self.mask_head(net)
        return net, up_mask, delta_flow


def _convex_upsample2(flow: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Convex upsampling from stride-8 flow to stride-4 flow."""
    n, _, h, w = flow.shape
    mask = mask.view(n, 1, 9, 2, 2, h, w)
    mask = torch.softmax(mask, dim=2)

    up_flow = F.unfold(2 * flow, [3, 3], padding=1)
    up_flow = up_flow.view(n, 2, 9, 1, 1, h, w)
    up_flow = torch.sum(mask * up_flow, dim=2)
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
    return up_flow.reshape(n, 2, 2 * h, 2 * w)


class IterativeFlowRefiner(nn.Module):
    """
    RAFT-style iterative refiner at stride-8 with optional convex upsample to stride-4.
    Global SAM-guided tokens are injected as a learned initial flow prior.
    """

    def __init__(
        self,
        feat_in_dim: int,
        context_in_dim: int,
        prior_dim: int,
        hidden_dim: int = 128,
        context_dim: int = 128,
        corr_levels: int = 4,
        corr_radius: int = 4,
        iters: int = 10,
        use_convex_upsampler: bool = True,
        boundary_gate_strength: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.iters = int(iters)
        self.use_convex_upsampler = bool(use_convex_upsampler)

        self.feature_proj = nn.Conv2d(feat_in_dim, 128, 1)
        self.context_encoder = nn.Sequential(
            nn.Conv2d(context_in_dim, hidden_dim + context_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim + context_dim, hidden_dim + context_dim, 3, padding=1),
        )
        self.prior_to_flow = nn.Sequential(
            nn.Conv2d(prior_dim, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 2, 3, padding=1),
        )
        self.update_block = HybridUpdateBlock(
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            corr_levels=corr_levels,
            corr_radius=corr_radius,
            boundary_gate_strength=boundary_gate_strength,
        )

    def forward(
        self,
        feat1: torch.Tensor,
        feat2: torch.Tensor,
        context_feat: torch.Tensor,
        prior: Optional[torch.Tensor] = None,
        boundary: Optional[torch.Tensor] = None,
        segment: Optional[torch.Tensor] = None,
        iters: Optional[int] = None,
    ) -> torch.Tensor:
        b, _, h, w = feat1.shape
        n_iters = int(iters) if iters is not None else self.iters

        fmap1 = F.normalize(self.feature_proj(feat1), dim=1)
        fmap2 = F.normalize(self.feature_proj(feat2), dim=1)
        corr_fn = CorrLookup(
            fmap1,
            fmap2,
            num_levels=self.corr_levels,
            radius=self.corr_radius,
        )

        cnet = self.context_encoder(context_feat)
        net, inp = torch.split(cnet, [self.hidden_dim, self.context_dim], dim=1)
        net = torch.tanh(net)
        inp = F.relu(inp)

        coords0 = _coords_grid(b, h, w, feat1.device)
        coords1 = _coords_grid(b, h, w, feat1.device)

        if prior is not None:
            if prior.shape[2:] != (h, w):
                prior = F.interpolate(prior, size=(h, w), mode="bilinear", align_corners=False)
            coords1 = coords1 + self.prior_to_flow(prior)

        up_mask = None
        for _ in range(n_iters):
            coords1_detached = coords1.detach()
            corr = corr_fn(coords1_detached)
            flow = coords1_detached - coords0
            net, up_mask, delta_flow = self.update_block(
                net,
                inp,
                corr,
                flow,
                boundary=boundary,
                segment=segment,
            )
            coords1 = coords1 + delta_flow

        flow8 = coords1 - coords0
        if self.use_convex_upsampler and up_mask is not None:
            return _convex_upsample2(flow8, up_mask)
        return 2.0 * F.interpolate(flow8, scale_factor=2.0, mode="bilinear", align_corners=True)
