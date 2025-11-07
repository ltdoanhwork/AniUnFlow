"""
AniFlowFormer-T: Temporal Cost-Memory Transformer for Un/Semi-supervised Optical Flow in Animation
-------------------------------------------------------------------------------------------------
- Multi-frame one-pass prediction
- FlowFormer-style cost tokens -> Latent Cost Memory (spatio-temporal)
- Global Temporal Regressor (GTR) for temporal aggregation
- SIM-style feature/memory reuse (KV-cache) for streaming
- Optional SAM-guidance: segment tokens + boundary-aware weighting

This is a self-contained reference skeleton intended for research prototyping.
All comments are in English per user's preference. Replace stub parts with your own data and utilities as needed.

Author: Doanh & ChatGPT (2025-11-06)
Python >=3.9, PyTorch >=2.1
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from einops import rearrange, reduce, repeat
except Exception:
    # Fallback minimal wrappers
    def rearrange(x, pattern, **kwargs):
        raise ImportError("Install einops for rearrange/reduce/repeat")
    def reduce(x, pattern, reduction):
        raise ImportError("Install einops for rearrange/reduce/repeat")
    def repeat(x, pattern, **kwargs):
        raise ImportError("Install einops for rearrange/reduce/repeat")

# ---------------------------------------------------------------
# Utility: Flow warp, SSIM, image gradients
# ---------------------------------------------------------------

def meshgrid_xy(B: int, H: int, W: int, device: torch.device):
    """Create normalized [-1,1] sampling grid in (x,y)."""
    ys, xs = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, device=device),
        torch.linspace(-1.0, 1.0, W, device=device),
        indexing="ij",
    )
    grid = torch.stack([xs, ys], dim=-1)  # H W 2
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)
    return grid


def flow_to_norm_grid(flow: torch.Tensor) -> torch.Tensor:
    """Convert flow (pixels) to normalized grid additive offsets.
    flow: Bx2xHxW (dx, dy) in pixels
    return: BxHxWx2 offsets in [-1,1]
    """
    B, _, H, W = flow.shape
    # scale from pixels to normalized [-1,1]
    # grid_sample expects normalized coords; x scaled by (2/W), y by (2/H)
    dx = flow[:, 0:1] * (2.0 / max(W - 1, 1))
    dy = flow[:, 1:1+1] * (2.0 / max(H - 1, 1))
    offsets = torch.stack([dx, dy], dim=-1)  # B,1,2,H,W -> incorrect
    # fix shape
    offsets = torch.cat([dx, dy], dim=1)  # B,2,H,W
    offsets = rearrange(offsets, 'b c h w -> b h w c')
    return offsets


def warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp img by flow (pixels). img: BxCxHxW, flow: Bx2xHxW."""
    B, C, H, W = img.shape
    base_grid = meshgrid_xy(B, H, W, img.device)
    norm_offsets = flow_to_norm_grid(flow)
    grid = base_grid + norm_offsets  # B H W 2
    return F.grid_sample(img, grid, align_corners=True, mode='bilinear', padding_mode='border')


class SSIM(nn.Module):
    """Simple SSIM for photometric loss (window 3x3)."""
    def __init__(self, channels: int = 3):
        super().__init__()
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
        self.pool = nn.AvgPool2d(3, 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mu_x = self.pool(x)
        mu_y = self.pool(y)
        sigma_x = self.pool(x * x) - mu_x * mu_x
        sigma_y = self.pool(y * y) - mu_y * mu_y
        sigma_xy = self.pool(x * y) - mu_x * mu_y
        ssim = ((2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)) / \
               ((mu_x * mu_x + mu_y * mu_y + self.C1) * (sigma_x + sigma_y + self.C2) + 1e-8)
        return torch.clamp((1 - ssim) / 2, 0, 1)


def image_gradients(img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Finite differences (Sobel-lite). img: BxCxHxW -> grads per-channel."""
    dx = img[..., :, 1:] - img[..., :, :-1]
    dy = img[..., 1:, :] - img[..., :-1, :]
    # pad to original size
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return dx, dy


# ---------------------------------------------------------------
# Feature Pyramid Encoder (SIM-ready KV cache)
# ---------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, s, p), nn.GroupNorm(8, c_out), nn.GELU(),
            nn.Conv2d(c_out, c_out, k, 1, p), nn.GroupNorm(8, c_out), nn.GELU(),
        )
    def forward(self, x):
        return self.net(x)


class PyramidEncoder(nn.Module):
    """3-level pyramid: 1/4, 1/8, 1/16 with light conv blocks.
    For SIM-style reuse, we expose a simple cache API.
    """
    def __init__(self, c: int = 64):
        super().__init__()
        self.lvl1 = nn.Sequential(
            nn.Conv2d(3, c, 3, 2, 1), nn.GELU(),  # 1/2
            nn.Conv2d(c, c, 3, 2, 1), nn.GELU(),   # 1/4
            ConvBlock(c, c),
        )
        self.lvl2 = nn.Sequential(
            nn.Conv2d(c, c*2, 3, 2, 1), nn.GELU(),  # 1/8
            ConvBlock(c*2, c*2)
        )
        self.lvl3 = nn.Sequential(
            nn.Conv2d(c*2, c*3, 3, 2, 1), nn.GELU(),  # 1/16
            ConvBlock(c*3, c*3)
        )
        self.cache: Dict[str, torch.Tensor] = {}

    def forward(self, frames: torch.Tensor, use_cache: bool = True) -> List[List[torch.Tensor]]:
        """frames: BxTx3xHxW -> features per level per frame
        returns: list over levels [ [B C1 H1 W1]_t, [B C2 H2 W2]_t, [B C3 H3 W3]_t ]
        """
        B, T, C, H, W = frames.shape
        feats_lvl1, feats_lvl2, feats_lvl3 = [], [], []
        # Simple SIM-style reuse on the coarsest level via cache key of previous frame
        for t in range(T):
            x = frames[:, t]
            f1 = self.lvl1(x)
            f2 = self.lvl2(f1)
            key3 = f"lvl3_{t-1}"
            if use_cache and key3 in self.cache:
                # Reuse previous K/V as a warm-start (this is a placeholder for real KV caching)
                f3 = self.lvl3[0](f2)  # only first conv
                f3 = self.lvl3[1](f3 + 0.0 * self.cache[key3])
            else:
                f3 = self.lvl3(f2)
            self.cache[f"lvl3_{t}"] = f3.detach()
            feats_lvl1.append(f1)
            feats_lvl2.append(f2)
            feats_lvl3.append(f3)
        return [feats_lvl1, feats_lvl2, feats_lvl3]


# ---------------------------------------------------------------
# Cost Tokenizer -> Latent Cost Memory (LCM)
# ---------------------------------------------------------------
class CostTokenizer(nn.Module):
    """Build correlation/cost volume between consecutive frames and tokenize it."""
    def __init__(self, dims: List[int] = [64, 128, 192], token_dim: int = 192, radius: int = 4):
        super().__init__()
        self.token_dim = token_dim
        self.radius = radius
        self.proj = nn.ModuleList([nn.Conv2d(r*(2*r+1)**2, token_dim, 1) for r in [dims[0], dims[1], dims[2]]])
        # Note: above is a placeholder; in practice use a learnable projector after constructing cost patches.

    @staticmethod
    def local_correlation(f1: torch.Tensor, f2: torch.Tensor, radius: int) -> torch.Tensor:
        """Compute local correlation volume around each pixel (displacement in [-r,r]).
        f1,f2: BxCxHxW -> B x (2r+1)^2 x H x W
        """
        B, C, H, W = f1.shape
        corr_list = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                f2_shift = torch.roll(f2, shifts=(dy, dx), dims=(2, 3))
                corr = (f1 * f2_shift).sum(dim=1, keepdim=True) / math.sqrt(C)
                corr_list.append(corr)
        corr = torch.cat(corr_list, dim=1)
        return corr

    def forward_level(self, fL: List[torch.Tensor], radius: int, proj: nn.Module) -> List[torch.Tensor]:
        tokens_per_pair = []
        for t in range(len(fL) - 1):
            f1, f2 = fL[t], fL[t+1]
            corr = self.local_correlation(f1, f2, radius)
            tok = proj(corr)  # B x token_dim x H x W
            tokens_per_pair.append(tok)
        return tokens_per_pair

    def forward(self, feats: List[List[torch.Tensor]]) -> List[List[torch.Tensor]]:
        """feats: list over levels -> list over levels of tokens per consecutive pair.
        return: tokens[level][pair_index] -> B x D x H x W
        """
        lvl1, lvl2, lvl3 = feats
        t1 = self.forward_level(lvl1, radius=3, proj=self.proj[0])
        t2 = self.forward_level(lvl2, radius=3, proj=self.proj[1])
        t3 = self.forward_level(lvl3, radius=3, proj=self.proj[2])
        return [t1, t2, t3]


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim)
        )
    def forward(self, x: torch.Tensor, kv: Optional[torch.Tensor] = None, attn_bias: Optional[torch.Tensor] = None):
        B, N, D = x.shape
        x2 = self.norm1(x)
        if kv is None:
            y, _ = self.attn(x2, x2, x2, need_weights=False)
        else:
            y, _ = self.attn(x2, kv, kv, need_weights=False)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class LatentCostMemory(nn.Module):
    """Spatio-temporal latent memory over cost tokens with causal update.
    Keeps a simple temporal state per sequence for streaming.
    """
    def __init__(self, token_dim: int = 192, depth: int = 6, heads: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(token_dim, heads) for _ in range(depth)])
        self.temporal_state: Optional[torch.Tensor] = None  # (B, N, D)

    def forward(self, tokens_per_level: List[List[torch.Tensor]], seg_tokens: Optional[torch.Tensor] = None,
                attn_bias: Optional[torch.Tensor] = None) -> List[List[torch.Tensor]]:
        """Tokens per level per pair -> latent embeddings same shape.
        We process each pair independently but can optionally fuse with temporal_state (EMA-like).
        """
        latent_levels: List[List[torch.Tensor]] = []
        for lvl_tokens in tokens_per_level:
            lvl_latent = []
            for tok in lvl_tokens:  # B x D x H x W
                B, D, H, W = tok.shape
                x = rearrange(tok, 'b d h w -> b (h w) d')
                for blk in self.blocks:
                    x = blk(x)
                # simple causal temporal fusion (EMA)
                if self.temporal_state is None or self.temporal_state.shape[1] != x.shape[1]:
                    self.temporal_state = x.detach()
                x = 0.8 * x + 0.2 * self.temporal_state
                self.temporal_state = x.detach()
                lvl_latent.append(rearrange(x, 'b n d -> b d n'))
            latent_levels.append(lvl_latent)
        return latent_levels


# ---------------------------------------------------------------
# Global Temporal Regressor (GTR)
# ---------------------------------------------------------------
class GlobalTemporalRegressor(nn.Module):
    """Aggregate multi-pair latent per level temporally and predict coarse flow fields."""
    def __init__(self, token_dim: int = 192, heads: int = 4, depth: int = 2):
        super().__init__()
        self.temporal_blocks = nn.ModuleList([TransformerBlock(token_dim, heads) for _ in range(depth)])
        self.head = nn.Sequential(
            nn.Linear(token_dim, token_dim), nn.GELU(), nn.Linear(token_dim, 2)  # predict 2D flow
        )

    def forward(self, latent_levels: List[List[torch.Tensor]], feats_levels: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        """For each consecutive pair, fuse across levels & time and regress a coarse flow at 1/8 scale.
        Returns flows: list of Bx2xH8xW8 per pair (use decoder to refine to higher res).
        """
        # choose level 2 (1/8) as the regression base
        lvl = 1
        per_pair = latent_levels[lvl]  # list length = T-1, each is B x D x (H*W)
        flows = []
        for k, lat in enumerate(per_pair):
            x = rearrange(lat, 'b d n -> b n d')
            for blk in self.temporal_blocks:
                x = blk(x)
            flow_vec = self.head(x)  # B N 2
            # reshape to HxW
            H8, W8 = feats_levels[lvl][k].shape[-2:]
            flow = rearrange(flow_vec, 'b (h w) c -> b c h w', h=H8, w=W8)
            flows.append(flow)
        return flows


# ---------------------------------------------------------------
# Multi-Scale Recurrent Decoder (coarse->fine)
# ---------------------------------------------------------------
class MSRecurrentDecoder(nn.Module):
    def __init__(self, chs: List[int] = [64, 128, 192], iters_per_level: int = 4):
        super().__init__()
        self.iters = iters_per_level
        self.refine1 = nn.Sequential(nn.Conv2d(chs[1] + 2 + chs[1], chs[1], 3, 1, 1), nn.GELU(), nn.Conv2d(chs[1], 2, 3, 1, 1))
        self.refine0 = nn.Sequential(nn.Conv2d(chs[0] + 2 + chs[0], chs[0], 3, 1, 1), nn.GELU(), nn.Conv2d(chs[0], 2, 3, 1, 1))

    def forward(self, coarse_flows: List[torch.Tensor], feats_levels: List[List[torch.Tensor]], latent_levels: List[List[torch.Tensor]], attn_bias=None) -> List[torch.Tensor]:
        """Refine flows from 1/8 -> 1/4 -> 1/2 (approximately). We produce outputs at 1/4 and 1/2 (~1/4 is typical for optical flow).
        Returns per-pair refined flows at level 1/4 (lvl1 features).
        """
        lvl1, lvl2, lvl3 = feats_levels
        out_flows = []
        for k, f8 in enumerate(coarse_flows):
            # Upsample coarse (1/8) to 1/4
            f4 = F.interpolate(f8, scale_factor=2.0, mode='bilinear', align_corners=True)
            feat4_1 = lvl2[k]  # 1/8 features for warping target
            feat4_0 = lvl2[k+1]
            # iterative refinement at 1/8 (named lvl2)
            cur = f8
            for _ in range(self.iters):
                # Prepare context: concat [feat_k, flow, warped feat_{k+1}]
                w_10 = warp(feat4_0, cur)
                x = torch.cat([feat4_1, cur, w_10], dim=1)
                upd = self.refine1(x)
                cur = cur + upd
            # upsample to 1/4 and another refinement using lvl1 feats
            f4 = F.interpolate(cur, scale_factor=2.0, mode='bilinear', align_corners=True)
            feat2_1 = lvl1[k]
            feat2_0 = lvl1[k+1]
            cur2 = f4
            for _ in range(self.iters):
                w_10 = warp(feat2_0, cur2)
                x = torch.cat([feat2_1, cur2, w_10], dim=1)
                upd = self.refine0(x)
                cur2 = cur2 + upd
            out_flows.append(cur2)
        return out_flows


# ---------------------------------------------------------------
# Optional: Occlusion Head
# ---------------------------------------------------------------
class OcclusionHead(nn.Module):
    def __init__(self, in_ch: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, in_ch, 3, 1, 1), nn.GELU(), nn.Conv2d(in_ch, 1, 1))
    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


# ---------------------------------------------------------------
# Optional: SAM Guidance Adapter (lightweight placeholder)
# ---------------------------------------------------------------
class SamGuidanceAdapter(nn.Module):
    """Convert per-frame binary masks into segment tokens and bias.
    masks: BxTxSxHxW (S segments per frame) or (B,T,1,H,W) for a single mask per frame.
    We compute mean features within segments at 1/8 and return tokens; also return boundary edges for losses.
    """
    def __init__(self, feat_dim: int = 128, token_dim: int = 192):
        super().__init__()
        self.proj = nn.Linear(feat_dim, token_dim)

    def forward(self, masks: torch.Tensor, lvl8_feats: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Build segment tokens by averaging lvl8 features within masks
        # lvl8_feats is a list over time of BxC8xH8xW8
        B = lvl8_feats[0].shape[0]
        T = len(lvl8_feats)
        if masks.ndim == 5:  # B T S H W
            S = masks.shape[2]
        else:
            raise ValueError("masks must be BxTxSxHxW")
        tokens = []
        edges = []
        for t in range(T):
            f = lvl8_feats[t]
            C8, H8, W8 = f.shape[1:]
            m = F.interpolate(masks[:, t].float(), size=(H8, W8), mode='nearest')  # B S H8 W8
            m = rearrange(m, 'b s h w -> b s (h w)')
            f_flat = rearrange(f, 'b c h w -> b (h w) c')
            # mean within each mask
            num = m.sum(-1, keepdim=True).clamp_min(1.0)
            seg_feat = (m @ f_flat) / num  # B S C
            tok = self.proj(seg_feat)      # B S D
            tokens.append(tok)
            # edges: simple boundary via maxpool difference
            m_img = m.sum(1).view(B, 1, H8, W8)
            edge = torch.abs(m_img - F.max_pool2d(m_img, 3, 1, 1)) > 0
            edges.append(edge.float())
        seg_tokens = torch.stack(tokens, dim=1)  # B T S D
        edge_map = torch.stack(edges, dim=1)     # B T 1 H8 W8
        return seg_tokens, edge_map


# ---------------------------------------------------------------
# Main Model Wrapper
# ---------------------------------------------------------------
@dataclass
class ModelConfig:
    enc_channels: int = 64
    token_dim: int = 192
    lcm_depth: int = 6
    lcm_heads: int = 4
    gtr_depth: int = 2
    gtr_heads: int = 4
    iters_per_level: int = 4
    use_sam: bool = False


class AniFlowFormerT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        c = cfg.enc_channels
        self.encoder = PyramidEncoder(c)
        self.tokenizer = CostTokenizer([c, c*2, c*3], token_dim=cfg.token_dim)
        self.lcm = LatentCostMemory(token_dim=cfg.token_dim, depth=cfg.lcm_depth, heads=cfg.lcm_heads)
        self.gtr = GlobalTemporalRegressor(token_dim=cfg.token_dim, heads=cfg.gtr_heads, depth=cfg.gtr_depth)
        self.decoder = MSRecurrentDecoder([c, c*2, c*3], iters_per_level=cfg.iters_per_level)
        self.occ_head = OcclusionHead(in_ch=c*2)
        self.sam_adapter = SamGuidanceAdapter(feat_dim=c*2, token_dim=cfg.token_dim)
        self.ssim = SSIM()

    @torch.cuda.amp.autocast(enabled=True)
    def forward(self, clip: torch.Tensor, sam_masks: Optional[torch.Tensor] = None, use_sam: Optional[bool] = None) -> Dict[str, List[torch.Tensor]]:
        """clip: BxTx3xHxW (T>=3)
        Returns dict with flows (list of Bx2xH4xW4 for consecutive pairs) and optional occ logits.
        """
        if use_sam is None:
            use_sam = self.cfg.use_sam and (sam_masks is not None)

        feats_levels = self.encoder(clip, use_cache=True)  # list over levels, each list over time
        tokens = self.tokenizer(feats_levels)

        seg_tokens = None
        if use_sam:
            # use level 1/8 features for tokens/edges
            lvl8_feats = feats_levels[1]
            seg_tokens, edge_map = self.sam_adapter(sam_masks, lvl8_feats)
        latent = self.lcm(tokens, seg_tokens=seg_tokens, attn_bias=None)
        coarse_flows = self.gtr(latent, feats_levels)
        flows = self.decoder(coarse_flows, feats_levels, latent)

        # produce simple occlusion logits at 1/8 for each pair (using lvl2 features of first frame)
        occ_logits = []
        lvl2 = feats_levels[1]
        for k in range(len(flows)):
            occ_logits.append(self.occ_head(lvl2[k]))

        return {"flows": flows, "occ": occ_logits}

    # -------------------- Losses --------------------
    def photometric_loss(self, frames: torch.Tensor, flows: List[torch.Tensor], alpha_ssim: float = 0.2) -> torch.Tensor:
        B, T, C, H, W = frames.shape
        loss = 0.0
        count = 0
        # flows[k] corresponds to (k -> k+1) at ~1/4 resolution
        for k, f in enumerate(flows):
            # upsample flow to full res for photometric (optional; here we upsample to match frame size)
            f_full = F.interpolate(f, size=(H, W), mode='bilinear', align_corners=True) * (W / f.shape[-1])
            I0 = frames[:, k]
            I1 = frames[:, k+1]
            I1_w = warp(I1, f_full)
            l1 = (I0 - I1_w).abs()
            ssim = self.ssim(I0, I1_w)
            photo = (1 - alpha_ssim) * l1.mean(dim=1, keepdim=True) + alpha_ssim * ssim
            loss = loss + photo.mean()
            count += 1
        return loss / max(count, 1)

    def edge_aware_smoothness(self, frames: torch.Tensor, flows: List[torch.Tensor], w: float = 0.1) -> torch.Tensor:
        B, T, C, H, W = frames.shape
        loss = 0.0
        count = 0
        # compute image gradients as edge weights at full res
        gx, gy = image_gradients(frames.reshape(B*T, C, H, W))
        mag = (gx.abs().mean(1, keepdim=True) + gy.abs().mean(1, keepdim=True))
        mag = 1.0 / (mag + 1e-3)  # higher weight on flat regions, lower near edges
        mag = mag.reshape(B, T, 1, H, W)
        for k, f in enumerate(flows):
            f_full = F.interpolate(f, size=(H, W), mode='bilinear', align_corners=True) * (W / f.shape[-1])
            dx, dy = image_gradients(f_full)
            sm = (dx.abs() + dy.abs()) * mag[:, k]
            loss = loss + sm.mean()
            count += 1
        return w * loss / max(count, 1)

    def temporal_composition(self, flows: List[torch.Tensor], w: float = 0.05) -> torch.Tensor:
        # Penalize: F_{t->t+2} ~ F_{t->t+1} ⊕ F_{t+1->t+2}. We only have consecutive flows, so compose pairs.
        loss = 0.0
        count = 0
        for k in range(len(flows) - 1):
            f01 = flows[k]
            f12 = flows[k+1]
            # Upsample to the same size (already same). Compose via warping second flow by first.
            f12_w = warp(f12, f01)
            f02 = f01 + f12_w
            # Zero target (we don't have direct f02); penalize magnitude drift to encourage consistency.
            loss = loss + f02.abs().mean()
            count += 1
        return w * loss / max(count, 1)

    def cycle_consistency(self, flows: List[torch.Tensor], w: float = 0.05) -> torch.Tensor:
        # Cycle t->t+1->t should approximately recover identity (0 flow) for small windows.
        loss = 0.0
        count = 0
        for k in range(len(flows)):
            # build backward estimate by negating forward and warping back (rough heuristic)
            f_fw = flows[k]
            f_bw = -warp(f_fw, f_fw)  # crude backward approx
            cyc = f_fw + warp(f_bw, f_fw)
            loss = loss + cyc.abs().mean()
            count += 1
        return w * loss / max(count, 1)

    def unsup_loss(self, clip: torch.Tensor, out: Dict[str, List[torch.Tensor]]) -> Dict[str, torch.Tensor]:
        flows = out["flows"]
        losses = {}
        losses["photo"] = self.photometric_loss(clip, flows, alpha_ssim=0.2)
        losses["smooth"] = self.edge_aware_smoothness(clip, flows, w=0.1)
        losses["temporal"] = self.temporal_composition(flows, w=0.05)
        losses["cycle"] = self.cycle_consistency(flows, w=0.05)
        losses["total"] = sum(losses.values())
        return losses


# ---------------------------------------------------------------
# Minimal Dataset stub (replace with your loader)
# ---------------------------------------------------------------
class DummyAnimeClipDataset(torch.utils.data.Dataset):
    """Stub dataset that returns random clips. Replace with real reader.
    Each item: clip tensor (B=1) shaped (T,3,H,W).
    """
    def __init__(self, length=2000, T=5, H=256, W=448):
        super().__init__()
        self.length = length
        self.T = T
        self.H = H
        self.W = W
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        clip = torch.rand(self.T, 3, self.H, self.W)
        return clip


# ---------------------------------------------------------------
# Training loop (unsupervised)
# ---------------------------------------------------------------
@dataclass
class TrainConfig:
    iters: int = 2000
    batch_size: int = 2
    lr: float = 1e-4
    amp: bool = True
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_unsupervised():
    cfg = ModelConfig(use_sam=False)
    model = AniFlowFormerT(cfg).to(TrainConfig.device)
    ds = DummyAnimeClipDataset(length=200, T=5, H=256, W=448)
    dl = torch.utils.data.DataLoader(ds, batch_size=TrainConfig.batch_size, shuffle=True, num_workers=2, drop_last=True)

    optim = torch.optim.AdamW(model.parameters(), lr=TrainConfig.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=TrainConfig.amp)

    model.train()
    it = 0
    while it < TrainConfig.iters:
        for batch in dl:
            it += 1
            if it > TrainConfig.iters: break
            clip = batch.to(TrainConfig.device)  # BxTx3xHxW
            with torch.cuda.amp.autocast(enabled=TrainConfig.amp):
                out = model(clip)
                losses = model.unsup_loss(clip, out)
            scaler.scale(losses["total"]).backward()
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)

            if it % 20 == 0:
                print(f"iter {it}: total={losses['total']:.4f} photo={losses['photo']:.4f} smooth={losses['smooth']:.4f} temporal={losses['temporal']:.4f} cycle={losses['cycle']:.4f}")

    print("Training done.")


if __name__ == "__main__":
    # Quick smoke test (synthetic). Replace with real dataset before actual training.
    train_unsupervised()
