from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
import torch.nn as nn

from .encoder import PyramidEncoder
from .tokenizer import CostTokenizer
from .lcm import LatentCostMemory
from .gtr import GlobalTemporalRegressor
from .decoder import MSRecurrentDecoder
from .occlusion import OcclusionHead
from .sam_adapter import SamGuidanceAdapter
from .utils import SSIM


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

        feats_levels = self.encoder(clip, use_cache=True) # list over levels, each list over time
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