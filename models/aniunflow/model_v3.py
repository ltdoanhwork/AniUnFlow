"""
AniFlowFormer-T V3
===================
Version 3 with full SAM integration.

Backward compatible - can load V1/V2 weights for shared components.
Uses new V3 modules for SAM guidance.

Key improvements:
1. SAMGuidanceAdapterV3: 4 modes of SAM integration
2. SAMGuidedGlobalMatcher: Boundary-aware matching
3. LatentCostMemoryV3: Segment cross-attention
4. All existing components (encoder, GTR, decoder) reused
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch
import torch.nn as nn

# Reuse existing components
from .encoder import PyramidEncoder
from .gtr import GlobalTemporalRegressor
from .decoder import MSRecurrentDecoder
from .occlusion import OcclusionHead
from .utils import SSIM

# V3 enhanced components
from .sam_adapter_v3 import SAMGuidanceAdapterV3, build_sam_adapter_v3
from .global_matcher_v3 import SAMGuidedGlobalMatchingTokenizer, build_sam_guided_matcher
from .lcm_v3 import LatentCostMemoryV3, build_lcm_v3


@dataclass
class ModelConfigV3:
    """Configuration for AniFlowFormer-T V3."""
    # Base architecture (same as V1/V2)
    enc_channels: int = 64
    token_dim: int = 192
    lcm_depth: int = 6
    lcm_heads: int = 4
    gtr_depth: int = 2
    gtr_heads: int = 4
    iters_per_level: int = 4
    
    # SAM integration (V3 specific)
    use_sam: bool = True
    sam_version: int = 3  # Use V3 components
    
    # V3 SAM guidance modes
    use_feature_concat: bool = True
    use_attention_bias: bool = True
    use_cost_modulation: bool = True
    use_object_pooling: bool = True
    
    # Segment-aware extensions (V2 compatible)
    use_segment_cost_modulation: bool = True
    use_segment_attention_mask: bool = True
    use_segment_refinement: bool = False
    
    # Number of SAM segments
    num_segments: int = 16


def debug_mag(x: torch.Tensor, name: str):
    """Debug helper to print flow magnitude."""
    with torch.no_grad():
        mag = (x ** 2).sum(1).sqrt()
        print(f"[DEBUG] {name}: mean |flow| = {mag.mean().item():.4f}")


class AniFlowFormerTV3(nn.Module):
    """
    AniFlowFormer-T Version 3.
    
    Full SAM integration with:
    - SAMGuidanceAdapterV3 for multi-mode SAM guidance
    - SAMGuidedGlobalMatcher for boundary-aware matching
    - LatentCostMemoryV3 for segment cross-attention
    
    Backward compatible: encoder, GTR, decoder same as V1.
    """
    
    def __init__(self, cfg: ModelConfigV3):
        super().__init__()
        self.cfg = cfg
        c = cfg.enc_channels
        
        # === Shared components (V1 compatible) ===
        self.encoder = PyramidEncoder(c)
        self.gtr = GlobalTemporalRegressor(
            token_dim=cfg.token_dim,
            heads=cfg.gtr_heads,
            depth=cfg.gtr_depth,
        )
        self.decoder = MSRecurrentDecoder(
            chs=[c, c * 2, c * 3],
            iters_per_level=cfg.iters_per_level,
        )
        self.occ_head = OcclusionHead(in_ch=c * 2)
        self.ssim = SSIM()
        
        # === V3 Enhanced components ===
        # SAM Guidance Adapter V3
        self.sam_adapter = SAMGuidanceAdapterV3(
            feat_dim=c * 2,  # Level 1/8 features
            token_dim=cfg.token_dim,
            num_segments=cfg.num_segments,
            num_heads=cfg.lcm_heads,
            use_feature_concat=cfg.use_feature_concat,
            use_attention_bias=cfg.use_attention_bias,
            use_cost_modulation=cfg.use_cost_modulation,
            use_object_pooling=cfg.use_object_pooling,
        )
        
        # SAM-Guided Global Matcher
        self.tokenizer = SAMGuidedGlobalMatchingTokenizer(
            dims=[c, c * 2, c * 3],
            token_dim=cfg.token_dim,
            num_heads=4,
            topk=128,
            use_boundary_modulation=cfg.use_cost_modulation,
            use_segment_affinity=cfg.use_attention_bias,
        )
        
        # Latent Cost Memory V3 with segment cross-attention
        self.lcm = LatentCostMemoryV3(
            token_dim=cfg.token_dim,
            depth=cfg.lcm_depth,
            heads=cfg.lcm_heads,
            use_segment_cross_attn=cfg.use_attention_bias,
        )
    
    def load_v1_weights(self, state_dict: Dict[str, torch.Tensor], strict: bool = False):
        """
        Load weights from V1/V2 model.
        Only loads compatible components (encoder, GTR, decoder, occ_head).
        """
        # Filter for compatible keys
        compatible_keys = {}
        for k, v in state_dict.items():
            # Keep encoder, GTR, decoder, occ_head
            if any(k.startswith(prefix) for prefix in ['encoder.', 'gtr.', 'decoder.', 'occ_head.']):
                compatible_keys[k] = v
        
        missing, unexpected = self.load_state_dict(compatible_keys, strict=False)
        print(f"[V3] Loaded {len(compatible_keys)} weights from V1/V2")
        print(f"[V3] Missing in V3 (new components): {len(missing)}")
        return missing, unexpected
    
    @torch.cuda.amp.autocast(enabled=True)
    def forward(
        self,
        clip: torch.Tensor,
        sam_masks: Optional[torch.Tensor] = None,
        use_sam: Optional[bool] = None,
        debug: bool = False,
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Forward pass with SAM guidance.
        
        Args:
            clip: (B, T, 3, H, W) video clip
            sam_masks: (B, T, S, H, W) SAM segment masks
            use_sam: Override config SAM setting
            debug: Print debug info
        
        Returns:
            dict with:
            - flows: List of (B, 2, H/4, W/4) flows for consecutive pairs
            - occ: List of occlusion logits
            - (optional) sam_outputs: SAM adapter outputs for visualization
        """
        if use_sam is None:
            use_sam = self.cfg.use_sam and (sam_masks is not None)
        
        # === 1. Encode frames ===
        feats_levels = self.encoder(clip, use_cache=True)
        # feats_levels: [[f0_l0, f1_l0, ...], [f0_l1, ...], [f0_l2, ...]]
        
        # === 2. SAM Guidance (V3) ===
        sam_outputs = None
        boundary_maps = None
        attn_biases = None
        seg_tokens = None
        
        if use_sam and sam_masks is not None:
            # Get level 1/8 features for SAM adapter
            lvl8_feats = feats_levels[1]  # List of T tensors
            
            # Process through SAM adapter V3
            sam_outputs = self.sam_adapter(sam_masks, lvl8_feats)
            
            # Extract guidance signals
            boundary_maps = [sam_outputs['boundary_maps'][:, t] for t in range(sam_masks.shape[1] - 1)]
            
            if 'attn_bias' in sam_outputs:
                attn_biases = sam_outputs['attn_bias'][:-1]  # T-1 biases for pairs
            
            if 'seg_tokens' in sam_outputs:
                seg_tokens = sam_outputs['seg_tokens']  # (B, T, S, D)
            
            # Use enhanced features if available
            if 'enhanced_features' in sam_outputs:
                # Replace level 1/8 features with enhanced versions
                feats_levels[1] = sam_outputs['enhanced_features']
        
        # === 3. Global Matching with SAM guidance ===
        tokens = self.tokenizer(
            feats_levels,
            boundary_maps=boundary_maps,
            attn_biases=attn_biases,
        )
        
        # === 4. Latent Cost Memory with segment cross-attention ===
        latent = self.lcm(
            tokens,
            seg_tokens=seg_tokens,
            attn_bias=attn_biases,
        )
        
        # === 5. Global Temporal Regressor ===
        coarse_flows = self.gtr(latent, feats_levels)
        
        if debug:
            debug_mag(coarse_flows[0], "coarse_flows[0]")
        
        # === 6. Multi-scale Recurrent Decoder ===
        flows = self.decoder(coarse_flows, feats_levels, latent)
        
        if debug:
            debug_mag(flows[0], "flows[0]")
        
        # === 7. Occlusion Head ===
        occ_logits = []
        lvl2 = feats_levels[1]
        for k in range(len(flows)):
            occ_logits.append(self.occ_head(lvl2[k]))
        
        # === Output ===
        output = {
            "flows": flows,
            "occ": occ_logits,
        }
        
        if sam_outputs is not None:
            output["sam_outputs"] = sam_outputs
        
        return output
    
    def reset_memory(self):
        """Reset LCM memory state."""
        self.lcm.reset_memory()


# ============= Builder functions =============
def build_model_v3(cfg: Dict) -> AniFlowFormerTV3:
    """Build AniFlowFormer-T V3 from config dict."""
    model_cfg = cfg.get('model', {}).get('args', {})
    sam_cfg = cfg.get('sam_guidance', {})
    
    config = ModelConfigV3(
        enc_channels=model_cfg.get('enc_channels', 64),
        token_dim=model_cfg.get('token_dim', 192),
        lcm_depth=model_cfg.get('lcm_depth', 6),
        lcm_heads=model_cfg.get('lcm_heads', 4),
        gtr_depth=model_cfg.get('gtr_depth', 2),
        gtr_heads=model_cfg.get('gtr_heads', 4),
        iters_per_level=model_cfg.get('iters_per_level', 4),
        use_sam=cfg.get('sam', {}).get('enabled', True),
        sam_version=3,
        use_feature_concat=sam_cfg.get('feature_concat', True),
        use_attention_bias=sam_cfg.get('attention_bias', True),
        use_cost_modulation=sam_cfg.get('cost_modulation', True),
        use_object_pooling=sam_cfg.get('object_pooling', True),
        num_segments=cfg.get('sam', {}).get('num_segments', 16),
    )
    
    return AniFlowFormerTV3(config)
