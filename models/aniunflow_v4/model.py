"""
AniFlowFormerT V4: Full SAM Integration Model
==============================================
Clean modular model with ablation-friendly SAM components.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import asdict

# V4 modules
from .config import V4Config, SAMConfig, LossConfig, ModelConfig
from .sam_encoder import SAMEncoderWrapper
from .fusion import ContextFusionModule, BoundaryAwareConcat
from .losses import compute_boundary_map, SAMGuidedLossBundle

# Reuse V1 components
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.aniunflow.encoder import PyramidEncoder
from models.aniunflow.decoder import MSRecurrentDecoder
from models.aniunflow.tokenizer import CostTokenizer
from models.aniunflow.lcm import LatentCostMemory
from models.aniunflow.global_matcher import GlobalMatchingTokenizer


class AniFlowFormerTV4(nn.Module):
    """
    V4 Model: Full SAM Integration with Ablation Support.
    
    All SAM components are toggleable via config:
    - sam.use_encoder_features: Use frozen SAM ViT features
    - sam.use_mask_guidance: Use segmentation masks
    - sam.feature_concat: Concatenate boundary features
    - sam.attention_bias: Same-segment attention boost
    """
    
    def __init__(self, config: V4Config = None, **kwargs):
        super().__init__()
        
        # Handle config from dict (YAML) or object
        if config is None:
            config = V4Config()
        elif isinstance(config, dict):
            config = V4Config.from_dict(config)
        
        self.config = config
        self.model_cfg = config.model
        self.sam_cfg = config.sam
        self.loss_cfg = config.loss
        
        # ========== Core Flow Components (from V1) ==========
        self.encoder = PyramidEncoder(
            in_channels=3,
            base_channels=self.model_cfg.enc_channels,
            out_channels=self.model_cfg.enc_out_channels,
        )
        
        self.tokenizer = CostVolumeTokenizer(
            feat_channels=self.model_cfg.enc_out_channels,
            token_dim=self.model_cfg.token_dim,
        )
        
        self.lcm = LatentCostMemory(
            token_dim=self.model_cfg.token_dim,
            depth=self.model_cfg.lcm_depth,
            heads=self.model_cfg.lcm_heads,
        )
        
        self.gtr = GlobalTokenRefinement(
            token_dim=self.model_cfg.token_dim,
            depth=self.model_cfg.gtr_depth,
            heads=self.model_cfg.gtr_heads,
        )
        
        self.decoder = IterativeFlowDecoder(
            token_dim=self.model_cfg.token_dim,
            feat_channels=self.model_cfg.enc_out_channels,
        )
        
        # ========== SAM Components (toggleable) ==========
        self.sam_encoder = None
        self.sam_fusion = None
        self.boundary_concat = None
        
        if self.sam_cfg.enabled:
            # SAM Encoder (frozen ViT features)
            if self.sam_cfg.use_encoder_features:
                self.sam_encoder = SAMEncoderWrapper(
                    checkpoint=self.sam_cfg.encoder_checkpoint,
                    config=self.sam_cfg.encoder_config,
                    freeze=self.sam_cfg.encoder_freeze,
                    feature_scales=self.sam_cfg.feature_scales,
                )
                
                # Fusion module
                self.sam_fusion = ContextFusionModule(
                    flow_dim=self.model_cfg.enc_out_channels,
                    sam_dim=self.sam_cfg.feature_dim,
                    out_dim=self.model_cfg.enc_out_channels,
                )
            
            # Mask guidance (boundary concat)
            if self.sam_cfg.use_mask_guidance:
                self.boundary_concat = BoundaryAwareConcat(
                    flow_dim=self.model_cfg.enc_out_channels,
                    out_dim=self.model_cfg.enc_out_channels,
                    num_segments=self.sam_cfg.num_segments,
                )
        
        # ========== Loss Bundle ==========
        self.sam_loss_bundle = SAMGuidedLossBundle(
            homography_weight=self.loss_cfg.homography_smooth,
            boundary_sharpness_weight=self.loss_cfg.boundary_sharpness,
            object_variance_weight=self.loss_cfg.object_variance,
            boundary_smooth_weight=self.loss_cfg.boundary_aware_smooth,
        )
    
    def forward(
        self,
        clip: torch.Tensor,              # (B, T, 3, H, W)
        sam_masks: Optional[torch.Tensor] = None,  # (B, T, H, W) or (B, T, 1, H, W)
        sam_features: Optional[Dict] = None,       # Precomputed SAM features
        return_losses: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional SAM integration.
        
        Args:
            clip: Input frames
            sam_masks: Precomputed segmentation masks (optional)
            sam_features: Precomputed SAM encoder features (optional)
            return_losses: If True, compute and return SAM-guided losses
            
        Returns:
            outputs: Dict containing:
                - flows_fw: List of forward flow tensors
                - flows_bw: List of backward flow tensors
                - sam_losses: Dict of SAM-guided losses (if return_losses=True)
        """
        B, T, C, H, W = clip.shape
        device = clip.device
        
        # ========== Feature Extraction ==========
        # Flatten batch and time for encoder
        frames = clip.view(B * T, C, H, W)
        feats_levels = self.encoder(frames)  # List of (B*T, C, H/s, W/s)
        
        # Reshape back to (B, T, C, H, W) at each level
        feats_per_time = []
        for feats in feats_levels:
            _, c, h, w = feats.shape
            feats = feats.view(B, T, c, h, w)
            feats_per_time.append(feats)
        
        # ========== SAM Feature Integration ==========
        boundary_maps = None
        
        if self.sam_cfg.enabled and self.sam_cfg.use_encoder_features:
            if sam_features is not None:
                # Use precomputed features
                sam_feat_dict = sam_features
            elif self.sam_encoder is not None:
                # Extract features online (expensive)
                sam_feat_dict = self.sam_encoder(frames)
            else:
                sam_feat_dict = None
            
            # Fuse SAM features with flow encoder features
            if sam_feat_dict and self.sam_fusion is not None:
                # Get SAM feature at appropriate scale
                for scale, sam_feat in sam_feat_dict.items():
                    # Find matching encoder level
                    for i, flow_feat in enumerate(feats_per_time):
                        if flow_feat.shape[3] == sam_feat.shape[3]:  # Match by width
                            # Fuse per frame
                            fused = []
                            for t in range(T):
                                f = self.sam_fusion(
                                    flow_feat[:, t],  # (B, C, H, W)
                                    sam_feat if sam_feat.dim() == 4 else sam_feat[:, t]
                                )
                                fused.append(f)
                            feats_per_time[i] = torch.stack(fused, dim=1)
                            break
        
        # ========== Mask Guidance Integration ==========
        if self.sam_cfg.enabled and self.sam_cfg.use_mask_guidance and sam_masks is not None:
            # Normalize mask format
            if sam_masks.dim() == 4:  # (B, T, H, W)
                masks = sam_masks
            else:  # (B, T, 1, H, W)
                masks = sam_masks.squeeze(2)
            
            # Compute boundaries
            boundary_maps = []
            for t in range(T):
                boundary = compute_boundary_map(masks[:, t])  # (B, 1, H, W)
                boundary_maps.append(boundary)
            boundary_maps = torch.stack(boundary_maps, dim=1)  # (B, T, 1, H, W)
            
            # Boundary-aware feature enhancement
            if self.boundary_concat is not None:
                for i, flow_feat in enumerate(feats_per_time):
                    enhanced = []
                    for t in range(T):
                        # Normalize segment labels
                        seg_labels = masks[:, t:t+1].float() / self.sam_cfg.num_segments
                        
                        # Resize boundary and labels to feature size
                        feat_h, feat_w = flow_feat.shape[3], flow_feat.shape[4]
                        boundary_t = F.interpolate(
                            boundary_maps[:, t], size=(feat_h, feat_w), mode='nearest'
                        )
                        seg_labels_t = F.interpolate(
                            seg_labels, size=(feat_h, feat_w), mode='nearest'
                        )
                        
                        # Concatenate
                        f = self.boundary_concat(flow_feat[:, t], boundary_t, seg_labels_t)
                        enhanced.append(f)
                    
                    feats_per_time[i] = torch.stack(enhanced, dim=1)
        
        # ========== Cost Volume and Tokenization ==========
        # Use level 1/8 features for matching
        ref_feats = feats_per_time[1]  # (B, T, C, H/8, W/8)
        
        # Center frame as reference
        center_idx = T // 2
        ref_feat = ref_feats[:, center_idx]  # (B, C, H/8, W/8)
        
        # Compute cost volumes for each frame pair
        tokens_list = []
        for t in range(T):
            if t == center_idx:
                continue
            
            tgt_feat = ref_feats[:, t]
            tokens = self.tokenizer(ref_feat, tgt_feat)
            tokens_list.append(tokens)
        
        # ========== LCM + GTR Processing ==========
        outputs_list = []
        for tokens in tokens_list:
            # LCM
            lcm_out = self.lcm(tokens)
            
            # GTR
            gtr_out = self.gtr(lcm_out)
            
            outputs_list.append(gtr_out)
        
        # ========== Flow Decoding ==========
        flows_fw = []
        flows_bw = []
        
        for i, (gtr_out, t) in enumerate(zip(outputs_list, [t for t in range(T) if t != center_idx])):
            # Forward flow (ref -> t)
            flow_fw = self.decoder(
                gtr_out,
                feats_per_time[0][:, center_idx],  # 1/4 scale
                target_size=(H, W),
                iters=self.model_cfg.iters_per_level,
            )
            flows_fw.append(flow_fw)
            
            # Backward flow (t -> ref)
            flow_bw = self.decoder(
                gtr_out,
                feats_per_time[0][:, t],
                target_size=(H, W),
                iters=self.model_cfg.iters_per_level,
            )
            flows_bw.append(flow_bw)
        
        outputs = {
            'flows_fw': flows_fw,
            'flows_bw': flows_bw,
        }
        
        # ========== SAM-Guided Losses ==========
        if return_losses and sam_masks is not None and len(flows_fw) > 0:
            masks_center = masks[:, center_idx] if masks is not None else None
            img_center = clip[:, center_idx]
            
            sam_losses = self.sam_loss_bundle(
                flow=flows_fw[0],  # First forward flow
                masks=masks_center,
                images=img_center,
                boundary=boundary_maps[:, center_idx] if boundary_maps is not None else None,
            )
            outputs['sam_losses'] = sam_losses
        
        return outputs
    
    @classmethod
    def from_config_file(cls, config_path: str) -> "AniFlowFormerTV4":
        """Load model from YAML config file."""
        import yaml
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(config=config_dict)
