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
from models.aniunflow.gtr import GlobalTemporalRegressor
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
            c=self.model_cfg.enc_channels
        )
        
        c = self.model_cfg.enc_channels
        self.tokenizer = GlobalMatchingTokenizer(
            dims=[c, c*2, c*3],
            token_dim=self.model_cfg.token_dim,
            num_heads=4, # Hardcoded or from config? V1 uses 4.
        )
        
        self.lcm = LatentCostMemory(
            token_dim=self.model_cfg.token_dim,
            depth=self.model_cfg.lcm_depth,
            heads=self.model_cfg.lcm_heads,
        )
        
        self.gtr = GlobalTemporalRegressor(
            token_dim=self.model_cfg.token_dim,
            heads=self.model_cfg.gtr_heads,
            depth=self.model_cfg.gtr_depth,
        )
        
        self.decoder = MSRecurrentDecoder(
            chs=[c, c*2, c*3],
            iters_per_level=self.model_cfg.iters_per_level,
        )
        
        # ========== SAM Components (toggleable) ==========
        self.sam_encoder = None
        self.sam_fusion_layers = None
        self.boundary_concat_layers = None
        
        feature_dims = [c, c*2, c*3]
        
        if self.sam_cfg.enabled:
            # SAM Encoder (frozen ViT features)
            if self.sam_cfg.use_encoder_features:
                try:
                    import sam2
                except ImportError:
                    print("[AniFlowFormerTV4] Warning: 'sam2' module not found. Disabling SAM encoder features.")
                    self.sam_cfg.use_encoder_features = False

            if self.sam_cfg.use_encoder_features:
                self.sam_encoder = SAMEncoderWrapper(
                    checkpoint=self.sam_cfg.encoder_checkpoint,
                    config=self.sam_cfg.encoder_config,
                    freeze=self.sam_cfg.encoder_freeze,
                    feature_scales=self.sam_cfg.feature_scales,
                )
                
                # Fusion module per level
                self.sam_fusion_layers = nn.ModuleList([
                    ContextFusionModule(
                        flow_dim=dim,
                        sam_dim=self.sam_cfg.feature_dim,
                        out_dim=dim, # Keep dimension same as encoder level
                    ) for dim in feature_dims
                ])
            
            # Mask guidance (boundary concat) per level
            if self.sam_cfg.use_mask_guidance:
                self.boundary_concat_layers = nn.ModuleList([
                    BoundaryAwareConcat(
                        flow_dim=dim,
                        out_dim=dim,
                        num_segments=self.sam_cfg.num_segments,
                    ) for dim in feature_dims
                ])
        
        # ========== Loss Bundle ==========
        self.sam_loss_bundle = SAMGuidedLossBundle(
            homography_weight=self.loss_cfg.homography_smooth,
            boundary_sharpness_weight=self.loss_cfg.boundary_sharpness,
            object_variance_weight=self.loss_cfg.object_variance,
            boundary_smooth_weight=self.loss_cfg.boundary_aware_smooth,
        )
    
    def _get_flows(
        self,
        clip: torch.Tensor,              # (B, T, 3, H, W)
        sam_masks: Optional[torch.Tensor] = None,  # (B, T, H, W) or (B, T, 1, H, W)
        sam_features: Optional[Dict] = None,       # Precomputed SAM features
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        """
        Internal flow computation.
        
        Returns:
            flows: List of flow tensors
            boundary_maps: Stacked boundary maps (or None)
        """
        B, T, C, H, W = clip.shape
        device = clip.device
        
        # ========== Feature Extraction ==========
        # V1 Encoder returns list of lists: [ [f1_t0, f1_t1...], [f2_t0...], ... ]
        feats_levels = self.encoder(clip) 
        
        # ========== SAM Feature Integration ==========
        boundary_maps = None
        
        if self.sam_cfg.enabled and self.sam_cfg.use_encoder_features:
            # Prepare SAM features (same as before)
            if sam_features is not None:
                sam_feat_dict = sam_features
            elif self.sam_encoder is not None:
                # Need to flatten for SAM encoder if it expects 4D
                frames_flat = clip.view(B * T, C, H, W)
                sam_feat_dict = self.sam_encoder(frames_flat)
                # Reshape back to (B, T, ...) logic handled in fusion loop
            else:
                sam_feat_dict = None
            
            # Fuse SAM features with flow encoder features
            if sam_feat_dict and self.sam_fusion_layers is not None:
                # Iterate levels
                new_feats_levels = []
                for i, level_feats in enumerate(feats_levels):
                    if i >= len(self.sam_fusion_layers): break
                    fusion_layer = self.sam_fusion_layers[i]
                    
                    # level_feats is list of T tensors, each (B, C, h, w)
                    # Stack to (B, T, C, h, w) for easier processing
                    feat_stack = torch.stack(level_feats, dim=1) 
                    
                    # Find matching SAM scale
                    fused_stack = feat_stack
                    for scale, sam_feat in sam_feat_dict.items():
                        # We use spatial dims to match
                        if sam_feat.shape[-1] == feat_stack.shape[-1]: # Match width
                            # Reshape SAM feat to (B, T, C, h, w)
                            if sam_feat.dim() == 4:
                                s_feat = sam_feat.view(B, T, -1, sam_feat.shape[2], sam_feat.shape[3])
                            else:
                                s_feat = sam_feat
                                
                            # Fuse per timestep
                            fused_list = []
                            for t in range(T):
                                f = fusion_layer(feat_stack[:, t], s_feat[:, t])
                                fused_list.append(f)
                            fused_stack = torch.stack(fused_list, dim=1)
                            break
                    
                    # Convert back to list of tensors
                    new_feats_levels.append([fused_stack[:, t] for t in range(T)])
                feats_levels = new_feats_levels

        # ========== Mask Guidance Integration ==========
        if self.sam_cfg.enabled and self.sam_cfg.use_mask_guidance and sam_masks is not None:
            # Normalize mask format
            if sam_masks.dim() == 4:
                masks = sam_masks
            else:
                masks = sam_masks.squeeze(2)
            
            # Compute boundaries (for losses mainly)
            boundary_maps = []
            for t in range(T):
                boundary = compute_boundary_map(masks[:, t])
                boundary_maps.append(boundary)
            boundary_maps = torch.stack(boundary_maps, dim=1)
            
            # Boundary Aware Concat
            if self.boundary_concat_layers is not None:
                new_feats_levels = []
                for i, level_feats in enumerate(feats_levels):
                    if i >= len(self.boundary_concat_layers):
                         new_feats_levels.append(level_feats)
                         continue
                    concat_layer = self.boundary_concat_layers[i]
                    
                    enhanced_list = []
                    for t in range(T):
                        seg_labels = masks[:, t:t+1].float() / self.sam_cfg.num_segments
                        feat_h, feat_w = level_feats[t].shape[-2:]
                        
                        boundary_t = F.interpolate(boundary_maps[:, t], size=(feat_h, feat_w), mode='nearest')
                        seg_labels_t = F.interpolate(seg_labels, size=(feat_h, feat_w), mode='nearest')
                        
                        f = concat_layer(level_feats[t], boundary_t, seg_labels_t)
                        enhanced_list.append(f)
                    
                    new_feats_levels.append(enhanced_list)
                feats_levels = new_feats_levels

        # ========== V1 Pipeline ==========
        # Tokenizer
        tokens = self.tokenizer(feats_levels)
        
        # LCM
        latent = self.lcm(tokens)
        
        # GTR
        coarse_flows = self.gtr(latent, feats_levels)
        
        # Decoder
        flows = self.decoder(coarse_flows, feats_levels, latent)
        
        return flows, boundary_maps
        
        # Wait, V1 AniFlowFormerT returns {"flows": flows, "occ": occ}
        # And Trainer expects flows_fw/flows_bw.
        # Trainer.py (V1):
        #   out_fw = self.model(clip)
        #   out_bw = self.model(clip_rev)
        # So model returns ONE direction. Trainer handles bidirectional call.
        
        # So for V4 with "return_losses", we might want to run bidirectional internally?
        # Or stick to V1 pattern: Trainer calls model twice.
        
        # If I want integrated losses in ONE call, I should run bidirectional here?
        # But that duplicates code/compute if Trainer also supports it.
        # My V4 Trainer update handles `out_fw["flows_fw"]` and `out_fw["flows_bw"]`.
        
        # Let's revert to separate calls for simplicity and compatibility with existing V1 flow.
        # But I added `return_losses`. Losses usually need bidirectional flow (consistency).
        # If I want model to compute losses, model needs to run bidirectional.
        
        if return_losses:
            # We need backward flow for consistency loss
            # Run backward pass
            clip_rev = torch.flip(clip, dims=[1])
            # Reuse encoder features? No, features are time dependent if we fuse SAM.
            # But we can flip features if they are symmetric.
            # Let's just run forward pass on reversed clip for simplicity (or optimize later).
            
            # Encoder on reverse
            feats_levels_rev = self.encoder(clip_rev)
            # ... apply SAM fusion to reverse ... (omitted for brevity, ideally share code)
            # For now, let's assume valid mask/sam features for reverse are flipped inputs
            
            # TO KEEP IT SIMPLE:
            # Just return Forward flows here.
            # Trainer will run Backward pass.
            # BUT then how to calculate losses inside Model?
            # Model needs both flows.
            
            # OK, the V4 design "Integrated forward + loss" implies model runs everything.
            pass

        # For now, let's match V1 behavior: Return forward flows.
        # Trainer will handle backward pass.
        # If "return_losses" is True, we can't compute them without BW flow.
        # So "return_losses" support requires bidirectional run inside model.
        
        # Let's clean up outputs
        outputs = {
            "flows": flows,
            "flows_fw": flows # Alias
        }
        
        if return_losses and sam_masks is not None:
             # Just compute Single-direction SAM losses (smoothness, boundary)
             # Consistency loss requires BW flow, so skip it or pass BW flow from Trainer?
             # My updated Trainer expects 'sam_losses' in output.
             if len(flows) > 0:
                 sam_losses = self.sam_loss_bundle(
                    flow=flows[0], 
                    masks=sam_masks[:, 0] if sam_masks.dim()==4 else sam_masks[:, 0, 0],
                    images=clip[:, 0],
                    boundary=boundary_maps[:, 0] if boundary_maps is not None else None
                 )
                 outputs['sam_losses'] = sam_losses

        return outputs
    
    def forward(
        self,
        clip: torch.Tensor,              # (B, T, 3, H, W)
        sam_masks: Optional[torch.Tensor] = None,  # (B, T, H, W) or (B, T, 1, H, W)
        sam_features: Optional[Dict] = None,       # Precomputed SAM features
        return_losses: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with bidirectional flow and SAM-guided losses.
        """
        B, T, C, H, W = clip.shape
        
        # Forward pass
        flows_fw, bn_fw = self._get_flows(clip, sam_masks, sam_features)
        
        outputs = {
            "flows": flows_fw,
            "flows_fw": flows_fw,
            "flows_bw": [], # Default empty
        }
        
        # If return_losses, compute backward flows and losses
        if return_losses:
            # Backward pass inputs
            clip_bw = torch.flip(clip, [1])
            
            masks_bw = None
            if sam_masks is not None:
                masks_bw = torch.flip(sam_masks, [1])
                
            feats_bw = None
            if sam_features is not None:
                feats_bw = {}
                for k, v in sam_features.items():
                    if v.dim() == 5: # (B, T, ...)
                        feats_bw[k] = torch.flip(v, [1])
                    else:
                        feats_bw[k] = v # Static features
            
            # Backward pass
            flows_bw_list, bn_bw = self._get_flows(clip_bw, masks_bw, feats_bw)
            
            # Reorder backward flows
            outputs["flows_bw"] = flows_bw_list
            
            # Compute SAM losses
            if sam_masks is not None and len(flows_fw) > 0:
                sam_loss_accum = {}
                count = 0
                
                # Constrain each forward flow
                for i, flow in enumerate(flows_fw):
                    # flow is from frame i to i+1
                    if i >= T-1: break 
                    
                    # Use mask at frame i
                    mask_i = sam_masks[:, i]  # (B, H, W) or (B, 1, H, W)
                    if mask_i.dim() == 4:  # B, 1, H, W
                        mask_i = mask_i.squeeze(1)
                    
                    # Downsample mask to match flow resolution
                    # flow is (B, 2, H_flow, W_flow), typically H/4, W/4
                    H_flow, W_flow = flow.shape[2:]
                    if mask_i.shape[1:] != (H_flow, W_flow):
                        # Downsample using nearest neighbor to preserve integer labels
                        mask_i = F.interpolate(
                            mask_i.unsqueeze(1).float(),  # (B, 1, H, W)
                            size=(H_flow, W_flow),
                            mode='nearest'
                        ).squeeze(1).long()  # (B, H_flow, W_flow)
                         
                    # Downsample boundary map to match flow resolution
                    bn_i = bn_fw[:, i] if bn_fw is not None else None
                    if bn_i is not None:
                        # Ensure 4D shape (B, 1, H, W) for SAM losses
                        if bn_i.dim() == 3:
                            bn_i = bn_i.unsqueeze(1)  # (B, 1, H, W)
                        
                        if bn_i.shape[2:] != (H_flow, W_flow):
                            bn_i = F.interpolate(
                                bn_i,
                                size=(H_flow, W_flow),
                                mode='bilinear',
                                align_corners=False
                            )  # Keep as (B, 1, H_flow, W_flow)
                    
                    # Downsample images to match flow resolution
                    img_i = clip[:, i]  # (B, 3, H, W)
                    if img_i.shape[2:] != (H_flow, W_flow):
                        img_i = F.interpolate(
                            img_i,
                            size=(H_flow, W_flow),
                            mode='bilinear',
                            align_corners=False
                        )  # (B, 3, H_flow, W_flow)
                    
                    losses = self.sam_loss_bundle(
                        flow=flow,
                        masks=mask_i,
                        images=img_i,
                        boundary=bn_i
                    )
                    
                    for k, v in losses.items():
                        if k not in sam_loss_accum:
                            sam_loss_accum[k] = 0.0
                        sam_loss_accum[k] = sam_loss_accum[k] + v # Accumulate
                        
                    count += 1
                
                if count > 0:
                    # Average
                    final_sam_losses = {k: v / count for k, v in sam_loss_accum.items()}
                    outputs['sam_losses'] = final_sam_losses
        
        return outputs
    
    @classmethod
    def from_config_file(cls, config_path: str) -> "AniFlowFormerTV4":
        """Load model from YAML config file."""
        import yaml
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(config=config_dict)
