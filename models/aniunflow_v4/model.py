"""
AniFlowFormerT V4/V4.5 model
============================
V4: original global-matching + LCM + GTR + decoder pipeline.
V4.5: SAM-guided matcher + LCMv3 + optional RAFT-style iterative refiner.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import V4Config
from .fusion import BoundaryAwareConcat, ContextFusionModule
from .iterative_refiner import IterativeFlowRefiner
from .losses import SAMGuidedLossBundle, compute_boundary_map
from .sam_encoder import SAMEncoderWrapper

from models.aniunflow.decoder import MSRecurrentDecoder
from models.aniunflow.encoder import PyramidEncoder
from models.aniunflow.global_matcher import GlobalMatchingTokenizer
from models.aniunflow.global_matcher_v3 import SAMGuidedGlobalMatchingTokenizer
from models.aniunflow.gtr import GlobalTemporalRegressor
from models.aniunflow.lcm import LatentCostMemory
from models.aniunflow.lcm_v3 import LatentCostMemoryV3


class AniFlowFormerTV4(nn.Module):
    def __init__(self, config: V4Config | dict | None = None, **kwargs):
        super().__init__()

        if config is None:
            config = V4Config()
        elif isinstance(config, dict):
            config = V4Config.from_dict(config)

        self.config = config
        self.model_cfg = config.model
        self.sam_cfg = config.sam
        self.loss_cfg = config.loss
        self.backbone = str(getattr(self.model_cfg, "backbone", "v4")).lower()

        c = self.model_cfg.enc_channels
        self.encoder = PyramidEncoder(c=c)

        # Baseline V4 components
        self.tokenizer = GlobalMatchingTokenizer(
            dims=[c, c * 2, c * 3],
            token_dim=self.model_cfg.token_dim,
            num_heads=self.model_cfg.num_heads,
            topk=int(getattr(self.sam_cfg, "matcher_topk", 96)),
            add_mask_corr=bool(self.sam_cfg.enabled and getattr(self.sam_cfg, "add_mask_corr", False)),
            mask_corr_aggregation=getattr(self.sam_cfg, "mask_corr_aggregation", "concat"),
            mask_corr_weight=float(getattr(self.sam_cfg, "mask_corr_weight", 1.0)),
            num_segments=int(getattr(self.sam_cfg, "num_segments", 32)),
            min_mask_pixels=int(getattr(self.sam_cfg, "mask_corr_min_pixels", 8)),
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
            chs=[c, c * 2, c * 3],
            iters_per_level=self.model_cfg.iters_per_level,
        )

        # V4.5 components (matcher/LCM and optional iterative refiner)
        self.tokenizer_v45_main = None
        self.tokenizer_v45_aux = None
        self.mask_corr_aux_alphas = None
        self.lcm_v45 = None
        self.segment_token_proj = None
        self.iterative_refiner = None

        if self.backbone in {"v4_5_matcher_lcm", "v4_5_hybrid_sam"}:
            matcher_topk = int(getattr(self.sam_cfg, "matcher_topk", 96))
            self.tokenizer_v45_main = SAMGuidedGlobalMatchingTokenizer(
                dims=[c, c * 2, c * 3],
                token_dim=self.model_cfg.token_dim,
                num_heads=self.model_cfg.num_heads,
                topk=matcher_topk,
                use_boundary_modulation=bool(getattr(self.sam_cfg, "cost_modulation", True)),
                use_segment_affinity=bool(getattr(self.sam_cfg, "attention_bias", True)),
            )

            # Auxiliary mask-correlation matcher (UnSAMFlow-style residual fusion).
            self.tokenizer_v45_aux = GlobalMatchingTokenizer(
                dims=[c, c * 2, c * 3],
                token_dim=self.model_cfg.token_dim,
                num_heads=self.model_cfg.num_heads,
                topk=matcher_topk,
                add_mask_corr=True,
                mask_corr_aggregation="residual",
                mask_corr_weight=1.0,
                num_segments=int(getattr(self.sam_cfg, "num_segments", 32)),
                min_mask_pixels=int(getattr(self.sam_cfg, "mask_corr_min_pixels", 8)),
            )
            self.mask_corr_aux_alphas = nn.Parameter(
                torch.full(
                    (3,),
                    float(getattr(self.sam_cfg, "mask_corr_weight_init", 0.5)),
                )
            )

            self.lcm_v45 = LatentCostMemoryV3(
                token_dim=self.model_cfg.token_dim,
                depth=self.model_cfg.lcm_depth,
                heads=self.model_cfg.lcm_heads,
                use_segment_cross_attn=bool(getattr(self.sam_cfg, "attention_bias", True)),
                segment_cross_attn_every=int(getattr(self.sam_cfg, "segment_cross_attn_every", 2)),
            )
            self.segment_token_proj = nn.Conv2d(c * 2, self.model_cfg.token_dim, 1)

            if self.backbone == "v4_5_hybrid_sam":
                self.iterative_refiner = IterativeFlowRefiner(
                    feat_in_dim=c * 2,
                    context_in_dim=c * 2,
                    prior_dim=self.model_cfg.token_dim,
                    hidden_dim=int(getattr(self.model_cfg, "refiner_hidden_dim", 128)),
                    context_dim=int(getattr(self.model_cfg, "refiner_context_dim", 128)),
                    feature_dim=int(getattr(self.model_cfg, "refiner_feature_dim", 128)),
                    motion_dim=int(getattr(self.model_cfg, "refiner_motion_dim", 128)),
                    corr_levels=int(getattr(self.model_cfg, "refiner_corr_levels", 4)),
                    corr_radius=int(getattr(self.model_cfg, "refiner_corr_radius", 4)),
                    iters=int(getattr(self.model_cfg, "refiner_iters", 10)),
                    use_convex_upsampler=bool(getattr(self.model_cfg, "use_convex_upsampler", True)),
                    boundary_gate_strength=float(getattr(self.sam_cfg, "boundary_gate_strength", 0.3)),
                    use_gradient_checkpointing=bool(
                        getattr(self.model_cfg, "refiner_gradient_checkpointing", False)
                    ),
                    delta_clip=float(getattr(self.model_cfg, "refiner_delta_clip", 0.0)),
                    use_prior_flow_init=bool(
                        getattr(self.model_cfg, "refiner_use_prior_flow_init", True)
                    ),
                    prior_flow_init_scale=float(
                        getattr(self.model_cfg, "refiner_prior_flow_init_scale", 1.0)
                    ),
                    prior_flow_init_clip=float(
                        getattr(self.model_cfg, "refiner_prior_flow_init_clip", 0.0)
                    ),
                    delta_damping=float(getattr(self.model_cfg, "refiner_delta_damping", 1.0)),
                    delta_damping_decay=float(
                        getattr(self.model_cfg, "refiner_delta_damping_decay", 1.0)
                    ),
                )

        # SAM components
        self.sam_encoder = None
        self.sam_fusion_layers = None
        self.boundary_concat_layers = None

        feature_dims = [c, c * 2, c * 3]
        if self.sam_cfg.enabled:
            if self.sam_cfg.use_encoder_features:
                self.sam_encoder = SAMEncoderWrapper(
                    checkpoint=self.sam_cfg.encoder_checkpoint,
                    config=self.sam_cfg.encoder_config,
                    freeze=self.sam_cfg.encoder_freeze,
                    feature_scales=self.sam_cfg.feature_scales,
                )
                self.sam_fusion_layers = nn.ModuleList(
                    [
                        ContextFusionModule(
                            flow_dim=dim,
                            sam_dim=self.sam_cfg.feature_dim,
                            out_dim=dim,
                        )
                        for dim in feature_dims
                    ]
                )

            if self.sam_cfg.use_mask_guidance:
                self.boundary_concat_layers = nn.ModuleList(
                    [
                        BoundaryAwareConcat(
                            flow_dim=dim,
                            out_dim=dim,
                            num_segments=self.sam_cfg.num_segments,
                        )
                        for dim in feature_dims
                    ]
                )

        self.sam_loss_bundle = SAMGuidedLossBundle(
            homography_weight=self.loss_cfg.homography_smooth,
            boundary_sharpness_weight=self.loss_cfg.boundary_sharpness,
            object_variance_weight=self.loss_cfg.object_variance,
            boundary_smooth_weight=self.loss_cfg.boundary_aware_smooth,
        )

    @staticmethod
    def _normalize_mask_labels(sam_masks: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if sam_masks is None:
            return None
        if sam_masks.dim() == 5:
            if sam_masks.shape[2] == 1:
                return sam_masks[:, :, 0].long()
            return (sam_masks.argmax(dim=2).long() + 1)
        if sam_masks.dim() == 4:
            return sam_masks.long()
        return None

    def _extract_features(
        self,
        clip: torch.Tensor,
        sam_masks: Optional[torch.Tensor] = None,
        sam_features: Optional[Dict] = None,
    ) -> Tuple[List[List[torch.Tensor]], Optional[torch.Tensor], Optional[torch.Tensor]]:
        b, t, c_img, h_img, w_img = clip.shape
        feats_levels = self.encoder(clip)

        mask_labels = self._normalize_mask_labels(sam_masks)
        boundary_maps = None

        if self.sam_cfg.enabled and self.sam_cfg.use_encoder_features:
            if sam_features is not None:
                sam_feat_dict = sam_features
            elif self.sam_encoder is not None:
                frames_flat = clip.view(b * t, c_img, h_img, w_img)
                sam_feat_dict = self.sam_encoder(frames_flat)
            else:
                sam_feat_dict = None

            if sam_feat_dict and self.sam_fusion_layers is not None:
                new_feats_levels: List[List[torch.Tensor]] = []
                for i, level_feats in enumerate(feats_levels):
                    if i >= len(self.sam_fusion_layers):
                        new_feats_levels.append(level_feats)
                        continue

                    fusion_layer = self.sam_fusion_layers[i]
                    feat_stack = torch.stack(level_feats, dim=1)
                    fused_stack = feat_stack

                    for _, sam_feat in sam_feat_dict.items():
                        if sam_feat.shape[-1] != feat_stack.shape[-1]:
                            continue
                        if sam_feat.dim() == 4:
                            s_feat = sam_feat.view(b, t, -1, sam_feat.shape[2], sam_feat.shape[3])
                        else:
                            s_feat = sam_feat

                        fused_list = [fusion_layer(feat_stack[:, ti], s_feat[:, ti]) for ti in range(t)]
                        fused_stack = torch.stack(fused_list, dim=1)
                        break

                    new_feats_levels.append([fused_stack[:, ti] for ti in range(t)])
                feats_levels = new_feats_levels

        if self.sam_cfg.enabled and self.sam_cfg.use_mask_guidance and mask_labels is not None:
            boundary_maps = torch.stack(
                [compute_boundary_map(mask_labels[:, ti]) for ti in range(t)],
                dim=1,
            )

            if self.boundary_concat_layers is not None:
                new_feats_levels = []
                for i, level_feats in enumerate(feats_levels):
                    if i >= len(self.boundary_concat_layers):
                        new_feats_levels.append(level_feats)
                        continue

                    concat_layer = self.boundary_concat_layers[i]
                    enhanced_list = []
                    for ti in range(t):
                        seg_labels = mask_labels[:, ti : ti + 1].float() / max(float(self.sam_cfg.num_segments), 1.0)
                        feat_h, feat_w = level_feats[ti].shape[-2:]
                        boundary_t = F.interpolate(boundary_maps[:, ti], size=(feat_h, feat_w), mode="nearest")
                        seg_labels_t = F.interpolate(seg_labels, size=(feat_h, feat_w), mode="nearest")
                        enhanced_list.append(concat_layer(level_feats[ti], boundary_t, seg_labels_t))

                    new_feats_levels.append(enhanced_list)
                feats_levels = new_feats_levels

        return feats_levels, boundary_maps, mask_labels

    def _build_segment_tokens(
        self,
        level8_feats: List[torch.Tensor],
        mask_labels: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if mask_labels is None or self.segment_token_proj is None:
            return None

        b = mask_labels.shape[0]
        t = mask_labels.shape[1]
        s = int(getattr(self.sam_cfg, "num_segments", 32))
        d = self.model_cfg.token_dim

        seg_tokens_all = []
        for ti in range(t):
            feat_t = self.segment_token_proj(level8_feats[ti])
            _, _, h8, w8 = feat_t.shape

            labels_t = mask_labels[:, ti : ti + 1].float()
            labels_t = F.interpolate(labels_t, size=(h8, w8), mode="nearest").squeeze(1).long()

            tok_t = feat_t.new_zeros((b, s, d))
            for bi in range(b):
                labels_b = labels_t[bi]
                feat_b = feat_t[bi]
                for seg_id in range(1, s + 1):
                    seg_mask = labels_b == seg_id
                    if seg_mask.any():
                        tok_t[bi, seg_id - 1] = feat_b[:, seg_mask].mean(dim=1)
            seg_tokens_all.append(tok_t)

        return torch.stack(seg_tokens_all, dim=1)

    def _run_v45_matcher(
        self,
        feats_levels: List[List[torch.Tensor]],
        boundary_maps: Optional[torch.Tensor],
        mask_labels: Optional[torch.Tensor],
    ) -> Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        assert self.tokenizer_v45_main is not None
        assert self.lcm_v45 is not None

        t = len(feats_levels[0])
        boundary_pairs = None
        if boundary_maps is not None:
            boundary_pairs = [boundary_maps[:, ti] for ti in range(max(0, t - 1))]

        tokens = self.tokenizer_v45_main(
            feats_levels,
            boundary_maps=boundary_pairs,
            attn_biases=None,
        )

        if self.tokenizer_v45_aux is not None and mask_labels is not None and self.mask_corr_aux_alphas is not None:
            aux_tokens = self.tokenizer_v45_aux(feats_levels, segment_masks=mask_labels)
            alphas = torch.sigmoid(self.mask_corr_aux_alphas)
            fused = []
            for lvl_idx in range(len(tokens)):
                lvl = []
                for ti in range(len(tokens[lvl_idx])):
                    lvl.append(tokens[lvl_idx][ti] + alphas[lvl_idx] * aux_tokens[lvl_idx][ti])
                fused.append(lvl)
            tokens = fused

        seg_tokens = self._build_segment_tokens(feats_levels[1], mask_labels)
        self.lcm_v45.reset_memory()
        latent = self.lcm_v45(tokens, seg_tokens=seg_tokens, attn_bias=None)
        return latent, tokens

    def _run_v4_backbone(
        self,
        feats_levels: List[List[torch.Tensor]],
        mask_labels: Optional[torch.Tensor],
    ) -> List[torch.Tensor]:
        tokens = self.tokenizer(
            feats_levels,
            segment_masks=(
                mask_labels
                if bool(self.sam_cfg.enabled and getattr(self.sam_cfg, "add_mask_corr", False))
                else None
            ),
        )
        latent = self.lcm(tokens)
        coarse_flows = self.gtr(latent, feats_levels)
        return self.decoder(coarse_flows, feats_levels, latent)

    def _run_v45_matcher_lcm(
        self,
        feats_levels: List[List[torch.Tensor]],
        boundary_maps: Optional[torch.Tensor],
        mask_labels: Optional[torch.Tensor],
    ) -> List[torch.Tensor]:
        latent, _ = self._run_v45_matcher(feats_levels, boundary_maps, mask_labels)
        coarse_flows = self.gtr(latent, feats_levels)
        return self.decoder(coarse_flows, feats_levels, latent)

    def _run_v45_hybrid(
        self,
        feats_levels: List[List[torch.Tensor]],
        boundary_maps: Optional[torch.Tensor],
        mask_labels: Optional[torch.Tensor],
    ) -> List[torch.Tensor]:
        assert self.iterative_refiner is not None

        latent, _ = self._run_v45_matcher(feats_levels, boundary_maps, mask_labels)
        flows = []

        pair_count = len(feats_levels[1]) - 1
        for k in range(pair_count):
            f1 = feats_levels[1][k]
            f2 = feats_levels[1][k + 1]
            h8, w8 = f1.shape[-2:]

            prior = latent[1][k]
            if prior.dim() == 3:
                prior = prior.view(prior.shape[0], prior.shape[1], h8, w8)

            boundary_k = None
            if boundary_maps is not None and k < boundary_maps.shape[1]:
                boundary_k = boundary_maps[:, k]

            segment_k = None
            if mask_labels is not None and k < mask_labels.shape[1]:
                segment_k = mask_labels[:, k : k + 1].float() / max(float(self.sam_cfg.num_segments), 1.0)

            flow_k = self.iterative_refiner(
                feat1=f1,
                feat2=f2,
                context_feat=f1,
                prior=prior,
                boundary=boundary_k,
                segment=segment_k,
                iters=int(getattr(self.model_cfg, "refiner_iters", 10)),
            )
            flows.append(flow_k)

        return flows

    def _get_flows(
        self,
        clip: torch.Tensor,
        sam_masks: Optional[torch.Tensor] = None,
        sam_features: Optional[Dict] = None,
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        feats_levels, boundary_maps, mask_labels = self._extract_features(
            clip,
            sam_masks=sam_masks,
            sam_features=sam_features,
        )

        if self.backbone == "v4_5_hybrid_sam":
            flows = self._run_v45_hybrid(feats_levels, boundary_maps, mask_labels)
        elif self.backbone == "v4_5_matcher_lcm":
            flows = self._run_v45_matcher_lcm(feats_levels, boundary_maps, mask_labels)
        else:
            flows = self._run_v4_backbone(feats_levels, mask_labels)

        return flows, boundary_maps

    def forward(
        self,
        clip: torch.Tensor,
        sam_masks: Optional[torch.Tensor] = None,
        sam_features: Optional[Dict] = None,
        return_losses: bool = False,
        compute_sam_losses: bool = True,
    ) -> Dict[str, torch.Tensor]:
        _, t, _, _, _ = clip.shape

        flows_fw, bn_fw = self._get_flows(clip, sam_masks, sam_features)
        outputs = {
            "flows": flows_fw,
            "flows_fw": flows_fw,
            "flows_bw": [],
        }

        if return_losses:
            clip_bw = torch.flip(clip, [1])
            masks_bw = torch.flip(sam_masks, [1]) if sam_masks is not None else None

            feats_bw = None
            if sam_features is not None:
                feats_bw = {}
                for k, v in sam_features.items():
                    if v.dim() == 5:
                        feats_bw[k] = torch.flip(v, [1])
                    else:
                        feats_bw[k] = v

            flows_bw_list, _ = self._get_flows(clip_bw, masks_bw, feats_bw)
            if len(flows_bw_list) == len(flows_fw):
                ln = len(flows_bw_list)
                outputs["flows_bw"] = [flows_bw_list[ln - 1 - i] for i in range(ln)]
            else:
                outputs["flows_bw"] = flows_bw_list

            if compute_sam_losses and sam_masks is not None and len(flows_fw) > 0:
                mask_labels_all = self._normalize_mask_labels(sam_masks)
                sam_loss_accum: Dict[str, torch.Tensor] = {}
                count = 0

                for i, flow in enumerate(flows_fw):
                    if i >= t - 1:
                        break

                    if mask_labels_all is None:
                        break
                    mask_i = mask_labels_all[:, i]

                    h_flow, w_flow = flow.shape[2:]
                    if mask_i.shape[1:] != (h_flow, w_flow):
                        mask_i = F.interpolate(
                            mask_i.unsqueeze(1).float(),
                            size=(h_flow, w_flow),
                            mode="nearest",
                        ).squeeze(1).long()

                    bn_i = bn_fw[:, i] if bn_fw is not None else None
                    if bn_i is not None:
                        if bn_i.dim() == 3:
                            bn_i = bn_i.unsqueeze(1)
                        if bn_i.shape[2:] != (h_flow, w_flow):
                            bn_i = F.interpolate(
                                bn_i,
                                size=(h_flow, w_flow),
                                mode="bilinear",
                                align_corners=False,
                            )

                    img_i = clip[:, i]
                    if img_i.shape[2:] != (h_flow, w_flow):
                        img_i = F.interpolate(
                            img_i,
                            size=(h_flow, w_flow),
                            mode="bilinear",
                            align_corners=False,
                        )

                    losses = self.sam_loss_bundle(
                        flow=flow,
                        masks=mask_i,
                        images=img_i,
                        boundary=bn_i,
                    )
                    for k, v in losses.items():
                        if k not in sam_loss_accum:
                            sam_loss_accum[k] = v
                        else:
                            sam_loss_accum[k] = sam_loss_accum[k] + v
                    count += 1

                if count > 0:
                    outputs["sam_losses"] = {k: v / count for k, v in sam_loss_accum.items()}

        return outputs

    @classmethod
    def from_config_file(cls, config_path: str) -> "AniFlowFormerTV4":
        import yaml

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(config=config_dict)
