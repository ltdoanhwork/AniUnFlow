from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.aniunflow.encoder import PyramidEncoder
from models.aniunflow.utils import warp
from models.aniunflow_v4.fusion import ContextFusionModule
from models.aniunflow_v4.losses import compute_boundary_map
from models.aniunflow_v4.sam_encoder import SAMEncoderWrapper

from .config import V5Config


def resize_flow(flow: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    if flow.shape[-2:] == size:
        return flow
    h0, w0 = flow.shape[-2:]
    h1, w1 = size
    out = F.interpolate(flow, size=size, mode="bilinear", align_corners=True)
    out[:, 0] *= float(w1) / max(float(w0), 1.0)
    out[:, 1] *= float(h1) / max(float(h0), 1.0)
    return out


class ResidualFlowRefiner(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int, num_blocks: int):
        super().__init__()
        in_dim = feat_dim * 2 + 4
        layers: List[nn.Module] = [
            nn.Conv2d(in_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        ]
        for _ in range(max(1, num_blocks)):
            layers.extend(
                [
                    nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                    nn.GroupNorm(8, hidden_dim),
                    nn.GELU(),
                ]
            )
        self.body = nn.Sequential(*layers)
        self.flow_head = nn.Conv2d(hidden_dim, 2, 3, padding=1)

    def forward(
        self,
        feat_a: torch.Tensor,
        feat_b: torch.Tensor,
        coarse_flow: torch.Tensor,
        boundary: torch.Tensor,
        conf_map: torch.Tensor,
    ) -> torch.Tensor:
        if boundary.shape[-2:] != feat_a.shape[-2:]:
            boundary = F.interpolate(boundary, size=feat_a.shape[-2:], mode="nearest")
        if conf_map.shape[-2:] != feat_a.shape[-2:]:
            conf_map = F.interpolate(conf_map, size=feat_a.shape[-2:], mode="nearest")
        coarse_flow = resize_flow(coarse_flow, feat_a.shape[-2:])
        x = torch.cat([feat_a, feat_b, coarse_flow, boundary, conf_map], dim=1)
        return self.flow_head(self.body(x))


class LocalCorrelationMatcher(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int, radius: int, delta_scale: float):
        super().__init__()
        self.radius = int(radius)
        self.delta_scale = float(delta_scale)
        corr_dim = (2 * self.radius + 1) ** 2
        in_dim = feat_dim * 2 + 3 + corr_dim
        self.stem = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        self.delta_head = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.conf_head = nn.Conv2d(hidden_dim, 1, 3, padding=1)

    @staticmethod
    def _shift(feat: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
        shifted = torch.roll(feat, shifts=(dy, dx), dims=(-2, -1))
        if dy > 0:
            shifted[:, :, :dy, :] = 0
        elif dy < 0:
            shifted[:, :, dy:, :] = 0
        if dx > 0:
            shifted[:, :, :, :dx] = 0
        elif dx < 0:
            shifted[:, :, :, dx:] = 0
        return shifted

    def _correlate(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        feat_a = F.normalize(feat_a, dim=1)
        feat_b = F.normalize(feat_b, dim=1)
        corrs = []
        for dy in range(-self.radius, self.radius + 1):
            for dx in range(-self.radius, self.radius + 1):
                shifted = self._shift(feat_b, dx, dy)
                corrs.append((feat_a * shifted).sum(dim=1, keepdim=True))
        return torch.cat(corrs, dim=1)

    def forward(
        self,
        feat_a: torch.Tensor,
        feat_b: torch.Tensor,
        prior_flow: Optional[torch.Tensor] = None,
        conf_map: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if prior_flow is None:
            prior_flow = feat_a.new_zeros((feat_a.shape[0], 2, feat_a.shape[2], feat_a.shape[3]))
        else:
            prior_flow = resize_flow(prior_flow, feat_a.shape[-2:])
        if conf_map is None:
            conf_map = feat_a.new_zeros((feat_a.shape[0], 1, feat_a.shape[2], feat_a.shape[3]))
        elif conf_map.shape[-2:] != feat_a.shape[-2:]:
            conf_map = F.interpolate(conf_map, size=feat_a.shape[-2:], mode="nearest")

        feat_b_warp = warp(feat_b, prior_flow)
        corr = self._correlate(feat_a, feat_b_warp)
        x = torch.cat([feat_a, feat_b_warp, prior_flow, conf_map, corr], dim=1)
        hidden = self.stem(x)
        delta = torch.tanh(self.delta_head(hidden)) * self.delta_scale
        confidence = torch.sigmoid(self.conf_head(hidden))
        return delta, confidence


class AniFlowFormerTV5(nn.Module):
    def __init__(self, config: V5Config | dict | None = None):
        super().__init__()
        if config is None:
            config = V5Config()
        elif isinstance(config, dict):
            config = V5Config.from_dict(config)

        self.config = config
        self.model_cfg = config.model
        self.sam_cfg = config.sam
        self.backbone = str(self.model_cfg.backbone).lower()
        self.use_dense_correlation = self.backbone.startswith("v5_1") or "dense" in self.backbone

        c = self.model_cfg.enc_channels
        d = self.model_cfg.slot_dim
        self.encoder = PyramidEncoder(c=c)
        self.level1_proj = nn.Conv2d(c, d, 1)
        self.level2_proj = nn.Conv2d(c * 2, d, 1)
        self.slot_fusion = nn.Sequential(
            nn.Linear(d * 2, self.model_cfg.slot_hidden_dim),
            nn.GELU(),
            nn.Linear(self.model_cfg.slot_hidden_dim, d),
        )

        if self.model_cfg.temporal_memory_depth > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d,
                nhead=self.model_cfg.temporal_memory_heads,
                dim_feedforward=self.model_cfg.slot_hidden_dim * 2,
                dropout=self.model_cfg.temporal_memory_dropout,
                batch_first=True,
                activation="gelu",
            )
            self.temporal_memory = nn.TransformerEncoder(
                encoder_layer,
                num_layers=self.model_cfg.temporal_memory_depth,
            )
        else:
            self.temporal_memory = nn.Identity()
        self.max_frames = 8
        self.time_embed = nn.Parameter(torch.randn(1, self.max_frames, 1, d) * 0.02)
        self.slot_embed = nn.Parameter(torch.randn(1, 1, self.model_cfg.num_slots, d) * 0.02)

        self.motion_head = nn.Sequential(
            nn.Linear(d * 2, self.model_cfg.slot_hidden_dim),
            nn.GELU(),
            nn.Linear(self.model_cfg.slot_hidden_dim, 6),
        )
        self.order_head = nn.Sequential(
            nn.Linear(d * 2, self.model_cfg.slot_hidden_dim),
            nn.GELU(),
            nn.Linear(self.model_cfg.slot_hidden_dim, 1),
        )
        self.occ_head = nn.Sequential(
            nn.Linear(d * 2, self.model_cfg.slot_hidden_dim),
            nn.GELU(),
            nn.Linear(self.model_cfg.slot_hidden_dim, 1),
        )

        if self.use_dense_correlation:
            self.l2_matcher = LocalCorrelationMatcher(
                feat_dim=c * 2,
                hidden_dim=self.model_cfg.dense_match_hidden_dim,
                radius=self.model_cfg.dense_match_radius_l2,
                delta_scale=self.model_cfg.dense_delta_scale_l2,
            )
            self.l1_matcher = LocalCorrelationMatcher(
                feat_dim=c,
                hidden_dim=self.model_cfg.dense_match_hidden_dim,
                radius=self.model_cfg.dense_match_radius_l1,
                delta_scale=self.model_cfg.dense_delta_scale_l1,
            )
        else:
            self.l2_matcher = None
            self.l1_matcher = None

        self.residual_refiner = ResidualFlowRefiner(
            feat_dim=c,
            hidden_dim=self.model_cfg.residual_hidden_dim,
            num_blocks=self.model_cfg.residual_blocks,
        )

        self.sam_encoder = None
        self.sam_fuser = None
        if self.sam_cfg.enabled and self.sam_cfg.use_encoder_features:
            self.sam_encoder = SAMEncoderWrapper(
                checkpoint=self.sam_cfg.encoder_checkpoint,
                config=self.sam_cfg.encoder_config,
                freeze=self.sam_cfg.encoder_freeze,
                feature_scales=self.sam_cfg.feature_scales,
            )
            self.sam_fuser = ContextFusionModule(
                flow_dim=c * 2,
                sam_dim=self.sam_cfg.feature_dim,
                out_dim=c * 2,
            )

    @staticmethod
    def _normalize_mask_labels(sam_masks: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if sam_masks is None:
            return None
        if sam_masks.dim() == 5:
            if sam_masks.shape[2] == 1:
                return sam_masks[:, :, 0].long()
            return sam_masks.argmax(dim=2).long() + 1
        if sam_masks.dim() == 4:
            if sam_masks.shape[1] == 1:
                return sam_masks[:, 0].long()
            return sam_masks.long()
        if sam_masks.dim() == 3:
            return sam_masks.long()
        return None

    def _extract_features(
        self,
        clip: torch.Tensor,
        sam_masks: Optional[torch.Tensor] = None,
        sam_features: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Tuple[List[List[torch.Tensor]], Optional[torch.Tensor], Optional[torch.Tensor]]:
        b, t, c_img, h_img, w_img = clip.shape
        feats_levels = self.encoder(clip)
        mask_labels = self._normalize_mask_labels(sam_masks)
        boundary_maps = None

        if self.sam_encoder is not None:
            sam_feat_dict = sam_features
            if sam_feat_dict is None:
                frames_flat = clip.view(b * t, c_img, h_img, w_img)
                sam_feat_dict = self.sam_encoder(frames_flat)
            if sam_feat_dict and self.sam_fuser is not None:
                fused_level = []
                feat_stack = torch.stack(feats_levels[1], dim=1)
                for sam_feat in sam_feat_dict.values():
                    if sam_feat.dim() == 4:
                        sam_feat = sam_feat.view(b, t, sam_feat.shape[1], sam_feat.shape[2], sam_feat.shape[3])
                    fused_level = [self.sam_fuser(feat_stack[:, ti], sam_feat[:, ti]) for ti in range(t)]
                    break
                if fused_level:
                    feats_levels[1] = fused_level

        if mask_labels is not None:
            boundary_maps = torch.stack([compute_boundary_map(mask_labels[:, ti]) for ti in range(t)], dim=1)

        return feats_levels, boundary_maps, mask_labels

    def _pool_slots(
        self,
        feat_l1: torch.Tensor,
        feat_l2: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = labels.shape[0]
        s = self.model_cfg.num_slots
        l1 = self.level1_proj(feat_l1)
        l2 = self.level2_proj(feat_l2)
        if labels.shape[-2:] != l1.shape[-2:]:
            labels_l1 = F.interpolate(labels.unsqueeze(1).float(), size=l1.shape[-2:], mode="nearest").squeeze(1).long()
        else:
            labels_l1 = labels.long()
        labels_l2 = F.interpolate(labels.unsqueeze(1).float(), size=l2.shape[-2:], mode="nearest").squeeze(1).long()
        labels_l1 = labels_l1.clamp(min=0, max=s)
        labels_l2 = labels_l2.clamp(min=0, max=s)

        def _scatter_pool(feat: torch.Tensor, lbl: torch.Tensor) -> torch.Tensor:
            feat_flat = feat.flatten(2).transpose(1, 2)
            lbl_flat = lbl.view(b, -1)
            tok_sum = feat.new_zeros((b, s + 1, feat.shape[1]))
            tok_cnt = feat.new_zeros((b, s + 1, 1))
            tok_sum.scatter_add_(1, lbl_flat.unsqueeze(-1).expand(-1, -1, feat.shape[1]), feat_flat)
            tok_cnt.scatter_add_(1, lbl_flat.unsqueeze(-1), feat.new_ones((b, lbl_flat.shape[1], 1)))
            return tok_sum[:, 1:] / tok_cnt[:, 1:].clamp_min(1.0)

        slots = self.slot_fusion(torch.cat([_scatter_pool(l1, labels_l1), _scatter_pool(l2, labels_l2)], dim=-1))
        return slots, labels_l1, labels_l2

    def _temporal_slots(
        self,
        feat_levels: List[List[torch.Tensor]],
        mask_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        t = len(feat_levels[0])
        slots_all = []
        labels_l1_all = []
        labels_l2_all = []
        for ti in range(t):
            slots_t, labels_l1_t, labels_l2_t = self._pool_slots(feat_levels[0][ti], feat_levels[1][ti], mask_labels[:, ti])
            slots_all.append(slots_t)
            labels_l1_all.append(labels_l1_t)
            labels_l2_all.append(labels_l2_t)

        slots = torch.stack(slots_all, dim=1)
        slots = slots + self.time_embed[:, :t] + self.slot_embed[:, :, : slots.shape[2]]
        slots_mem = self.temporal_memory(slots.view(slots.shape[0], -1, slots.shape[-1]))
        slots_mem = slots_mem.view_as(slots)
        return slots_mem, labels_l1_all, labels_l2_all

    def _mask_iou(self, labels_a: torch.Tensor, labels_b: torch.Tensor) -> torch.Tensor:
        s = self.model_cfg.num_slots
        one_a = F.one_hot(labels_a.clamp(min=0, max=s), num_classes=s + 1)[..., 1:].float()
        one_b = F.one_hot(labels_b.clamp(min=0, max=s), num_classes=s + 1)[..., 1:].float()
        one_a = one_a.view(labels_a.shape[0], -1, s)
        one_b = one_b.view(labels_b.shape[0], -1, s)
        inter = torch.einsum("bns,bnt->bst", one_a, one_b)
        area_a = one_a.sum(dim=1).unsqueeze(-1)
        area_b = one_b.sum(dim=1).unsqueeze(1)
        union = area_a + area_b - inter
        return inter / union.clamp_min(1.0)

    def _match_slots(
        self,
        slots_a: torch.Tensor,
        slots_b: torch.Tensor,
        labels_a: torch.Tensor,
        labels_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = torch.matmul(slots_a, slots_b.transpose(1, 2)) / math.sqrt(slots_a.shape[-1])
        logits = logits + self.model_cfg.overlap_prior_weight * self._mask_iou(labels_a, labels_b)
        probs = logits.softmax(dim=-1)
        conf, _ = probs.max(dim=-1)
        matched_slots = torch.matmul(probs, slots_b)
        return probs, conf, matched_slots

    def _slot_scalar_map(self, labels: torch.Tensor, slot_values: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(labels.clamp(min=0, max=slot_values.shape[1]), num_classes=slot_values.shape[1] + 1)[..., 1:]
        one_hot = one_hot.permute(0, 3, 1, 2).float()
        return torch.einsum("bshw,bs->bhw", one_hot, slot_values).unsqueeze(1)

    def _affine_grid_flow(
        self,
        labels: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        b, h, w = labels.shape
        s = params.shape[1]
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=labels.device),
            torch.linspace(-1.0, 1.0, w, device=labels.device),
            indexing="ij",
        )
        x = xs.view(1, 1, h, w)
        y = ys.view(1, 1, h, w)
        u = params[:, :, 0:1, None] * x + params[:, :, 1:2, None] * y + params[:, :, 2:3, None]
        v = params[:, :, 3:4, None] * x + params[:, :, 4:5, None] * y + params[:, :, 5:6, None]
        slot_flow = torch.cat([u, v], dim=2)
        slot_flow = slot_flow.view(b, s, 2, h, w)
        one_hot = F.one_hot(labels.clamp(min=0, max=s), num_classes=s + 1)[..., 1:].permute(0, 3, 1, 2).float()
        flow = torch.einsum("bshw,bschw->bchw", one_hot, slot_flow)
        flow[:, 0] *= w * 0.5
        flow[:, 1] *= h * 0.5
        return flow

    def _compose_flow(self, flow_01: torch.Tensor, flow_12: torch.Tensor) -> torch.Tensor:
        flow_12 = resize_flow(flow_12, flow_01.shape[-2:])
        return flow_01 + warp(flow_12, flow_01)

    def _confidence_gate(self, support_conf: torch.Tensor, pred_conf: torch.Tensor) -> torch.Tensor:
        support_conf = support_conf.clamp(0.0, 1.0)
        pred_conf = pred_conf.clamp(0.0, 1.0)
        # Dense refinement should not override the object prior when slot matching is weak.
        joint = pred_conf * support_conf
        floor = float(self.model_cfg.dense_confidence_floor)
        return floor + (1.0 - floor) * joint

    def _pair_flow(
        self,
        feat_a_l1: torch.Tensor,
        feat_b_l1: torch.Tensor,
        feat_a_l2: torch.Tensor,
        feat_b_l2: torch.Tensor,
        slots_a: torch.Tensor,
        slots_b: torch.Tensor,
        labels_a_l1: torch.Tensor,
        labels_b_l1: torch.Tensor,
        labels_a_l2: torch.Tensor,
        labels_b_l2: torch.Tensor,
        boundary: Optional[torch.Tensor],
        enable_residual_branch: bool,
    ) -> Dict[str, torch.Tensor]:
        probs, conf, matched_slots = self._match_slots(slots_a, slots_b, labels_a_l1, labels_b_l1)
        pair_feat = torch.cat([slots_a, matched_slots], dim=-1)
        params = torch.tanh(self.motion_head(pair_feat)) * 0.25
        order = torch.tanh(self.order_head(pair_feat).squeeze(-1)) * 2.0
        occ = torch.sigmoid(self.occ_head(pair_feat).squeeze(-1))

        slot_flow_l2 = self._affine_grid_flow(labels_a_l2, params)
        conf_map_l2 = self._slot_scalar_map(labels_a_l2, conf)
        dense_prior_l2 = slot_flow_l2
        corr_conf_l2 = conf_map_l2.new_zeros(conf_map_l2.shape)
        if self.use_dense_correlation and self.l2_matcher is not None:
            delta_l2, corr_conf_l2 = self.l2_matcher(feat_a_l2, feat_b_l2, slot_flow_l2, conf_map_l2)
            gate_l2 = self._confidence_gate(conf_map_l2, corr_conf_l2)
            dense_prior_l2 = slot_flow_l2 + self.model_cfg.dense_update_scale_l2 * delta_l2 * gate_l2

        slot_flow_l1 = self._affine_grid_flow(labels_a_l1, params)
        dense_prior_up = resize_flow(dense_prior_l2, feat_a_l1.shape[-2:])
        corr_conf_up = F.interpolate(corr_conf_l2, size=feat_a_l1.shape[-2:], mode="bilinear", align_corners=False)
        conf_map_l1 = self._slot_scalar_map(labels_a_l1, conf)
        if self.use_dense_correlation and self.l1_matcher is not None:
            blended_prior = (
                self.model_cfg.dense_prior_mix * dense_prior_up
                + (1.0 - self.model_cfg.dense_prior_mix) * slot_flow_l1
            )
            corr_seed = 0.5 * (conf_map_l1 + corr_conf_up)
            delta_l1, corr_conf_l1 = self.l1_matcher(feat_a_l1, feat_b_l1, blended_prior, corr_seed)
            gate_l1 = self._confidence_gate(corr_seed, corr_conf_l1)
            dense_prior = blended_prior + self.model_cfg.dense_update_scale_l1 * delta_l1 * gate_l1
            corr_conf = torch.maximum(corr_conf_up, corr_conf_l1)
        else:
            dense_prior = slot_flow_l1
            corr_conf = conf_map_l1.new_zeros(conf_map_l1.shape)

        boundary_l1 = boundary if boundary is not None else torch.zeros_like(conf_map_l1)
        if boundary_l1.shape[-2:] != feat_a_l1.shape[-2:]:
            boundary_l1 = F.interpolate(boundary_l1, size=feat_a_l1.shape[-2:], mode="nearest")
        refiner_conf = 0.5 * (conf_map_l1 + corr_conf) if self.use_dense_correlation else conf_map_l1
        if enable_residual_branch:
            residual = self.residual_refiner(feat_a_l1, feat_b_l1, dense_prior, boundary_l1, refiner_conf)
            residual_gate = boundary_l1 * self.model_cfg.residual_boundary_scale + self.model_cfg.residual_base_scale
            residual_gate = residual_gate * (
                self.model_cfg.residual_confidence_floor
                + (1.0 - self.model_cfg.residual_confidence_floor) * refiner_conf.clamp(0.0, 1.0)
            )
            residual = residual * residual_gate
        else:
            residual = torch.zeros_like(dense_prior)
        flow = dense_prior + residual
        return {
            "flow": flow,
            "coarse_flow": dense_prior,
            "slot_flow": slot_flow_l1,
            "dense_prior_flow": dense_prior,
            "residual_flow": residual,
            "match_probs": probs,
            "match_confidence": conf,
            "corr_confidence": corr_conf.squeeze(1),
            "segment_params": params,
            "layer_order": order,
            "occlusion_slots": occ,
        }

    def _forward_single_direction(
        self,
        clip: torch.Tensor,
        sam_masks: Optional[torch.Tensor],
        sam_features: Optional[Dict[int, torch.Tensor]],
        enable_residual_branch: bool,
    ) -> Dict[str, List[torch.Tensor]]:
        feats_levels, boundary_maps, mask_labels = self._extract_features(
            clip,
            sam_masks=sam_masks,
            sam_features=sam_features,
        )
        if mask_labels is None:
            raise ValueError("AniFlowFormerTV5 requires SAM masks for object-memory training.")

        slots_mem, labels_l1_all, labels_l2_all = self._temporal_slots(feats_levels, mask_labels)
        outputs: Dict[str, List[torch.Tensor]] = {
            "flows": [],
            "coarse_flow": [],
            "slot_flow": [],
            "dense_prior_flow": [],
            "residual_flow": [],
            "match_probs": [],
            "match_confidence": [],
            "corr_confidence": [],
            "segment_params": [],
            "layer_order": [],
            "occlusion_slots": [],
            "match_probs_long": [],
            "layer_order_all": [slots_mem.new_zeros((slots_mem.shape[0], slots_mem.shape[2])) for _ in range(slots_mem.shape[1])],
        }

        for k in range(clip.shape[1] - 1):
            boundary_k = boundary_maps[:, k] if boundary_maps is not None else None
            pair_out = self._pair_flow(
                feats_levels[0][k],
                feats_levels[0][k + 1],
                feats_levels[1][k],
                feats_levels[1][k + 1],
                slots_mem[:, k],
                slots_mem[:, k + 1],
                labels_l1_all[k],
                labels_l1_all[k + 1],
                labels_l2_all[k],
                labels_l2_all[k + 1],
                boundary_k,
                enable_residual_branch=enable_residual_branch,
            )
            for key in (
                "flow",
                "coarse_flow",
                "slot_flow",
                "dense_prior_flow",
                "residual_flow",
                "match_probs",
                "match_confidence",
                "corr_confidence",
                "segment_params",
                "layer_order",
                "occlusion_slots",
            ):
                outputs[key if key != "flow" else "flows"].append(pair_out[key])
            outputs["layer_order_all"][k] = pair_out["layer_order"]
            if k == clip.shape[1] - 2:
                outputs["layer_order_all"][k + 1] = torch.matmul(
                    pair_out["match_probs"].transpose(1, 2),
                    pair_out["layer_order"].unsqueeze(-1),
                ).squeeze(-1)

        if self.model_cfg.use_long_gap_matching and clip.shape[1] >= 3:
            for k in range(clip.shape[1] - 2):
                probs_long, _, _ = self._match_slots(
                    slots_mem[:, k],
                    slots_mem[:, k + 2],
                    labels_l1_all[k],
                    labels_l1_all[k + 2],
                )
                outputs["match_probs_long"].append(probs_long)

        return outputs

    def forward(
        self,
        clip: torch.Tensor,
        sam_masks: Optional[torch.Tensor] = None,
        sam_features: Optional[Dict[int, torch.Tensor]] = None,
        return_losses: bool = False,
        enable_residual_branch: bool = True,
        **_: Dict,
    ) -> Dict[str, torch.Tensor]:
        out_fw = self._forward_single_direction(
            clip,
            sam_masks=sam_masks,
            sam_features=sam_features,
            enable_residual_branch=enable_residual_branch,
        )
        flows_long = []
        for k in range(max(0, len(out_fw["flows"]) - 1)):
            flows_long.append(self._compose_flow(out_fw["flows"][k], out_fw["flows"][k + 1]))

        outputs = {
            "flows": out_fw["flows"],
            "flows_fw": out_fw["flows"],
            "flows_bw": [],
            "flows_long": flows_long,
            "segment_params_fw": out_fw["segment_params"],
            "segment_params_bw": [],
            "match_probs_fw": out_fw["match_probs"],
            "match_probs_bw": [],
            "match_probs_long": out_fw["match_probs_long"],
            "match_confidence_fw": out_fw["match_confidence"],
            "match_confidence_bw": [],
            "corr_confidence_fw": out_fw["corr_confidence"],
            "corr_confidence_bw": [],
            "layer_order_fw": out_fw["layer_order"],
            "layer_order_bw": [],
            "layer_order_all": out_fw["layer_order_all"],
            "occlusion_slots_fw": out_fw["occlusion_slots"],
            "occlusion_slots_bw": [],
            "slot_flow_fw": out_fw["slot_flow"],
            "slot_flow_bw": [],
            "dense_prior_flow_fw": out_fw["dense_prior_flow"],
            "dense_prior_flow_bw": [],
            "residual_flow_fw": out_fw["residual_flow"],
            "residual_flow_bw": [],
            "coarse_flow_fw": out_fw["coarse_flow"],
            "coarse_flow_bw": [],
        }

        if return_losses:
            clip_bw = torch.flip(clip, [1])
            masks_bw = torch.flip(sam_masks, [1]) if sam_masks is not None else None
            feats_bw = None
            if sam_features is not None:
                feats_bw = {}
                for key, value in sam_features.items():
                    feats_bw[key] = torch.flip(value, [1]) if value.dim() == 5 else value
            out_bw = self._forward_single_direction(
                clip_bw,
                sam_masks=masks_bw,
                sam_features=feats_bw,
                enable_residual_branch=enable_residual_branch,
            )
            n = len(out_bw["flows"])
            outputs["flows_bw"] = [out_bw["flows"][n - 1 - i] for i in range(n)]
            outputs["segment_params_bw"] = [out_bw["segment_params"][n - 1 - i] for i in range(n)]
            outputs["match_probs_bw"] = [out_bw["match_probs"][n - 1 - i] for i in range(n)]
            outputs["match_confidence_bw"] = [out_bw["match_confidence"][n - 1 - i] for i in range(n)]
            outputs["corr_confidence_bw"] = [out_bw["corr_confidence"][n - 1 - i] for i in range(n)]
            outputs["layer_order_bw"] = [out_bw["layer_order"][n - 1 - i] for i in range(n)]
            outputs["occlusion_slots_bw"] = [out_bw["occlusion_slots"][n - 1 - i] for i in range(n)]
            outputs["slot_flow_bw"] = [out_bw["slot_flow"][n - 1 - i] for i in range(n)]
            outputs["dense_prior_flow_bw"] = [out_bw["dense_prior_flow"][n - 1 - i] for i in range(n)]
            outputs["residual_flow_bw"] = [out_bw["residual_flow"][n - 1 - i] for i in range(n)]
            outputs["coarse_flow_bw"] = [out_bw["coarse_flow"][n - 1 - i] for i in range(n)]

        return outputs
