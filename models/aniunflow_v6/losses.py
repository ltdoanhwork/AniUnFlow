from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.aniunflow.utils import warp
from models.aniunflow_v4.losses import compute_boundary_map


def resize_flow(flow: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if flow.shape[-2:] == size:
        return flow
    h0, w0 = flow.shape[-2:]
    h1, w1 = size
    out = F.interpolate(flow, size=size, mode="bilinear", align_corners=True)
    out[:, 0] *= float(w1) / max(float(w0), 1.0)
    out[:, 1] *= float(h1) / max(float(h0), 1.0)
    return out


class V6GlobalSearchLossBundle(nn.Module):
    def __init__(
        self,
        num_slots: int = 32,
        segment_warp_weight: float = 0.20,
        dense_slot_consistency_weight: float = 0.18,
        boundary_residual_weight: float = 0.018,
        segment_cycle_weight: float = 0.05,
        global_fused_consistency_weight: float = 0.08,
        visibility_consistency_weight: float = 0.06,
        occlusion_composite_weight: float = 0.05,
        slot_deformation_reg_weight: float = 0.02,
        hard_motion_reweight_weight: float = 0.06,
    ):
        super().__init__()
        self.num_slots = int(num_slots)
        self.segment_warp_weight = float(segment_warp_weight)
        self.dense_slot_consistency_weight = float(dense_slot_consistency_weight)
        self.boundary_residual_weight = float(boundary_residual_weight)
        self.segment_cycle_weight = float(segment_cycle_weight)
        self.global_fused_consistency_weight = float(global_fused_consistency_weight)
        self.visibility_consistency_weight = float(visibility_consistency_weight)
        self.occlusion_composite_weight = float(occlusion_composite_weight)
        self.slot_deformation_reg_weight = float(slot_deformation_reg_weight)
        self.hard_motion_reweight_weight = float(hard_motion_reweight_weight)

    @classmethod
    def from_config(cls, cfg: Dict) -> "V6GlobalSearchLossBundle":
        model_cfg = cfg.get("model", {})
        loss_cfg = cfg.get("loss", {})
        return cls(
            num_slots=int(model_cfg.get("num_slots", cfg.get("sam", {}).get("num_segments", 32))),
            segment_warp_weight=float(loss_cfg.get("segment_warp", 0.20)),
            dense_slot_consistency_weight=float(loss_cfg.get("dense_slot_consistency", 0.18)),
            boundary_residual_weight=float(loss_cfg.get("boundary_residual", 0.018)),
            segment_cycle_weight=float(loss_cfg.get("segment_cycle", 0.05)),
            global_fused_consistency_weight=float(loss_cfg.get("global_fused_consistency", 0.08)),
            visibility_consistency_weight=float(loss_cfg.get("visibility_consistency", 0.06)),
            occlusion_composite_weight=float(loss_cfg.get("occlusion_composite", 0.05)),
            slot_deformation_reg_weight=float(loss_cfg.get("slot_deformation_reg", 0.02)),
            hard_motion_reweight_weight=float(loss_cfg.get("hard_motion_reweight", 0.06)),
        )

    def _normalize_labels(self, sam_masks: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if sam_masks is None:
            return None
        if sam_masks.dim() == 5:
            if sam_masks.shape[2] == 1:
                return sam_masks[:, :, 0].long()
            return sam_masks.argmax(dim=2).long() + 1
        if sam_masks.dim() == 4:
            if sam_masks.shape[1] == 1:
                return sam_masks[:, 0].long()
            return sam_masks.argmax(dim=1).long() + 1
        if sam_masks.dim() == 3:
            return sam_masks.long()
        return None

    def _one_hot_labels(self, labels: torch.Tensor, num_slots: int) -> torch.Tensor:
        labels = labels.clamp(min=0, max=num_slots)
        one_hot = F.one_hot(labels, num_classes=num_slots + 1)[..., 1:]
        return one_hot.permute(0, 3, 1, 2).float()

    def _slot_scalar_map(self, labels: torch.Tensor, slot_values: torch.Tensor) -> torch.Tensor:
        one_hot = self._one_hot_labels(labels, slot_values.shape[1])
        return torch.einsum("bshw,bs->bhw", one_hot, slot_values).unsqueeze(1)

    def _segment_warp_consistency(
        self,
        labels: torch.Tensor,
        flows_fw: List[torch.Tensor],
        match_probs: List[torch.Tensor],
        match_conf: List[torch.Tensor],
    ) -> torch.Tensor:
        losses = []
        for k, flow in enumerate(flows_fw):
            if k + 1 >= labels.shape[1] or k >= len(match_probs):
                break
            src_labels = labels[:, k]
            tgt_labels = labels[:, k + 1]
            h, w = flow.shape[-2:]
            if src_labels.shape[-2:] != (h, w):
                src_labels = F.interpolate(src_labels.unsqueeze(1).float(), size=(h, w), mode="nearest").squeeze(1).long()
                tgt_labels = F.interpolate(tgt_labels.unsqueeze(1).float(), size=(h, w), mode="nearest").squeeze(1).long()
            src_one_hot = self._one_hot_labels(src_labels, self.num_slots)
            tgt_one_hot = self._one_hot_labels(tgt_labels, self.num_slots)
            warped_src = warp(src_one_hot, flow)
            soft_target = torch.einsum("bij,bjhw->bihw", match_probs[k], tgt_one_hot)
            conf = match_conf[k].unsqueeze(-1).unsqueeze(-1)
            losses.append(((warped_src - soft_target).abs() * conf).mean())
        if not losses:
            return labels.new_tensor(0.0, dtype=torch.float32)
        return sum(losses) / len(losses)

    def _segment_cycle(self, match_probs: List[torch.Tensor], long_match_probs: List[torch.Tensor]) -> torch.Tensor:
        losses = []
        limit = min(len(match_probs) - 1, len(long_match_probs))
        for k in range(max(0, limit)):
            cycle = torch.matmul(match_probs[k], match_probs[k + 1])
            losses.append((cycle - long_match_probs[k]).abs().mean())
        if not losses:
            ref = long_match_probs[0] if long_match_probs else None
            device = ref.device if ref is not None else "cpu"
            return torch.tensor(0.0, device=device)
        return sum(losses) / len(losses)

    def _dense_slot_consistency(
        self,
        labels: torch.Tensor,
        slot_flows: List[torch.Tensor],
        dense_priors: List[torch.Tensor],
        match_conf: List[torch.Tensor],
    ) -> torch.Tensor:
        losses = []
        limit = min(len(slot_flows), len(dense_priors), len(match_conf), labels.shape[1])
        for k in range(limit):
            slot_flow = slot_flows[k]
            dense_prior = dense_priors[k]
            if slot_flow.shape[-2:] != dense_prior.shape[-2:]:
                slot_flow = resize_flow(slot_flow, dense_prior.shape[-2:])
            lbl = labels[:, k]
            h, w = dense_prior.shape[-2:]
            if lbl.shape[-2:] != (h, w):
                lbl = F.interpolate(lbl.unsqueeze(1).float(), size=(h, w), mode="nearest").squeeze(1).long()
            boundary = compute_boundary_map(lbl).float()
            interior = 1.0 - boundary
            conf_map = self._slot_scalar_map(lbl, match_conf[k].clamp(0.0, 1.0))
            diff = torch.norm(dense_prior - slot_flow, dim=1, keepdim=True)
            losses.append((diff * interior * conf_map).mean())
        if not losses:
            return labels.new_tensor(0.0, dtype=torch.float32)
        return sum(losses) / len(losses)

    def _boundary_residual_specialization(self, labels: torch.Tensor, residual_flows: List[torch.Tensor]) -> torch.Tensor:
        losses = []
        for k, residual in enumerate(residual_flows):
            lbl = labels[:, k]
            h, w = residual.shape[-2:]
            if lbl.shape[-2:] != (h, w):
                lbl = F.interpolate(lbl.unsqueeze(1).float(), size=(h, w), mode="nearest").squeeze(1).long()
            boundary = compute_boundary_map(lbl).float()
            interior = 1.0 - boundary
            mag = torch.norm(residual, dim=1, keepdim=True)
            boundary_mean = (mag * boundary).sum() / boundary.sum().clamp_min(1.0)
            interior_mean = (mag * interior).sum() / interior.sum().clamp_min(1.0)
            losses.append(interior_mean / (boundary_mean + 1e-4))
        if not losses:
            return labels.new_tensor(0.0, dtype=torch.float32)
        return sum(losses) / len(losses)

    def _global_fused_consistency(
        self,
        global_flows: List[torch.Tensor],
        fused_flows: List[torch.Tensor],
        global_conf: List[torch.Tensor],
    ) -> torch.Tensor:
        losses = []
        limit = min(len(global_flows), len(fused_flows), len(global_conf))
        for k in range(limit):
            global_flow = resize_flow(global_flows[k], fused_flows[k].shape[-2:])
            conf = global_conf[k]
            if conf.dim() == 3:
                conf = conf.unsqueeze(1)
            if conf.shape[-2:] != fused_flows[k].shape[-2:]:
                conf = F.interpolate(conf, size=fused_flows[k].shape[-2:], mode="bilinear", align_corners=False)
            diff = torch.norm(fused_flows[k] - global_flow, dim=1, keepdim=True)
            losses.append((diff * conf).mean())
        if not losses:
            device = global_flows[0].device if global_flows else "cpu"
            return torch.tensor(0.0, device=device)
        return sum(losses) / len(losses)

    def _visibility_consistency(
        self,
        match_probs: List[torch.Tensor],
        match_conf: List[torch.Tensor],
        slot_visibility: List[torch.Tensor],
    ) -> torch.Tensor:
        losses = []
        for k in range(min(len(match_probs), len(match_conf), len(slot_visibility) - 1)):
            vis_a = slot_visibility[k].clamp(0.0, 1.0)
            vis_b = slot_visibility[k + 1].clamp(0.0, 1.0)
            matched_vis = torch.matmul(match_probs[k], vis_b.unsqueeze(-1)).squeeze(-1)
            conf = match_conf[k].clamp(0.0, 1.0)
            losses.append(((vis_a - matched_vis).abs() * conf).sum() / (conf.sum() + 1e-6))
        if not losses:
            device = slot_visibility[0].device if slot_visibility else "cpu"
            return torch.tensor(0.0, device=device)
        return sum(losses) / len(losses)

    def _occlusion_composite(
        self,
        labels: torch.Tensor,
        slot_flows: List[torch.Tensor],
        final_flows: List[torch.Tensor],
        dense_occ: List[torch.Tensor],
        slot_visibility: List[torch.Tensor],
    ) -> torch.Tensor:
        losses = []
        limit = min(len(slot_flows), len(final_flows), len(dense_occ), len(slot_visibility), labels.shape[1])
        for k in range(limit):
            slot_flow = resize_flow(slot_flows[k], final_flows[k].shape[-2:])
            occ = dense_occ[k]
            if occ.shape[-2:] != final_flows[k].shape[-2:]:
                occ = F.interpolate(occ, size=final_flows[k].shape[-2:], mode="bilinear", align_corners=False)
            lbl = labels[:, k]
            h, w = final_flows[k].shape[-2:]
            if lbl.shape[-2:] != (h, w):
                lbl = F.interpolate(lbl.unsqueeze(1).float(), size=(h, w), mode="nearest").squeeze(1).long()
            vis_map = self._slot_scalar_map(lbl, slot_visibility[k].clamp(0.0, 1.0))
            diff = torch.norm(final_flows[k] - slot_flow, dim=1, keepdim=True)
            losses.append((diff * vis_map * (1.0 - occ)).mean())
        if not losses:
            device = final_flows[0].device if final_flows else "cpu"
            return torch.tensor(0.0, device=device)
        return sum(losses) / len(losses)

    def _slot_deformation_reg(
        self,
        basis_coeffs: List[torch.Tensor],
        match_conf: List[torch.Tensor],
    ) -> torch.Tensor:
        losses = []
        limit = min(len(basis_coeffs), len(match_conf))
        for k in range(limit):
            coeff = basis_coeffs[k]
            conf = match_conf[k].clamp(0.0, 1.0)
            energy = coeff.pow(2).mean(dim=-1)
            losses.append((energy * (1.0 - conf)).mean())
        if not losses:
            device = basis_coeffs[0].device if basis_coeffs else "cpu"
            return torch.tensor(0.0, device=device)
        return sum(losses) / len(losses)

    def _hard_motion_reweight(
        self,
        final_flows: List[torch.Tensor],
        fused_flows: List[torch.Tensor],
        global_flows: List[torch.Tensor],
    ) -> torch.Tensor:
        losses = []
        limit = min(len(final_flows), len(fused_flows), len(global_flows))
        for k in range(limit):
            fused = resize_flow(fused_flows[k], final_flows[k].shape[-2:])
            global_flow = resize_flow(global_flows[k], final_flows[k].shape[-2:])
            proxy_mag = torch.norm(global_flow.detach(), dim=1, keepdim=True)
            weight = 1.0 + proxy_mag / proxy_mag.flatten(1).mean(dim=-1, keepdim=True).view(-1, 1, 1, 1).clamp_min(1.0)
            diff = torch.norm(final_flows[k] - fused, dim=1, keepdim=True)
            losses.append((diff * weight).mean())
        if not losses:
            device = final_flows[0].device if final_flows else "cpu"
            return torch.tensor(0.0, device=device)
        return sum(losses) / len(losses)

    def forward(
        self,
        sam_masks: Optional[torch.Tensor],
        model_out: Dict[str, List[torch.Tensor]],
        enable_segment_cycle: bool = True,
        enable_visibility_terms: bool = True,
        enable_residual_terms: bool = True,
        enable_hard_motion_reweight: bool = True,
    ) -> Dict[str, torch.Tensor]:
        labels = self._normalize_labels(sam_masks)
        if labels is None:
            zero = torch.tensor(0.0)
            return {
                "segment_warp": zero,
                "dense_slot_consistency": zero,
                "boundary_residual": zero,
                "segment_cycle": zero,
                "global_fused_consistency": zero,
                "visibility_consistency": zero,
                "occlusion_composite": zero,
                "slot_deformation_reg": zero,
                "hard_motion_reweight": zero,
                "total": zero,
            }

        device = model_out["flows_fw"][0].device if model_out.get("flows_fw") else labels.device
        labels = labels.to(device)

        flows_fw = model_out.get("flows_fw", [])
        residual_fw = model_out.get("residual_flow_fw", [])
        match_probs = model_out.get("match_probs_fw", [])
        match_conf = model_out.get("match_confidence_fw", [])
        long_match = model_out.get("match_probs_long", [])
        slot_flow_fw = model_out.get("slot_flow_fw", [])
        global_flow_fw = model_out.get("global_flow_fw", [])
        fused_coarse_fw = model_out.get("fused_coarse_flow_fw", [])
        dense_prior_fw = model_out.get("dense_prior_flow_fw", [])
        global_conf_fw = model_out.get("global_corr_confidence_fw", [])
        slot_visibility_fw = model_out.get("slot_visibility_fw", [])
        dense_occ_fw = model_out.get("dense_occlusion_fw", [])
        basis_coeff_fw = model_out.get("slot_basis_coeffs_fw", [])

        segment_warp = self._segment_warp_consistency(labels, flows_fw, match_probs, match_conf)
        dense_slot_consistency = self._dense_slot_consistency(labels, slot_flow_fw, dense_prior_fw, match_conf)
        boundary_residual = self._boundary_residual_specialization(labels, residual_fw) if enable_residual_terms else torch.tensor(0.0, device=device)
        segment_cycle = self._segment_cycle(match_probs, long_match) if enable_segment_cycle else torch.tensor(0.0, device=device)
        global_fused_consistency = self._global_fused_consistency(global_flow_fw, fused_coarse_fw, global_conf_fw)
        visibility_consistency = self._visibility_consistency(match_probs, match_conf, slot_visibility_fw) if enable_visibility_terms else torch.tensor(0.0, device=device)
        occlusion_composite = self._occlusion_composite(labels, slot_flow_fw, flows_fw, dense_occ_fw, slot_visibility_fw) if enable_visibility_terms else torch.tensor(0.0, device=device)
        slot_deformation_reg = self._slot_deformation_reg(basis_coeff_fw, match_conf)
        hard_motion_reweight = self._hard_motion_reweight(flows_fw, fused_coarse_fw, global_flow_fw) if enable_hard_motion_reweight else torch.tensor(0.0, device=device)

        total = (
            self.segment_warp_weight * segment_warp
            + self.dense_slot_consistency_weight * dense_slot_consistency
            + self.boundary_residual_weight * boundary_residual
            + self.segment_cycle_weight * segment_cycle
            + self.global_fused_consistency_weight * global_fused_consistency
            + self.visibility_consistency_weight * visibility_consistency
            + self.occlusion_composite_weight * occlusion_composite
            + self.slot_deformation_reg_weight * slot_deformation_reg
            + self.hard_motion_reweight_weight * hard_motion_reweight
        )
        return {
            "segment_warp": segment_warp,
            "dense_slot_consistency": dense_slot_consistency,
            "boundary_residual": boundary_residual,
            "segment_cycle": segment_cycle,
            "global_fused_consistency": global_fused_consistency,
            "visibility_consistency": visibility_consistency,
            "occlusion_composite": occlusion_composite,
            "slot_deformation_reg": slot_deformation_reg,
            "hard_motion_reweight": hard_motion_reweight,
            "total": total,
        }
