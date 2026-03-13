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


class V5ObjectMemoryLossBundle(nn.Module):
    def __init__(
        self,
        num_slots: int = 32,
        segment_warp_weight: float = 0.12,
        piecewise_residual_weight: float = 0.03,
        segment_cycle_weight: float = 0.04,
        layered_order_weight: float = 0.03,
        boundary_residual_weight: float = 0.02,
        dense_slot_consistency_weight: float = 0.05,
    ):
        super().__init__()
        self.num_slots = int(num_slots)
        self.segment_warp_weight = float(segment_warp_weight)
        self.piecewise_residual_weight = float(piecewise_residual_weight)
        self.segment_cycle_weight = float(segment_cycle_weight)
        self.layered_order_weight = float(layered_order_weight)
        self.boundary_residual_weight = float(boundary_residual_weight)
        self.dense_slot_consistency_weight = float(dense_slot_consistency_weight)

    @classmethod
    def from_config(cls, cfg: Dict) -> "V5ObjectMemoryLossBundle":
        model_cfg = cfg.get("model", {})
        loss_cfg = cfg.get("loss", {})
        return cls(
            num_slots=int(model_cfg.get("num_slots", cfg.get("sam", {}).get("num_segments", 32))),
            segment_warp_weight=float(loss_cfg.get("segment_warp", 0.12)),
            piecewise_residual_weight=float(loss_cfg.get("piecewise_residual", 0.03)),
            segment_cycle_weight=float(loss_cfg.get("segment_cycle", 0.04)),
            layered_order_weight=float(loss_cfg.get("layered_order", 0.03)),
            boundary_residual_weight=float(loss_cfg.get("boundary_residual", 0.02)),
            dense_slot_consistency_weight=float(loss_cfg.get("dense_slot_consistency", 0.05)),
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

    def _piecewise_residual(
        self,
        labels: torch.Tensor,
        residual_flows: List[torch.Tensor],
        match_conf: List[torch.Tensor],
    ) -> torch.Tensor:
        losses = []
        for k, residual in enumerate(residual_flows):
            if k >= len(match_conf):
                break
            lbl = labels[:, k]
            h, w = residual.shape[-2:]
            if lbl.shape[-2:] != (h, w):
                lbl = F.interpolate(lbl.unsqueeze(1).float(), size=(h, w), mode="nearest").squeeze(1).long()
            boundary = compute_boundary_map(lbl).float()
            interior = 1.0 - boundary
            conf_map = self._slot_scalar_map(lbl, match_conf[k].clamp(0.0, 1.0))
            mag = torch.norm(residual, dim=1, keepdim=True)
            losses.append((mag * interior * conf_map).mean())
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

    def _layered_order_consistency(
        self,
        match_probs: List[torch.Tensor],
        match_conf: List[torch.Tensor],
        layer_orders: List[torch.Tensor],
        occlusion_slots: List[torch.Tensor],
    ) -> torch.Tensor:
        losses = []
        for k in range(min(len(match_probs), len(layer_orders), len(occlusion_slots))):
            if k + 1 >= len(layer_orders):
                break
            order_a = layer_orders[k]
            order_b = layer_orders[k + 1]
            occ = occlusion_slots[k].clamp(0.0, 1.0)
            matched_order = torch.matmul(match_probs[k], order_b.unsqueeze(-1)).squeeze(-1)
            target = torch.sigmoid(order_a - matched_order)
            conf = match_conf[k].clamp(0.0, 1.0)
            losses.append(F.binary_cross_entropy(occ, target, weight=conf, reduction="sum") / (conf.sum() + 1e-6))
        if not losses:
            device = layer_orders[0].device if layer_orders else "cpu"
            return torch.tensor(0.0, device=device)
        return sum(losses) / len(losses)

    def _boundary_residual_specialization(
        self,
        labels: torch.Tensor,
        residual_flows: List[torch.Tensor],
    ) -> torch.Tensor:
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

    def forward(
        self,
        sam_masks: Optional[torch.Tensor],
        model_out: Dict[str, List[torch.Tensor]],
        enable_segment_cycle: bool = True,
        enable_layered_order: bool = True,
        enable_residual_terms: bool = True,
    ) -> Dict[str, torch.Tensor]:
        labels = self._normalize_labels(sam_masks)
        if labels is None:
            zero = torch.tensor(0.0)
            return {
                "segment_warp": zero,
                "piecewise_residual": zero,
                "segment_cycle": zero,
                "layered_order": zero,
                "boundary_residual": zero,
                "dense_slot_consistency": zero,
                "total": zero,
            }

        device = model_out["flows_fw"][0].device if model_out.get("flows_fw") else labels.device
        labels = labels.to(device)

        flows_fw = model_out.get("flows_fw", [])
        residual_fw = model_out.get("residual_flow_fw", [])
        match_probs = model_out.get("match_probs_fw", [])
        match_conf = model_out.get("match_confidence_fw", [])
        long_match = model_out.get("match_probs_long", [])
        layer_orders = model_out.get("layer_order_all", [])
        occlusion_slots = model_out.get("occlusion_slots_fw", [])
        slot_flow_fw = model_out.get("slot_flow_fw", [])
        dense_prior_fw = model_out.get("dense_prior_flow_fw", [])

        segment_warp = self._segment_warp_consistency(labels, flows_fw, match_probs, match_conf)
        piecewise_residual = self._piecewise_residual(labels, residual_fw, match_conf) if enable_residual_terms else torch.tensor(0.0, device=device)
        segment_cycle = self._segment_cycle(match_probs, long_match) if enable_segment_cycle else torch.tensor(0.0, device=device)
        layered_order = self._layered_order_consistency(match_probs, match_conf, layer_orders, occlusion_slots) if enable_layered_order else torch.tensor(0.0, device=device)
        boundary_residual = self._boundary_residual_specialization(labels, residual_fw) if enable_residual_terms else torch.tensor(0.0, device=device)
        dense_slot_consistency = self._dense_slot_consistency(labels, slot_flow_fw, dense_prior_fw, match_conf) if dense_prior_fw else torch.tensor(0.0, device=device)

        total = (
            self.segment_warp_weight * segment_warp
            + self.piecewise_residual_weight * piecewise_residual
            + self.segment_cycle_weight * segment_cycle
            + self.layered_order_weight * layered_order
            + self.boundary_residual_weight * boundary_residual
            + self.dense_slot_consistency_weight * dense_slot_consistency
        )
        return {
            "segment_warp": segment_warp,
            "piecewise_residual": piecewise_residual,
            "segment_cycle": segment_cycle,
            "layered_order": layered_order,
            "boundary_residual": boundary_residual,
            "dense_slot_consistency": dense_slot_consistency,
            "total": total,
        }
