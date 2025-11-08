from __future__ import annotations
from typing import Dict, List
import torch
import torch.nn.functional as F

from .utils import image_gradients, warp


class UnsupervisedFlowLoss:
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
        mag = 1.0 / (mag + 1e-3) # higher weight on flat regions, lower near edges
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
            f_bw = -warp(f_fw, f_fw) # crude backward approx
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