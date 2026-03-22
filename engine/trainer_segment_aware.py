# file: engine/trainer_segment_aware.py
"""
Segment-Aware Unsupervised Flow Trainer
========================================
Extended trainer for AniFlowFormer-T with segment-aware components.

Features:
- Pluggable loss components (all toggleable)
- SAM-2 segment guidance integration
- AnimeRun-compatible validation metrics (epe, epe_occ, epe_nonocc, epe_flat, epe_line, motion bins)
- Comprehensive TensorBoard logging
- Flow magnitude and segment statistics monitoring
"""
from __future__ import annotations
import copy
import math
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from tqdm import tqdm

# Model imports
# Model imports
from models import AniFlowFormerT, ModelConfig as AFConfig
from models.aniunflow_v4.model import AniFlowFormerTV4, V4Config
from models.aniunflow_v5 import AniFlowFormerTV5, V5Config, V5ObjectMemoryLossBundle
from models.aniunflow_v6 import AniFlowFormerTV6, V6Config, V6GlobalSearchLossBundle
from models.aniunflow.losses import UnsupervisedFlowLoss as AFTLosses

# Segment-aware imports
from losses.segment_aware_losses import SegmentAwareLossModule, build_segment_aware_losses
from models.aniunflow.sam2_guidance import SAM2GuidanceModule, build_sam2_guidance
from utils.warp import flow_warp


# ------------------------------ Utilities ------------------------------ #
def concat_mean(chunks) -> float:
    """Mean of concatenated numpy/tensor arrays; NaN if empty."""
    if not chunks:
        return float("nan")
    arrs = []
    for x in chunks:
        if x is None:
            continue
        if isinstance(x, np.ndarray):
            arrs.append(x)
        elif torch.is_tensor(x):
            arrs.append(x.detach().cpu().numpy())
        else:
            arrs.append(np.asarray(x))
    if not arrs:
        return float("nan")
    cat = np.concatenate(arrs)
    return float(np.mean(cat)) if cat.size else float("nan")


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move batch tensors to device."""
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, (list, tuple)) and v and torch.is_tensor(v[0]):
            out[k] = [t.to(device, non_blocking=True) for t in v]
        else:
            out[k] = v
    return out


def _masked_epe(pred: torch.Tensor, gt: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute AEPE per-batch with optional valid mask."""
    epe_map = torch.norm(pred - gt, dim=1)
    if valid is not None:
        if valid.dim() == 2:
            valid = valid.unsqueeze(0).expand_as(epe_map)
        elif valid.dim() == 3 and valid.size(0) == 1:
            valid = valid.expand_as(epe_map)
        valid = valid.bool()
        num = valid.flatten(1).sum(-1).clamp_min(1)
        epe = (epe_map * valid).flatten(1).sum(-1) / num
    else:
        epe = epe_map.flatten(1).mean(-1)
    return epe


def _flow_to_rgb(flow: torch.Tensor) -> torch.Tensor:
    """Convert flow to RGB visualization."""
    try:
        from utils.flow_viz import flow_to_image as _flow_to_image_np
        imgs = []
        for f in flow.detach().permute(0, 2, 3, 1).cpu().numpy():
            im = _flow_to_image_np(f)
            imgs.append(torch.from_numpy(im).permute(2, 0, 1))
        out = torch.stack(imgs).float() / 255.0
        return out.to(flow.device)
    except Exception:
        # Fallback: HSV visualization
        u, v = flow[:, 0], flow[:, 1]
        mag = torch.sqrt(u * u + v * v)
        ang = torch.atan2(v, u)
        h = (ang + math.pi) / (2 * math.pi)
        m95 = torch.quantile(mag.flatten(1), 0.95, dim=1).view(-1, 1, 1)
        v_ = torch.clamp(mag / (m95 + 1e-6), 0, 1)
        s = torch.ones_like(v_)
        
        # Simple HSV to RGB
        h6 = h * 6.0
        i = h6.long() % 6
        f = h6 - i.float()
        p = v_ * (1 - s)
        q = v_ * (1 - f * s)
        t = v_ * (1 - (1 - f) * s)
        
        rgb = torch.stack([v_, t, p], dim=1)
        return rgb.clamp(0, 1)


def resize_mask(mask: Optional[torch.Tensor], target_hw: Tuple[int, int], device: torch.device) -> Optional[torch.Tensor]:
    """Resize mask to target size."""
    if mask is None:
        return None
    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask)
    mask = mask.to(device)
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    Ht, Wt = target_hw
    if mask.shape[-2:] != (Ht, Wt):
        mask = F.interpolate(mask.unsqueeze(1).float(), size=(Ht, Wt), mode="nearest").squeeze(1)
    return mask.bool()


# ------------------------------ Trainer ------------------------------ #
class SegmentAwareTrainer:
    """
    Segment-Aware Unsupervised Optical Flow Trainer.
    
    Extends the base trainer with:
    - SAM-2 segment guidance
    - Segment-aware loss components
    - AnimeRun-compatible validation metrics
    - Comprehensive logging
    """
    
    def __init__(self, cfg: Dict, workspace: Path):
        self.cfg = cfg
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_metric = float("inf")
        self.global_step = 0
        self.current_epoch = 0
        
        # Build model
        self.model = self._build_model().to(self.device)
        self.model.train()
        
        # Build loss modules.
        # Support both legacy keys (alpha_ssim/w_smooth/...) and V4 keys
        # (photo_ssim_alpha/smooth/consistency/mag_reg/...).
        loss_cfg = cfg.get("loss", {})

        def _loss_get(primary: str, fallback: Optional[str], default: float):
            if not isinstance(loss_cfg, dict):
                return default
            if primary in loss_cfg:
                return loss_cfg[primary]
            if fallback is not None and fallback in loss_cfg:
                return loss_cfg[fallback]
            return default

        self.base_loss = AFTLosses(
            alpha_ssim=float(_loss_get("alpha_ssim", "photo_ssim_alpha", 0.2)),
            w_smooth=float(_loss_get("w_smooth", "smooth", 0.1)),
            w_cons=float(_loss_get("w_cons", "consistency", 0.05)),
            # Anti-collapse regularization
            w_mag_reg=float(_loss_get("w_mag_reg", "mag_reg", 0.01)),
            min_flow_mag=float(_loss_get("min_flow_mag", "min_flow_mag", 0.5)),
            use_photo_gradient=bool(_loss_get("use_photo_gradient", None, True)),
        )

        # Warmup settings for occlusion masking
        self.warmup_steps = int(_loss_get("warmup_steps", None, 0))
        self.disable_occ_during_warmup = bool(
            _loss_get("disable_occ_during_warmup", None, self.warmup_steps > 0)
        )
        # Optional epoch-based occlusion schedule (UnSAMFlow-like)
        self.occ_aware_start_epoch = int(_loss_get("occ_aware_start_epoch", None, 1))

        # Optional epoch-based SAM loss schedule (2-stage style)
        self.sam_loss_start_epoch = int(_loss_get("sam_loss_start_epoch", None, 1))
        self.sam_loss_ramp_epochs = int(_loss_get("sam_loss_ramp_epochs", None, 0))
        self.sam_loss_scale = float(_loss_get("sam_loss_scale", None, 1.0))

        # Optional long-gap unsupervised objective (t -> t+2).
        self.long_gap_photo_weight = float(_loss_get("long_gap_photo_weight", None, 0.0))
        self.long_gap_consistency_weight = float(_loss_get("long_gap_consistency_weight", None, 0.0))
        self.long_gap_start_epoch = int(_loss_get("long_gap_start_epoch", None, 999))
        self.long_gap_ramp_epochs = int(_loss_get("long_gap_ramp_epochs", None, 0))
        self.skip_sam_compute_until_epoch = int(_loss_get("skip_sam_compute_until_epoch", None, 1))
        self.tri_cycle_weight = float(_loss_get("tri_cycle_weight", None, 0.0))
        self.uncertainty_reg_weight = float(_loss_get("uncertainty_reg_weight", None, 0.0))
        self.occlusion_reg_weight = float(_loss_get("occlusion_reg_weight", None, 0.0))
        self.occlusion_prior = float(_loss_get("occlusion_prior", None, 0.15))
        self.tri_cycle_start_epoch = int(_loss_get("tri_cycle_start_epoch", None, 1))
        self.tri_cycle_ramp_epochs = int(_loss_get("tri_cycle_ramp_epochs", None, 0))
        self.uncertainty_reg_start_epoch = int(_loss_get("uncertainty_reg_start_epoch", None, 1))
        self.occlusion_reg_start_epoch = int(_loss_get("occlusion_reg_start_epoch", None, 1))
        
        # Segment-aware losses
        self.use_segment_losses = self._has_segment_losses()
        if self.use_segment_losses:
            self.segment_loss = build_segment_aware_losses(cfg)
        else:
            self.segment_loss = None
        self.v5_loss_bundle: Optional[V5ObjectMemoryLossBundle] = None
        if isinstance(self.model, AniFlowFormerTV5):
            self.v5_loss_bundle = V5ObjectMemoryLossBundle.from_config(cfg)
        self.v6_loss_bundle: Optional[V6GlobalSearchLossBundle] = None
        if isinstance(self.model, AniFlowFormerTV6):
            self.v6_loss_bundle = V6GlobalSearchLossBundle.from_config(cfg)
        
        # SAM-2 guidance module
        sam_cfg = cfg.get("sam", {})
        self.use_sam = sam_cfg.get("enabled", False)
        if self.use_sam:
            self.sam_guidance = build_sam2_guidance(cfg)
        else:
            self.sam_guidance = None
        
        # Optimizer
        optim_cfg = cfg.get("optim", {})
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(optim_cfg.get("lr", 2e-4)),
            weight_decay=float(optim_cfg.get("weight_decay", 1e-4)),
            betas=tuple(optim_cfg.get("betas", [0.9, 0.999])),
        )

        # Optional stage-C style encoder freezing.
        self.freeze_encoder_epoch = int(optim_cfg.get("freeze_encoder_epoch", 0))
        self.freeze_encoder_levels = optim_cfg.get("freeze_encoder_levels", ["lvl1"])
        self._encoder_frozen = False
        
        # Scheduler (configured later)
        self.scheduler = None
        self.sched_per_batch = False
        self._resume_scheduler_state = None
        
        # AMP scaler
        self.scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
        self.clip_grad = float(optim_cfg.get("clip", 1.0))
        self.accum_steps = int(optim_cfg.get("accum_steps", 1))
        
        # Semi-supervised weight
        self.w_epe_sup = float(_loss_get("w_epe_sup", None, 0.0))

        # Online EMA teacher for one-stage self-distillation.
        teacher_cfg = cfg.get("teacher", {})
        self.teacher_enabled = bool(teacher_cfg.get("enabled", False))
        self.teacher_ema_decay = float(teacher_cfg.get("ema_decay", 0.999))
        self.teacher_start_step = int(teacher_cfg.get("start_step", 0))
        self.teacher_distill_weight = float(teacher_cfg.get("distill_weight", 0.0))
        self.teacher_conf_threshold = float(teacher_cfg.get("confidence_threshold", 0.0))
        self.teacher_every_n_steps = max(1, int(teacher_cfg.get("every_n_steps", 1)))
        self.teacher_model: Optional[nn.Module] = None
        if self.teacher_enabled and isinstance(self.model, (AniFlowFormerTV4, AniFlowFormerTV5, AniFlowFormerTV6)):
            self.teacher_model = copy.deepcopy(self.model).to(self.device)
            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad = False
        
        # Logging
        log_cfg = cfg.get("logging", {})
        self.use_tb = log_cfg.get("use_tb", True)
        self.log_every = int(log_cfg.get("log_every", 100))
        self.log_flow_stats = log_cfg.get("log_flow_stats", True)
        self.log_segment_stats = log_cfg.get("log_segment_stats", True)
        self.log_individual_losses = log_cfg.get("log_individual_losses", True)
        
        self.writer: Optional[SummaryWriter] = None
        if self.use_tb:
            tb_dir = self.workspace / log_cfg.get("tb_dir", "tb")
            tb_dir.mkdir(exist_ok=True, parents=True)
            self.writer = SummaryWriter(str(tb_dir))
        
        # Visualization
        viz_cfg = cfg.get("viz", {})
        self.viz_enable = viz_cfg.get("enable", True)
        self.viz_max = int(viz_cfg.get("max_samples", 8))
        self.viz_dir = self.workspace / viz_cfg.get("save_dir", "val_vis")
        self.viz_dir.mkdir(exist_ok=True, parents=True)
        
        # Validation config
        val_cfg = cfg.get("validation", {})
        self.val_every_epochs = val_cfg.get("every_n_epochs", 5)
        self.val_every_steps = val_cfg.get("every_n_steps", None)
        self.early_stop_delta = float(val_cfg.get("early_stop_delta", 0.0))
        self.early_stop_patience = int(val_cfg.get("early_stop_patience", 0))
        self._early_stop_worse_count = 0
        self._prev_val_metric = None

        # Optional data curriculum for stride range.
        data_cfg = cfg.get("data", {})
        self.motion_curriculum = data_cfg.get("motion_curriculum", [])
        runtime_cfg = cfg.get("runtime", {})
        self.runtime_schedule = runtime_cfg.get("schedule", [])
        self._runtime_state: Dict[str, Any] = {
            "stage_idx": None,
            "stride": None,
            "refiner_iters": None,
            "matcher_topk": None,
            "refiner_gradient_checkpointing": None,
            "enable_residual_branch": None,
            "enable_segment_cycle": None,
            "enable_layered_order": None,
            "enable_global_matcher": None,
            "enable_local_refine": None,
            "enable_visibility_composite": None,
            "enable_hard_motion_reweight": None,
        }
        self._runtime_overrides: Dict[str, Optional[bool]] = {
            "enable_occ_aware": None,
            "enable_long_gap": None,
            "enable_sam_losses": None,
            "enable_residual_branch": None,
            "enable_segment_cycle": None,
            "enable_layered_order": None,
            "enable_global_matcher": None,
            "enable_local_refine": None,
            "enable_visibility_composite": None,
            "enable_hard_motion_reweight": None,
        }
        # Global numerical guards against occasional non-finite spikes.
        self.nan_guard_enabled = bool(cfg.get("runtime", {}).get("nan_guard_enabled", True))
        self.nan_guard_flow_clip = float(cfg.get("runtime", {}).get("nan_guard_flow_clip", 5e2))
        self._nan_warn_count = 0

    def _current_v5_deform_scale(self) -> Optional[float]:
        if not isinstance(self.model, AniFlowFormerTV5):
            return None
        if not getattr(self.model, "use_deformable_slots", False):
            return None
        model_cfg = self.cfg.get("model", {})
        max_scale = float(model_cfg.get("slot_basis_scale", 0.0))
        start_epoch = int(model_cfg.get("slot_basis_start_epoch", 1))
        ramp_epochs = int(model_cfg.get("slot_basis_ramp_epochs", 0))
        if self.current_epoch < start_epoch:
            return 0.0
        if ramp_epochs <= 0:
            return max_scale
        progress = (self.current_epoch - start_epoch + 1) / float(ramp_epochs)
        progress = max(0.0, min(1.0, progress))
        return max_scale * progress

    def _current_sam_scale(self) -> float:
        """Epoch-based SAM loss scaling (for 2-stage training schedules)."""
        if self.current_epoch < self.sam_loss_start_epoch:
            return 0.0
        if self.sam_loss_ramp_epochs <= 0:
            return self.sam_loss_scale
        # Linear ramp from start epoch
        p = (self.current_epoch - self.sam_loss_start_epoch + 1) / float(self.sam_loss_ramp_epochs)
        p = max(0.0, min(1.0, p))
        return self.sam_loss_scale * p

    def _current_long_gap_weights(self) -> Tuple[float, float]:
        """Epoch-based long-gap loss scaling."""
        if self.current_epoch < self.long_gap_start_epoch:
            return 0.0, 0.0
        if self.long_gap_ramp_epochs <= 0:
            return self.long_gap_photo_weight, self.long_gap_consistency_weight
        p = (self.current_epoch - self.long_gap_start_epoch + 1) / float(self.long_gap_ramp_epochs)
        p = max(0.0, min(1.0, p))
        return self.long_gap_photo_weight * p, self.long_gap_consistency_weight * p

    def _epoch_ramp(self, start_epoch: int, ramp_epochs: int = 0) -> float:
        if self.current_epoch < start_epoch:
            return 0.0
        if ramp_epochs <= 0:
            return 1.0
        p = (self.current_epoch - start_epoch + 1) / float(ramp_epochs)
        return float(max(0.0, min(1.0, p)))

    @staticmethod
    def _compose_flow(flow_01: torch.Tensor, flow_12: torch.Tensor) -> torch.Tensor:
        """Compose 0->1 and 1->2 into 0->2."""
        if flow_12.shape[-2:] != flow_01.shape[-2:]:
            h0, w0 = flow_12.shape[-2:]
            flow_12 = F.interpolate(flow_12, size=flow_01.shape[-2:], mode="bilinear", align_corners=True)
            sx = flow_01.shape[-1] / max(w0, 1)
            sy = flow_01.shape[-2] / max(h0, 1)
            flow_12[:, 0] *= sx
            flow_12[:, 1] *= sy
        return flow_01 + flow_warp(flow_12, flow_01)

    def _tri_cycle_loss(self, flows_fw: List[torch.Tensor], flows_long: List[torch.Tensor]) -> torch.Tensor:
        """Cycle consistency between direct long flow and composed adjacent flows."""
        if len(flows_fw) < 2 or len(flows_long) == 0:
            return torch.tensor(0.0, device=self.device)
        flow_01 = flows_fw[0]
        flow_12 = flows_fw[1]
        flow_02 = flows_long[0]
        composed_02 = self._compose_flow(flow_01, flow_12)
        if flow_02.shape[-2:] != composed_02.shape[-2:]:
            flow_02 = F.interpolate(flow_02, size=composed_02.shape[-2:], mode="bilinear", align_corners=True)
        return (flow_02 - composed_02).abs().mean()

    def _compute_distill_loss(self, student_out: Dict[str, Any], teacher_out: Dict[str, Any]) -> Optional[torch.Tensor]:
        sf = student_out.get("flows_fw", [])
        tf = teacher_out.get("flows_fw", [])
        if not sf or not tf:
            return None
        n = min(len(sf), len(tf))
        losses = []
        for i in range(n):
            s = sf[i]
            t = tf[i].detach()
            if t.shape[-2:] != s.shape[-2:]:
                t = F.interpolate(t, size=s.shape[-2:], mode="bilinear", align_corners=True)

            conf = None
            tu = teacher_out.get("uncertainty_fw", [])
            to = teacher_out.get("occlusion_fw", [])
            if i < len(tu):
                log_var = tu[i].detach()
                if log_var.shape[-2:] != s.shape[-2:]:
                    log_var = F.interpolate(log_var, size=s.shape[-2:], mode="bilinear", align_corners=False)
                conf = torch.exp(-log_var.clamp(min=-4.0, max=4.0))
            if i < len(to):
                occ = to[i].detach()
                if occ.shape[-2:] != s.shape[-2:]:
                    occ = F.interpolate(occ, size=s.shape[-2:], mode="bilinear", align_corners=False)
                occ = occ.clamp(0.0, 1.0)
                conf = (1.0 - occ) if conf is None else conf * (1.0 - occ)

            if conf is None:
                conf = torch.ones((s.shape[0], 1, s.shape[2], s.shape[3]), device=s.device, dtype=s.dtype)

            if self.teacher_conf_threshold > 0:
                conf = conf * (conf > self.teacher_conf_threshold).float()
            diff = (s - t).abs().mean(dim=1, keepdim=True)
            losses.append((diff * conf).sum() / (conf.sum() + 1e-6))

        if not losses:
            flow_loss = None
        else:
            flow_loss = sum(losses) / len(losses)

        sp = student_out.get("match_probs_fw", [])
        tp = teacher_out.get("match_probs_fw", [])
        prob_losses = []
        for i in range(min(len(sp), len(tp))):
            s_prob = sp[i]
            t_prob = tp[i].detach()
            if t_prob.shape != s_prob.shape:
                continue
            prob_losses.append((s_prob - t_prob).abs().mean())

        if flow_loss is None and not prob_losses:
            return None
        if flow_loss is None:
            return sum(prob_losses) / len(prob_losses)
        if not prob_losses:
            return flow_loss
        return flow_loss + 0.25 * (sum(prob_losses) / len(prob_losses))

    @torch.no_grad()
    def _update_ema_teacher(self):
        if self.teacher_model is None:
            return
        decay = max(0.0, min(0.99999, self.teacher_ema_decay))
        teacher_params = dict(self.teacher_model.named_parameters())
        for name, s_param in self.model.named_parameters():
            t_param = teacher_params.get(name, None)
            if t_param is None or t_param.shape != s_param.shape:
                continue
            t_param.data.mul_(decay).add_(s_param.data, alpha=1.0 - decay)

    @staticmethod
    def _set_matcher_topk(module: Optional[nn.Module], topk: int):
        """Set top-k on all matcher submodules that expose a `topk` attribute."""
        if module is None:
            return
        for sub in module.modules():
            if hasattr(sub, "topk"):
                try:
                    sub.topk = int(topk)
                except Exception:
                    continue

    def _sanitize_tensor(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Replace non-finite values and clamp magnitude to keep training stable."""
        if not torch.is_tensor(x):
            return x, 0
        bad = (~torch.isfinite(x)).sum().item()
        if bad > 0:
            x = torch.nan_to_num(
                x,
                nan=0.0,
                posinf=self.nan_guard_flow_clip,
                neginf=-self.nan_guard_flow_clip,
            )
        x = x.clamp(min=-self.nan_guard_flow_clip, max=self.nan_guard_flow_clip)
        return x, int(bad)

    def _sanitize_tensor_list(self, xs: List[torch.Tensor]) -> Tuple[List[torch.Tensor], int]:
        ys = []
        total_bad = 0
        for x in xs:
            y, bad = self._sanitize_tensor(x)
            ys.append(y)
            total_bad += bad
        return ys, total_bad

    def _current_runtime_stage(self) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
        """Return active runtime stage (index, dict) for current epoch."""
        if not self.runtime_schedule:
            return None, None
        for idx, stage in enumerate(self.runtime_schedule):
            start = int(stage.get("start_epoch", 1))
            end = int(stage.get("end_epoch", 10**9))
            if start <= self.current_epoch <= end:
                return idx, stage
        return None, None

    def _apply_stride_range(self, train_loader: DataLoader, stride_min: int, stride_max: int, source: str):
        ds = self._unwrap_dataset(train_loader.dataset)
        if not hasattr(ds, "set_stride_range"):
            return False
        changed = ds.set_stride_range(int(stride_min), int(stride_max))
        if changed:
            print(
                f"[Trainer] {source} epoch {self.current_epoch}: "
                f"stride {int(stride_min)}..{int(stride_max)} | samples={len(ds)}"
            )
        return changed

    @staticmethod
    def _reset_dataloader_iterator(loader: DataLoader):
        """
        Force DataLoader to recreate workers/iterator on the next __iter__ call.
        This is required when the underlying dataset length changes while
        persistent workers are enabled.
        """
        iterator = getattr(loader, "_iterator", None)
        if iterator is None:
            return
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            shutdown()
        loader._iterator = None

    @staticmethod
    def _estimate_num_batches(num_samples: int, batch_size: int, drop_last: bool) -> int:
        if batch_size <= 0:
            return max(1, num_samples)
        if drop_last:
            return max(1, num_samples // batch_size)
        return max(1, math.ceil(num_samples / float(batch_size)))

    def _estimate_train_batches_for_epoch(self, train_loader: DataLoader, epoch: int) -> int:
        ds = self._unwrap_dataset(train_loader.dataset)
        batch_size = int(getattr(train_loader, "batch_size", 1) or 1)
        drop_last = bool(getattr(train_loader, "drop_last", False))

        stride_pair = None
        for stage in self.runtime_schedule:
            start = int(stage.get("start_epoch", 1))
            end = int(stage.get("end_epoch", 10**9))
            if start <= epoch <= end and "stride_min" in stage and "stride_max" in stage:
                stride_pair = (int(stage["stride_min"]), int(stage["stride_max"]))
                break

        if stride_pair is None:
            for stage in self.motion_curriculum:
                start = int(stage.get("start_epoch", 1))
                end = int(stage.get("end_epoch", 10**9))
                if start <= epoch <= end:
                    stride_pair = (
                        int(stage.get("stride_min", getattr(ds, "smin", 1))),
                        int(stage.get("stride_max", getattr(ds, "smax", 1))),
                    )
                    break

        if stride_pair is not None and hasattr(ds, "count_samples_for_stride_range"):
            num_samples = int(ds.count_samples_for_stride_range(stride_pair[0], stride_pair[1]))
            return self._estimate_num_batches(num_samples, batch_size, drop_last)

        return max(1, len(train_loader))

    def _apply_runtime_schedule(self, train_loader: DataLoader):
        """Apply epoch-based runtime overrides (iters/topk/stride/loss toggles)."""
        stage_idx, stage = self._current_runtime_stage()
        if stage is None:
            self._runtime_overrides = {
                "enable_occ_aware": None,
                "enable_long_gap": None,
                "enable_sam_losses": None,
                "enable_residual_branch": None,
                "enable_segment_cycle": None,
                "enable_layered_order": None,
                "enable_global_matcher": None,
                "enable_local_refine": None,
                "enable_visibility_composite": None,
                "enable_hard_motion_reweight": None,
            }
            return

        self._runtime_overrides = {
            "enable_occ_aware": stage.get("enable_occ_aware", None),
            "enable_long_gap": stage.get("enable_long_gap", None),
            "enable_sam_losses": stage.get("enable_sam_losses", None),
            "enable_residual_branch": stage.get("enable_residual_branch", None),
            "enable_segment_cycle": stage.get("enable_segment_cycle", None),
            "enable_layered_order": stage.get("enable_layered_order", None),
            "enable_global_matcher": stage.get("enable_global_matcher", None),
            "enable_local_refine": stage.get("enable_local_refine", None),
            "enable_visibility_composite": stage.get("enable_visibility_composite", None),
            "enable_hard_motion_reweight": stage.get("enable_hard_motion_reweight", None),
        }

        changed_msgs: List[str] = []

        # Per-stage stride override.
        if "stride_min" in stage and "stride_max" in stage:
            stride_pair = (int(stage["stride_min"]), int(stage["stride_max"]))
            if self._runtime_state.get("stride") != stride_pair:
                self._runtime_state["stride"] = stride_pair
                changed = self._apply_stride_range(
                    train_loader,
                    stride_pair[0],
                    stride_pair[1],
                    "Runtime schedule",
                )
                if changed:
                    self._reset_dataloader_iterator(train_loader)
                changed_msgs.append(f"stride={stride_pair[0]}..{stride_pair[1]}")

        # Refiner iterations override (AniFlowFormerTV4 only).
        refiner_iters = stage.get("refiner_iters", None)
        if refiner_iters is not None and isinstance(self.model, AniFlowFormerTV4):
            refiner_iters = int(refiner_iters)
            if self._runtime_state.get("refiner_iters") != refiner_iters:
                self._runtime_state["refiner_iters"] = refiner_iters
                if hasattr(self.model, "model_cfg"):
                    self.model.model_cfg.refiner_iters = refiner_iters
                if getattr(self.model, "iterative_refiner", None) is not None:
                    self.model.iterative_refiner.iters = refiner_iters
                changed_msgs.append(f"refiner_iters={refiner_iters}")

        # Matcher top-k override.
        matcher_topk = stage.get("matcher_topk", None)
        if matcher_topk is not None and isinstance(self.model, AniFlowFormerTV4):
            matcher_topk = int(matcher_topk)
            if self._runtime_state.get("matcher_topk") != matcher_topk:
                self._runtime_state["matcher_topk"] = matcher_topk
                self._set_matcher_topk(getattr(self.model, "tokenizer", None), matcher_topk)
                self._set_matcher_topk(getattr(self.model, "tokenizer_v45_main", None), matcher_topk)
                self._set_matcher_topk(getattr(self.model, "tokenizer_v45_aux", None), matcher_topk)
                changed_msgs.append(f"matcher_topk={matcher_topk}")

        # Runtime toggle for refiner checkpointing.
        refiner_ckpt = stage.get("refiner_gradient_checkpointing", None)
        if refiner_ckpt is not None and isinstance(self.model, AniFlowFormerTV4):
            refiner_ckpt = bool(refiner_ckpt)
            if self._runtime_state.get("refiner_gradient_checkpointing") != refiner_ckpt:
                self._runtime_state["refiner_gradient_checkpointing"] = refiner_ckpt
                if hasattr(self.model, "model_cfg"):
                    self.model.model_cfg.refiner_gradient_checkpointing = refiner_ckpt
                if getattr(self.model, "iterative_refiner", None) is not None:
                    self.model.iterative_refiner.use_gradient_checkpointing = refiner_ckpt
                changed_msgs.append(f"refiner_ckpt={refiner_ckpt}")

        for key in (
            "enable_residual_branch",
            "enable_segment_cycle",
            "enable_layered_order",
            "enable_global_matcher",
            "enable_local_refine",
            "enable_visibility_composite",
            "enable_hard_motion_reweight",
        ):
            if key in stage:
                value = bool(stage[key])
                if self._runtime_state.get(key) != value:
                    self._runtime_state[key] = value
                    changed_msgs.append(f"{key}={value}")

        if self._runtime_state.get("stage_idx") != stage_idx:
            self._runtime_state["stage_idx"] = stage_idx
            changed_msgs.insert(0, f"stage={stage_idx}")

        if changed_msgs:
            print(f"[Trainer] Runtime schedule epoch {self.current_epoch}: " + ", ".join(changed_msgs))

    @staticmethod
    def _unwrap_dataset(ds):
        """Unwrap common wrappers (e.g., Subset) to access base dataset methods."""
        base = ds
        while hasattr(base, "dataset"):
            base = base.dataset
        return base

    def _apply_motion_curriculum(self, train_loader: DataLoader):
        """Apply epoch-based stride curriculum if dataset supports it."""
        stage_idx, stage = self._current_runtime_stage()
        if stage_idx is not None and stage is not None and "stride_min" in stage and "stride_max" in stage:
            return

        if not self.motion_curriculum:
            return

        target = None
        for stage in self.motion_curriculum:
            start = int(stage.get("start_epoch", 1))
            end = int(stage.get("end_epoch", 10**9))
            if start <= self.current_epoch <= end:
                target = (int(stage.get("stride_min", 1)), int(stage.get("stride_max", 1)))
                break
        if target is None:
            return

        changed = self._apply_stride_range(train_loader, target[0], target[1], "Motion curriculum")
        if changed:
            self._reset_dataloader_iterator(train_loader)
    
    def _build_model(self) -> nn.Module:
        """Build AniFlowFormer-T model from config."""
        mcfg = self.cfg.get("model", {})
        args = dict(mcfg.get("args", {}))
        
        # Check if V3 model should be used
        sam_version = args.pop("sam_version", None)  # Remove from args to avoid V1 error
        model_name = mcfg.get("name", "AniFlowFormerT")
        
        if sam_version == 3 or model_name == "AniFlowFormerTV3":
            # Use V3 model
            from models.aniunflow.model_v3 import AniFlowFormerTV3, ModelConfigV3
            
            # Map config args to V3 config
            sam_guidance_cfg = self.cfg.get("sam_guidance", {})
            sam_cfg = self.cfg.get("sam", {})
            
            v3_config = ModelConfigV3(
                enc_channels=args.get("enc_channels", 64),
                token_dim=args.get("token_dim", 192),
                lcm_depth=args.get("lcm_depth", 6),
                lcm_heads=args.get("lcm_heads", 4),
                gtr_depth=args.get("gtr_depth", 2),
                gtr_heads=args.get("gtr_heads", 4),
                iters_per_level=args.get("iters_per_level", 4),
                use_sam=args.get("use_sam", True),
                sam_version=3,
                use_feature_concat=sam_guidance_cfg.get("feature_concat", True),
                use_attention_bias=sam_guidance_cfg.get("attention_bias", True),
                use_cost_modulation=sam_guidance_cfg.get("cost_modulation", True),
                use_object_pooling=sam_guidance_cfg.get("object_pooling", True),
                num_segments=sam_cfg.get("num_segments", args.get("num_segments", 16)),
            )
            print("[Trainer] Using AniFlowFormerTV3 with SAM guidance")
            return AniFlowFormerTV3(v3_config)
        else:
            backbone_name = str(mcfg.get("backbone", "")).lower()
            is_v5_cfg = (
                model_name == "AniFlowFormerTV5"
                or backbone_name == "v5_object_memory_sam"
                or backbone_name.startswith("v5_1_object_memory")
            )
            if is_v5_cfg:
                print("[Trainer] Initializing AniFlowFormerTV5...")
                config = V5Config.from_dict(self.cfg)
                return AniFlowFormerTV5(config)

            is_v6_cfg = (
                model_name == "AniFlowFormerTV6"
                or backbone_name == "v6_global_slot_search"
                or backbone_name.startswith("v6_")
            )
            if is_v6_cfg:
                print("[Trainer] Initializing AniFlowFormerTV6...")
                config = V6Config.from_dict(self.cfg)
                return AniFlowFormerTV6(config)

            # V4 model detection:
            # - explicit model name, or
            # - flat V4-style model config without nested "args"
            is_v4_flat_cfg = ("args" not in mcfg) and all(
                k in mcfg for k in ("enc_channels", "token_dim", "lcm_depth", "iters_per_level")
            )
            if model_name == "AniFlowFormerTV4" or is_v4_flat_cfg:
                from models.aniunflow_v4.model import AniFlowFormerTV4, V4Config
                print("[Trainer] Initializing AniFlowFormerTV4...")
                config = V4Config.from_dict(self.cfg)
                return AniFlowFormerTV4(config)

            # Use V1/V2 model
            return AniFlowFormerT(AFConfig(**args))
    
    def _has_segment_losses(self) -> bool:
        """Check if any segment-aware losses are enabled."""
        loss_cfg = self.cfg.get("loss", {})
        
        # V4 Config support (flat structure)
        if hasattr(loss_cfg, 'homography_smooth'):
             return (loss_cfg.homography_smooth > 0 or 
                     loss_cfg.boundary_sharpness > 0 or 
                     loss_cfg.object_variance > 0 or
                     loss_cfg.boundary_aware_smooth > 0)
        
        # V3/V1 Config support (nested dicts)
        if isinstance(loss_cfg, dict):
            seg_cons = loss_cfg.get("segment_consistency", {}).get("enabled", False) if isinstance(loss_cfg.get("segment_consistency"), dict) else False
            boundary = loss_cfg.get("boundary_aware_smooth", {}).get("enabled", False) if isinstance(loss_cfg.get("boundary_aware_smooth"), dict) else False
            temporal = loss_cfg.get("temporal_memory", {}).get("enabled", False) if isinstance(loss_cfg.get("temporal_memory"), dict) else False

            # Flat V4 float-style keys
            flat_keys = [
                "homography_smooth",
                "boundary_sharpness",
                "object_variance",
                "segment_consistency",
                "boundary_aware_smooth",
                "temporal_memory",
            ]
            for key in flat_keys:
                v = loss_cfg.get(key, 0.0)
                if isinstance(v, (int, float)) and v > 0:
                    return True

            return seg_cons or boundary or temporal
            
        return False
    
    def _init_scheduler(self, train_loader: DataLoader, epochs: int):
        """Initialize learning rate scheduler."""
        sched_cfg = self.cfg.get("optim", {}).get("scheduler", {})
        sched_type = sched_cfg.get("type", "cosine").lower()
        self.sched_per_batch = sched_cfg.get("per_batch", False)
        
        warmup_epochs = int(self.cfg.get("optim", {}).get("warmup_epochs", 0))
        
        if sched_type == "cosine":
            if self.sched_per_batch:
                T = epochs * max(1, len(train_loader))
            else:
                T = epochs
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=T,
                eta_min=float(sched_cfg.get("min_lr", 1e-6)),
            )
        elif sched_type == "onecycle":
            pct_start = sched_cfg.get("pct_start", None)
            if pct_start is None:
                pct_start = warmup_epochs / epochs if epochs > 0 else 0.05
            pct_start = float(max(0.01, min(0.99, pct_start)))
            total_steps = sum(
                self._estimate_train_batches_for_epoch(train_loader, epoch_idx)
                for epoch_idx in range(1, epochs + 1)
            )
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=float(sched_cfg.get("max_lr", self.cfg["optim"]["lr"])),
                total_steps=max(1, total_steps),
                pct_start=pct_start,
                div_factor=float(sched_cfg.get("div_factor", 25.0)),
                final_div_factor=float(sched_cfg.get("final_div_factor", 1e4)),
                anneal_strategy=str(sched_cfg.get("anneal_strategy", "cos")),
            )
            self.sched_per_batch = True
        else:
            self.scheduler = None

    def _restore_scheduler_state(self):
        """Restore scheduler state if a compatible checkpoint state was loaded."""
        if self.scheduler is None or self._resume_scheduler_state is None:
            return False

        sched_state = self._resume_scheduler_state
        ckpt_total = sched_state.get("total_steps", None) if isinstance(sched_state, dict) else None
        curr_total = getattr(self.scheduler, "total_steps", None)
        if ckpt_total is not None and curr_total is not None and int(ckpt_total) != int(curr_total):
            print(
                "[Trainer] Skipping scheduler state from checkpoint: "
                f"total_steps mismatch ({ckpt_total} vs {curr_total})."
            )
            self._resume_scheduler_state = None
            return False

        try:
            self.scheduler.load_state_dict(sched_state)
            print("[Trainer] Restored scheduler state from checkpoint.")
            restored = True
        except Exception as exc:
            print(f"[Trainer] Skipping scheduler state from checkpoint: {exc}")
            restored = False
        finally:
            self._resume_scheduler_state = None
        return restored

    def _fast_forward_scheduler(self):
        """
        If we resume without a compatible scheduler state, advance a per-batch
        scheduler to the already-completed global step count.
        """
        if self.scheduler is None or not self.sched_per_batch or self.global_step <= 0:
            return
        max_steps = getattr(self.scheduler, "total_steps", None)
        steps_to_advance = int(self.global_step)
        if max_steps is not None:
            steps_to_advance = min(steps_to_advance, max(0, int(max_steps) - 1))
        if steps_to_advance <= 0:
            return
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Detected call of `lr_scheduler.step\(\)` before `optimizer.step\(\)`",
                category=UserWarning,
            )
            for _ in range(steps_to_advance):
                self.scheduler.step()
        print(f"[Trainer] Fast-forwarded scheduler by {steps_to_advance} steps.")

    def _load_module_state(self, module: nn.Module, state_dict: Dict[str, torch.Tensor], module_name: str):
        """
        Load checkpoint weights with compatibility filtering for transient buffers
        such as lazily-created positional encodings.
        """
        current_state = module.state_dict()
        filtered_state = {}
        dropped_keys = []
        shape_mismatch = []

        for key, value in state_dict.items():
            if key not in current_state:
                dropped_keys.append(key)
                continue
            target_value = current_state[key]
            if hasattr(target_value, "shape") and hasattr(value, "shape") and target_value.shape != value.shape:
                shape_mismatch.append((key, tuple(value.shape), tuple(target_value.shape)))
                continue
            filtered_state[key] = value

        missing_keys, unexpected_keys = module.load_state_dict(filtered_state, strict=False)

        if dropped_keys:
            preview = ", ".join(dropped_keys[:5])
            suffix = " ..." if len(dropped_keys) > 5 else ""
            print(
                f"[Trainer] Ignored {len(dropped_keys)} unexpected {module_name} checkpoint keys: "
                f"{preview}{suffix}"
            )
        if shape_mismatch:
            preview = ", ".join(f"{k}:{src}->{dst}" for k, src, dst in shape_mismatch[:3])
            suffix = " ..." if len(shape_mismatch) > 3 else ""
            print(
                f"[Trainer] Ignored {len(shape_mismatch)} shape-mismatched {module_name} keys: "
                f"{preview}{suffix}"
            )
        if missing_keys:
            preview = ", ".join(missing_keys[:5])
            suffix = " ..." if len(missing_keys) > 5 else ""
            print(
                f"[Trainer] Missing {len(missing_keys)} {module_name} keys after checkpoint load: "
                f"{preview}{suffix}"
            )
        if unexpected_keys:
            preview = ", ".join(unexpected_keys[:5])
            suffix = " ..." if len(unexpected_keys) > 5 else ""
            print(
                f"[Trainer] Unexpected {module_name} keys after filtered load: "
                f"{preview}{suffix}"
            )

    def _maybe_freeze_encoder(self):
        """Freeze selected encoder levels after configured epoch."""
        if self._encoder_frozen:
            return
        if self.freeze_encoder_epoch <= 0 or self.current_epoch < self.freeze_encoder_epoch:
            return
        if not hasattr(self.model, "encoder"):
            return

        encoder = getattr(self.model, "encoder")
        frozen_params = 0
        levels = self.freeze_encoder_levels or []

        if not levels:
            for p in encoder.parameters():
                if p.requires_grad:
                    p.requires_grad = False
                    frozen_params += p.numel()
        else:
            for level_name in levels:
                module = getattr(encoder, level_name, None)
                if module is None:
                    continue
                for p in module.parameters():
                    if p.requires_grad:
                        p.requires_grad = False
                        frozen_params += p.numel()

        self._encoder_frozen = frozen_params > 0
        if self._encoder_frozen:
            print(
                f"[Trainer] Froze encoder params at epoch {self.current_epoch}: "
                f"{frozen_params} params ({levels if levels else 'all'})"
            )
    
    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """Main training loop."""
        epochs = int(self.cfg["optim"]["epochs"])
        self._init_scheduler(train_loader, epochs)
        restored_scheduler = self._restore_scheduler_state()
        if not restored_scheduler:
            self._fast_forward_scheduler()

        start_epoch = max(1, int(self.current_epoch) + 1)
        if start_epoch > epochs:
            print(
                f"[Trainer] Checkpoint is already at epoch {self.current_epoch}, "
                f"which is >= configured epochs ({epochs}). Nothing to do."
            )
            return

        for epoch in range(start_epoch, epochs + 1):
            self.current_epoch = epoch
            t0 = time.time()
            self._maybe_freeze_encoder()
            self._apply_runtime_schedule(train_loader)
            self._apply_motion_curriculum(train_loader)
            
            # Train one epoch
            train_metrics = self._train_one_epoch(train_loader)
            
            # Validation
            val_metric = None
            if val_loader is not None and epoch % self.val_every_epochs == 0:
                val_metrics = self.validate(val_loader, epoch=epoch)
                val_metric = val_metrics.get("epe", val_metrics.get("total", float("inf")))
            
            # Per-epoch scheduler step
            if self.scheduler and not self.sched_per_batch:
                self.scheduler.step()
            
            # Logging
            dt = time.time() - t0
            lr = self.optimizer.param_groups[0]["lr"]
            train_loss = train_metrics["loss"] / max(1, train_metrics["steps"])
            nan_skips = int(train_metrics.get("nan_skips", 0))
            if val_metric is None:
                val_metric_text = "SKIP"
            elif np.isfinite(val_metric):
                val_metric_text = f"{val_metric}"
            else:
                val_metric_text = "nan"
            
            print(
                f"[Epoch {epoch:03d}/{epochs}] "
                f"train_loss={train_loss:.4f} "
                f"val_epe={val_metric_text} "
                f"lr={lr:.2e} time={dt:.1f}s "
                f"nan_skips={nan_skips}"
            )
            
            if self.writer:
                self.writer.add_scalar("train/loss_epoch", train_loss, epoch)
                self.writer.add_scalar("train/lr_epoch", lr, epoch)
                self.writer.add_scalar("train/nan_skips_epoch", nan_skips, epoch)
                if val_metric is not None and np.isfinite(val_metric):
                    self.writer.add_scalar("val/epe_epoch", val_metric, epoch)
            
            # Save best
            if val_metric is not None and np.isfinite(val_metric) and val_metric < self.best_metric:
                self.best_metric = val_metric
                self._save_checkpoint("best.pth")

            # Early stop if val metric degrades by > delta for N consecutive validations.
            if val_metric is not None and np.isfinite(val_metric):
                if self._prev_val_metric is not None:
                    if (
                        self.early_stop_patience > 0
                        and val_metric > (self._prev_val_metric + self.early_stop_delta)
                    ):
                        self._early_stop_worse_count += 1
                    else:
                        self._early_stop_worse_count = 0
                self._prev_val_metric = val_metric

                if (
                    self.early_stop_patience > 0
                    and self._early_stop_worse_count >= self.early_stop_patience
                ):
                    print(
                        "[Trainer] Early stop triggered: "
                        f"val metric worsened by > {self.early_stop_delta} for "
                        f"{self._early_stop_worse_count} consecutive validations."
                    )
                    break
            
            # Periodic checkpoint
            ckpt_cfg = self.cfg.get("ckpt", {})
            if epoch % ckpt_cfg.get("save_every", 5) == 0:
                self._save_checkpoint(f"ckpt_e{epoch:03d}.pth")
        
        print(f"\nTraining complete! Best metric: {self.best_metric:.4f}")
    
    def _train_one_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        accum = {"loss": 0.0, "steps": 0, "nan_skips": 0}
        self.optimizer.zero_grad(set_to_none=True)
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        for batch_idx, batch in enumerate(pbar):
            batch = _to_device(batch, self.device)
            clip = batch["clip"]  # (B, T, 3, H, W)
            
            if clip.dtype == torch.uint8:
                clip = clip.float() / 255.0
            
            B, T, C, H, W = clip.shape
            
            # Get SAM masks if enabled
            segment_masks = None
            boundary_maps = None
            sam_features = None
            if self.use_sam and self.sam_guidance is not None:
                # Use pre-loaded masks if available (faster)
                if "sam_masks" in batch:
                    segment_masks = batch["sam_masks"]
                else:
                    # Fallback to online generation (slower)
                    with torch.no_grad():
                        segment_masks = self.sam_guidance.extract_segment_masks(clip)
                # Boundary maps are needed only for non-V4 segment-aware loss path.
                need_boundary_maps = self.use_segment_losses and not isinstance(
                    self.model,
                    (AniFlowFormerTV4, AniFlowFormerTV5, AniFlowFormerTV6),
                )
                if need_boundary_maps and segment_masks is not None:
                    with torch.no_grad():
                        boundary_maps = self.sam_guidance.compute_boundary_maps(segment_masks)

            # Optional precomputed SAM features
            if "sam_features" in batch:
                sam_features = batch["sam_features"]

            with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                # Forward pass
                if isinstance(self.model, AniFlowFormerTV4):
                    sam_scale = self._current_sam_scale()
                    if self._runtime_overrides["enable_sam_losses"] is False:
                        sam_scale = 0.0
                    compute_sam_losses = (
                        sam_scale > 0.0
                        and self.current_epoch >= self.skip_sam_compute_until_epoch
                    )

                    # V4 Model: Integrated forward + loss
                    out_fw = self.model(
                        clip,
                        sam_masks=segment_masks,
                        sam_features=sam_features,
                        return_losses=True,
                        compute_sam_losses=compute_sam_losses,
                    )
                    flows_fw = out_fw["flows_fw"]
                    flows_bw = out_fw["flows_bw"]
                    if self.nan_guard_enabled:
                        flows_fw, bad_fw = self._sanitize_tensor_list(flows_fw)
                        flows_bw, bad_bw = self._sanitize_tensor_list(flows_bw)
                        out_fw["flows_fw"] = flows_fw
                        out_fw["flows_bw"] = flows_bw
                        if out_fw.get("flows_long"):
                            out_fw["flows_long"], bad_long = self._sanitize_tensor_list(out_fw["flows_long"])
                        else:
                            bad_long = 0
                        if out_fw.get("uncertainty_fw"):
                            out_fw["uncertainty_fw"], bad_u = self._sanitize_tensor_list(out_fw["uncertainty_fw"])
                        else:
                            bad_u = 0
                        if out_fw.get("occlusion_fw"):
                            out_fw["occlusion_fw"], bad_o = self._sanitize_tensor_list(out_fw["occlusion_fw"])
                        else:
                            bad_o = 0
                        bad_total = bad_fw + bad_bw + bad_long + bad_u + bad_o
                        if bad_total > 0 and self._nan_warn_count < 20:
                            print(
                                f"[NaNGuard] Sanitized {bad_total} non-finite values "
                                f"at epoch={self.current_epoch} step={self.global_step}"
                            )
                            self._nan_warn_count += 1

                    # Unsupervised photometric loss
                    use_occ_mask = not (
                        self.disable_occ_during_warmup and self.global_step < self.warmup_steps
                    )
                    use_occ_mask = use_occ_mask and (self.current_epoch >= self.occ_aware_start_epoch)
                    if self._runtime_overrides["enable_occ_aware"] is False:
                        use_occ_mask = False
                    if self._runtime_overrides["enable_occ_aware"] is True:
                        use_occ_mask = not (
                            self.disable_occ_during_warmup and self.global_step < self.warmup_steps
                        )

                    long_gap_photo_w, long_gap_cons_w = self._current_long_gap_weights()
                    if self._runtime_overrides["enable_long_gap"] is False:
                        long_gap_photo_w, long_gap_cons_w = 0.0, 0.0
                    loss_dict = self.base_loss.unsup_bidirectional(
                        clip,
                        flows_fw,
                        flows_bw,
                        use_occ_mask=use_occ_mask,
                        long_gap_photo_weight=long_gap_photo_w,
                        long_gap_consistency_weight=long_gap_cons_w,
                    )
                    loss_dict["long_gap_photo_w"] = long_gap_photo_w
                    loss_dict["long_gap_cons_w"] = long_gap_cons_w
                    total_loss = loss_dict["total"]

                    # Add SAM losses from model output
                    if "sam_losses" in out_fw and sam_scale > 0.0:
                        for name, value in out_fw["sam_losses"].items():
                            scaled = value * sam_scale
                            total_loss = total_loss + scaled
                            loss_dict[f"sam_{name}"] = float(scaled.detach())

                    # Tri-frame cycle consistency (direct 0->2 vs composed 0->1->2).
                    tri_w = self.tri_cycle_weight * self._epoch_ramp(
                        self.tri_cycle_start_epoch,
                        self.tri_cycle_ramp_epochs,
                    )
                    if tri_w > 0:
                        tri_cycle = self._tri_cycle_loss(
                            flows_fw=flows_fw,
                            flows_long=out_fw.get("flows_long", []),
                        )
                        total_loss = total_loss + tri_w * tri_cycle
                        loss_dict["tri_cycle"] = float(tri_cycle.detach())
                        loss_dict["tri_cycle_w"] = float(tri_w)

                    # Uncertainty and occlusion regularization to avoid degenerate predictions.
                    ureg_w = (
                        self.uncertainty_reg_weight
                        if self.current_epoch >= self.uncertainty_reg_start_epoch
                        else 0.0
                    )
                    oreg_w = (
                        self.occlusion_reg_weight
                        if self.current_epoch >= self.occlusion_reg_start_epoch
                        else 0.0
                    )
                    if ureg_w > 0 and out_fw.get("uncertainty_fw"):
                        u = torch.cat([x.view(x.shape[0], -1) for x in out_fw["uncertainty_fw"]], dim=1)
                        uncertainty_reg = (u.clamp(min=-4.0, max=4.0) ** 2).mean()
                        total_loss = total_loss + ureg_w * uncertainty_reg
                        loss_dict["uncertainty_reg"] = float(uncertainty_reg.detach())
                        loss_dict["uncertainty_reg_w"] = float(ureg_w)
                    if oreg_w > 0 and out_fw.get("occlusion_fw"):
                        occ = torch.cat([x.view(x.shape[0], -1) for x in out_fw["occlusion_fw"]], dim=1)
                        occ_target = torch.full_like(occ, self.occlusion_prior)
                        # BCE is unsafe under autocast; evaluate this term in fp32.
                        with torch.amp.autocast(device_type="cuda", enabled=False):
                            occ_reg = F.binary_cross_entropy(
                                occ.float().clamp(1e-4, 1.0 - 1e-4),
                                occ_target.float(),
                            )
                        total_loss = total_loss + oreg_w * occ_reg
                        loss_dict["occlusion_reg"] = float(occ_reg.detach())
                        loss_dict["occlusion_reg_w"] = float(oreg_w)

                    # Online EMA-teacher distillation (single-stage).
                    if (
                        self.teacher_model is not None
                        and self.teacher_distill_weight > 0
                        and self.global_step >= self.teacher_start_step
                        and (self.global_step % self.teacher_every_n_steps == 0)
                    ):
                        with torch.no_grad():
                            teacher_out = self.teacher_model(
                                clip,
                                sam_masks=segment_masks,
                                sam_features=sam_features,
                                return_losses=False,
                                compute_sam_losses=False,
                            )
                        distill_loss = self._compute_distill_loss(out_fw, teacher_out)
                        if distill_loss is not None:
                            total_loss = total_loss + self.teacher_distill_weight * distill_loss
                            loss_dict["distill"] = float(distill_loss.detach())

                elif isinstance(self.model, AniFlowFormerTV5):
                    enable_residual_branch = self._runtime_overrides["enable_residual_branch"] is not False
                    deform_scale_override = self._current_v5_deform_scale()
                    out_fw = self.model(
                        clip,
                        sam_masks=segment_masks,
                        sam_features=sam_features,
                        return_losses=True,
                        enable_residual_branch=enable_residual_branch,
                        deform_scale_override=deform_scale_override,
                    )
                    flows_fw = out_fw["flows_fw"]
                    flows_bw = out_fw["flows_bw"]
                    if self.nan_guard_enabled:
                        flows_fw, bad_fw = self._sanitize_tensor_list(flows_fw)
                        flows_bw, bad_bw = self._sanitize_tensor_list(flows_bw)
                        out_fw["flows_fw"] = flows_fw
                        out_fw["flows_bw"] = flows_bw
                        if out_fw.get("flows_long"):
                            out_fw["flows_long"], bad_long = self._sanitize_tensor_list(out_fw["flows_long"])
                        else:
                            bad_long = 0
                        if out_fw.get("residual_flow_fw"):
                            out_fw["residual_flow_fw"], bad_res = self._sanitize_tensor_list(out_fw["residual_flow_fw"])
                        else:
                            bad_res = 0
                        if out_fw.get("dense_prior_flow_fw"):
                            out_fw["dense_prior_flow_fw"], bad_dense = self._sanitize_tensor_list(out_fw["dense_prior_flow_fw"])
                        else:
                            bad_dense = 0
                        bad_total = bad_fw + bad_bw + bad_long + bad_res + bad_dense
                        if bad_total > 0 and self._nan_warn_count < 20:
                            print(
                                f"[NaNGuard] Sanitized {bad_total} non-finite values "
                                f"at epoch={self.current_epoch} step={self.global_step}"
                            )
                            self._nan_warn_count += 1

                    use_occ_mask = not (
                        self.disable_occ_during_warmup and self.global_step < self.warmup_steps
                    )
                    use_occ_mask = use_occ_mask and (self.current_epoch >= self.occ_aware_start_epoch)
                    if self._runtime_overrides["enable_occ_aware"] is False:
                        use_occ_mask = False
                    if self._runtime_overrides["enable_occ_aware"] is True:
                        use_occ_mask = not (
                            self.disable_occ_during_warmup and self.global_step < self.warmup_steps
                        )

                    long_gap_photo_w, long_gap_cons_w = self._current_long_gap_weights()
                    if self._runtime_overrides["enable_long_gap"] is False:
                        long_gap_photo_w, long_gap_cons_w = 0.0, 0.0
                    loss_dict = self.base_loss.unsup_bidirectional(
                        clip,
                        flows_fw,
                        flows_bw,
                        use_occ_mask=use_occ_mask,
                        long_gap_photo_weight=long_gap_photo_w,
                        long_gap_consistency_weight=long_gap_cons_w,
                    )
                    loss_dict["long_gap_photo_w"] = long_gap_photo_w
                    loss_dict["long_gap_cons_w"] = long_gap_cons_w
                    total_loss = loss_dict["total"]

                    if self.v5_loss_bundle is not None and segment_masks is not None:
                        v5_losses = self.v5_loss_bundle(
                            sam_masks=segment_masks,
                            model_out=out_fw,
                            enable_segment_cycle=self._runtime_overrides["enable_segment_cycle"] is not False,
                            enable_layered_order=self._runtime_overrides["enable_layered_order"] is not False,
                            enable_residual_terms=enable_residual_branch,
                        )
                        total_loss = total_loss + v5_losses["total"]
                        loss_dict["v5_segment_warp"] = float(v5_losses["segment_warp"].detach())
                        loss_dict["v5_piecewise_residual"] = float(v5_losses["piecewise_residual"].detach())
                        loss_dict["v5_segment_cycle"] = float(v5_losses["segment_cycle"].detach())
                        loss_dict["v5_layered_order"] = float(v5_losses["layered_order"].detach())
                        loss_dict["v5_boundary_residual"] = float(v5_losses["boundary_residual"].detach())
                        loss_dict["v5_dense_slot_consistency"] = float(v5_losses["dense_slot_consistency"].detach())
                        loss_dict["v5_global_dense_consistency"] = float(v5_losses["global_dense_consistency"].detach())
                        loss_dict["v5_total"] = float(v5_losses["total"].detach())

                    global_photo_w = float(self.cfg.get("loss", {}).get("global_photo", 0.0))
                    if (
                        global_photo_w > 0.0
                        and out_fw.get("global_flow_fw")
                        and out_fw.get("global_flow_bw")
                    ):
                        global_loss = self.base_loss.unsup_bidirectional(
                            clip,
                            out_fw["global_flow_fw"],
                            out_fw["global_flow_bw"],
                            use_occ_mask=use_occ_mask,
                            long_gap_photo_weight=0.0,
                            long_gap_consistency_weight=0.0,
                        )
                        total_loss = total_loss + global_photo_w * global_loss["total"]
                        loss_dict["v5_global_photo"] = float(global_loss["total"].detach())
                        loss_dict["v5_global_photo_w"] = float(global_photo_w)

                    slot_photo_w = float(self.cfg.get("loss", {}).get("slot_photo", 0.0))
                    if (
                        slot_photo_w > 0.0
                        and out_fw.get("slot_flow_fw")
                        and out_fw.get("slot_flow_bw")
                    ):
                        slot_loss = self.base_loss.unsup_bidirectional(
                            clip,
                            out_fw["slot_flow_fw"],
                            out_fw["slot_flow_bw"],
                            use_occ_mask=use_occ_mask,
                            long_gap_photo_weight=0.0,
                            long_gap_consistency_weight=0.0,
                        )
                        total_loss = total_loss + slot_photo_w * slot_loss["total"]
                        loss_dict["v5_slot_photo"] = float(slot_loss["total"].detach())
                        loss_dict["v5_slot_photo_w"] = float(slot_photo_w)

                    if (
                        self.teacher_model is not None
                        and self.teacher_distill_weight > 0
                        and self.global_step >= self.teacher_start_step
                        and (self.global_step % self.teacher_every_n_steps == 0)
                    ):
                        with torch.no_grad():
                            teacher_out = self.teacher_model(
                                clip,
                                sam_masks=segment_masks,
                                sam_features=sam_features,
                                return_losses=False,
                                enable_residual_branch=enable_residual_branch,
                                deform_scale_override=deform_scale_override,
                            )
                        distill_loss = self._compute_distill_loss(out_fw, teacher_out)
                        if distill_loss is not None:
                            total_loss = total_loss + self.teacher_distill_weight * distill_loss
                            loss_dict["distill"] = float(distill_loss.detach())

                elif isinstance(self.model, AniFlowFormerTV6):
                    enable_residual_branch = self._runtime_overrides["enable_residual_branch"] is not False
                    enable_global_matcher = self._runtime_overrides["enable_global_matcher"] is not False
                    enable_local_refine = self._runtime_overrides["enable_local_refine"] is not False
                    enable_visibility_composite = self._runtime_overrides["enable_visibility_composite"] is not False
                    out_fw = self.model(
                        clip,
                        sam_masks=segment_masks,
                        sam_features=sam_features,
                        return_losses=True,
                        enable_global_matcher=enable_global_matcher,
                        enable_local_refine=enable_local_refine,
                        enable_visibility_composite=enable_visibility_composite,
                        enable_residual_branch=enable_residual_branch,
                    )
                    flows_fw = out_fw["flows_fw"]
                    flows_bw = out_fw["flows_bw"]
                    if self.nan_guard_enabled:
                        flows_fw, bad_fw = self._sanitize_tensor_list(flows_fw)
                        flows_bw, bad_bw = self._sanitize_tensor_list(flows_bw)
                        out_fw["flows_fw"] = flows_fw
                        out_fw["flows_bw"] = flows_bw
                        bad_total = bad_fw + bad_bw
                        for key in (
                            "flows_long",
                            "slot_flow_fw",
                            "global_flow_fw",
                            "fused_coarse_flow_fw",
                            "dense_prior_flow_fw",
                            "residual_flow_fw",
                        ):
                            if out_fw.get(key):
                                out_fw[key], bad = self._sanitize_tensor_list(out_fw[key])
                                bad_total += bad
                        if bad_total > 0 and self._nan_warn_count < 20:
                            print(
                                f"[NaNGuard] Sanitized {bad_total} non-finite values "
                                f"at epoch={self.current_epoch} step={self.global_step}"
                            )
                            self._nan_warn_count += 1

                    use_occ_mask = not (
                        self.disable_occ_during_warmup and self.global_step < self.warmup_steps
                    )
                    use_occ_mask = use_occ_mask and (self.current_epoch >= self.occ_aware_start_epoch)
                    if self._runtime_overrides["enable_occ_aware"] is False:
                        use_occ_mask = False
                    if self._runtime_overrides["enable_occ_aware"] is True:
                        use_occ_mask = not (
                            self.disable_occ_during_warmup and self.global_step < self.warmup_steps
                        )

                    long_gap_photo_w, long_gap_cons_w = self._current_long_gap_weights()
                    if self._runtime_overrides["enable_long_gap"] is False:
                        long_gap_photo_w, long_gap_cons_w = 0.0, 0.0
                    loss_dict = self.base_loss.unsup_bidirectional(
                        clip,
                        flows_fw,
                        flows_bw,
                        use_occ_mask=use_occ_mask,
                        long_gap_photo_weight=long_gap_photo_w,
                        long_gap_consistency_weight=long_gap_cons_w,
                    )
                    loss_dict["long_gap_photo_w"] = long_gap_photo_w
                    loss_dict["long_gap_cons_w"] = long_gap_cons_w
                    total_loss = loss_dict["total"]

                    if self.v6_loss_bundle is not None and segment_masks is not None:
                        v6_losses = self.v6_loss_bundle(
                            sam_masks=segment_masks,
                            model_out=out_fw,
                            enable_segment_cycle=self._runtime_overrides["enable_segment_cycle"] is not False,
                            enable_visibility_terms=self._runtime_overrides["enable_visibility_composite"] is not False,
                            enable_residual_terms=enable_residual_branch,
                            enable_hard_motion_reweight=self._runtime_overrides["enable_hard_motion_reweight"] is not False,
                        )
                        total_loss = total_loss + v6_losses["total"]
                        for key, value in v6_losses.items():
                            loss_dict[f"v6_{key}"] = float(value.detach())

                    global_photo_w = float(self.cfg.get("loss", {}).get("global_photo", 0.0))
                    if global_photo_w > 0.0 and out_fw.get("global_flow_fw") and out_fw.get("global_flow_bw"):
                        global_loss = self.base_loss.unsup_bidirectional(
                            clip,
                            out_fw["global_flow_fw"],
                            out_fw["global_flow_bw"],
                            use_occ_mask=use_occ_mask,
                            long_gap_photo_weight=0.0,
                            long_gap_consistency_weight=0.0,
                        )
                        total_loss = total_loss + global_photo_w * global_loss["total"]
                        loss_dict["v6_global_photo"] = float(global_loss["total"].detach())
                        loss_dict["v6_global_photo_w"] = float(global_photo_w)

                    slot_photo_w = float(self.cfg.get("loss", {}).get("slot_photo", 0.0))
                    if slot_photo_w > 0.0 and out_fw.get("slot_flow_fw") and out_fw.get("slot_flow_bw"):
                        slot_loss = self.base_loss.unsup_bidirectional(
                            clip,
                            out_fw["slot_flow_fw"],
                            out_fw["slot_flow_bw"],
                            use_occ_mask=use_occ_mask,
                            long_gap_photo_weight=0.0,
                            long_gap_consistency_weight=0.0,
                        )
                        total_loss = total_loss + slot_photo_w * slot_loss["total"]
                        loss_dict["v6_slot_photo"] = float(slot_loss["total"].detach())
                        loss_dict["v6_slot_photo_w"] = float(slot_photo_w)

                    if (
                        self.teacher_model is not None
                        and self.teacher_distill_weight > 0
                        and self.global_step >= self.teacher_start_step
                        and (self.global_step % self.teacher_every_n_steps == 0)
                    ):
                        with torch.no_grad():
                            teacher_out = self.teacher_model(
                                clip,
                                sam_masks=segment_masks,
                                sam_features=sam_features,
                                return_losses=False,
                                enable_global_matcher=enable_global_matcher,
                                enable_local_refine=enable_local_refine,
                                enable_visibility_composite=enable_visibility_composite,
                                enable_residual_branch=enable_residual_branch,
                            )
                        distill_loss = self._compute_distill_loss(out_fw, teacher_out)
                        if distill_loss is not None:
                            total_loss = total_loss + self.teacher_distill_weight * distill_loss
                            loss_dict["distill"] = float(distill_loss.detach())

                else:
                    # V1/V3 Model: Separate forward and loss
                    out_fw = self.model(clip, sam_masks=segment_masks)
                    flows_fw = out_fw["flows"]

                    # Backward pass (reverse clip)
                    clip_rev = torch.flip(clip, dims=[1])
                    segment_masks_rev = torch.flip(segment_masks, dims=[1]) if segment_masks is not None else None
                    out_bw = self.model(clip_rev, sam_masks=segment_masks_rev)
                    flows_bw_rev = out_bw["flows"]

                    # Re-index backward flows
                    flows_bw = [flows_bw_rev[len(flows_bw_rev) - 1 - k] for k in range(len(flows_fw))]
                    if self.nan_guard_enabled:
                        flows_fw, bad_fw = self._sanitize_tensor_list(flows_fw)
                        flows_bw, bad_bw = self._sanitize_tensor_list(flows_bw)
                        bad_total = bad_fw + bad_bw
                        if bad_total > 0 and self._nan_warn_count < 20:
                            print(
                                f"[NaNGuard] Sanitized {bad_total} non-finite values "
                                f"at epoch={self.current_epoch} step={self.global_step}"
                            )
                            self._nan_warn_count += 1

                    use_occ_mask = not (
                        self.disable_occ_during_warmup and self.global_step < self.warmup_steps
                    )
                    use_occ_mask = use_occ_mask and (self.current_epoch >= self.occ_aware_start_epoch)
                    if self._runtime_overrides["enable_occ_aware"] is False:
                        use_occ_mask = False
                    if self._runtime_overrides["enable_occ_aware"] is True:
                        use_occ_mask = not (
                            self.disable_occ_during_warmup and self.global_step < self.warmup_steps
                        )
                    long_gap_photo_w, long_gap_cons_w = self._current_long_gap_weights()
                    if self._runtime_overrides["enable_long_gap"] is False:
                        long_gap_photo_w, long_gap_cons_w = 0.0, 0.0
                    loss_dict = self.base_loss.unsup_bidirectional(
                        clip,
                        flows_fw,
                        flows_bw,
                        use_occ_mask=use_occ_mask,
                        long_gap_photo_weight=long_gap_photo_w,
                        long_gap_consistency_weight=long_gap_cons_w,
                    )
                    loss_dict["long_gap_photo_w"] = long_gap_photo_w
                    loss_dict["long_gap_cons_w"] = long_gap_cons_w
                    total_loss = loss_dict["total"]

                    # Add segment-aware losses (V3 style)
                    if self.use_segment_losses and segment_masks is not None:
                        seg_losses = self.segment_loss(flows_fw, clip, segment_masks, boundary_maps)
                        total_loss = total_loss + seg_losses["total_segment_loss"]
                        loss_dict.update(seg_losses)

                # Optional semi-supervised loss (Common)
                if self.w_epe_sup > 0 and "flow" in batch:
                    gt = batch["flow"]
                    if len(flows_fw) > 0:
                        pred = flows_fw[0]
                        if pred.shape[-2:] != gt.shape[-2:]:
                            pred = F.interpolate(pred, size=gt.shape[-2:], mode="bilinear", align_corners=True)
                        epe = _masked_epe(pred, gt).mean()
                        total_loss = total_loss + self.w_epe_sup * epe
                        loss_dict["epe_sup"] = epe.item()

                if not torch.isfinite(total_loss):
                    if self._nan_warn_count < 20:
                        print(
                            f"[NaNGuard] Non-finite total_loss at epoch={self.current_epoch} "
                            f"step={self.global_step}. Skipping batch."
                        )
                        self._nan_warn_count += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    accum["nan_skips"] += 1
                    continue
            
            # Backward
            loss_scaled = total_loss / self.accum_steps
            self.scaler.scale(loss_scaled).backward()
            
            # Optimizer step
            if (batch_idx + 1) % self.accum_steps == 0:
                if self.clip_grad > 0:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                    if not torch.isfinite(grad_norm):
                        if self._nan_warn_count < 20:
                            print(
                                f"[NaNGuard] Non-finite gradient norm at epoch={self.current_epoch} "
                                f"step={self.global_step}. Skipping optimizer step."
                            )
                            self._nan_warn_count += 1
                        self.optimizer.zero_grad(set_to_none=True)
                        self.scaler.update()
                        accum["nan_skips"] += 1
                        continue
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.teacher_model is not None:
                    self._update_ema_teacher()
                
                if self.sched_per_batch and self.scheduler:
                    self.scheduler.step()
            
            self.global_step += 1
            accum["loss"] += float(total_loss.detach())
            accum["steps"] += 1
            
            pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})
            
            # Logging
            if self.writer and self.global_step % self.log_every == 0:
                self._log_training_step(clip, flows_fw, loss_dict, segment_masks, model_out=out_fw)
        
        return accum
    
    def _log_training_step(
        self,
        clip: torch.Tensor,
        flows: List[torch.Tensor],
        loss_dict: Dict[str, Any],
        segment_masks: Optional[torch.Tensor] = None,
        model_out: Optional[Dict[str, Any]] = None,
    ):
        """Log training metrics and visualizations."""
        step = self.global_step
        
        # Log individual losses
        if self.log_individual_losses:
            for k, v in loss_dict.items():
                if torch.is_tensor(v):
                    v = v.detach().item()
                self.writer.add_scalar(f"train_loss/{k}", v, step)
        
        # Log learning rate
        self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], step)
        
        # Flow magnitude statistics
        if self.log_flow_stats and flows:
            flow = flows[0].detach()
            mag = (flow[:, 0] ** 2 + flow[:, 1] ** 2).sqrt()
            self.writer.add_scalar("train_flow/mag_mean", mag.mean().item(), step)
            self.writer.add_scalar("train_flow/mag_std", mag.std().item(), step)
            self.writer.add_scalar("train_flow/mag_max", mag.max().item(), step)
            if model_out is not None:
                if model_out.get("slot_flow_fw"):
                    slot = model_out["slot_flow_fw"][0].detach()
                    slot_mag = (slot[:, 0] ** 2 + slot[:, 1] ** 2).sqrt()
                    debug_branch = model_out.get("debug_branch", "")
                    if debug_branch == "v5_3b":
                        tag_prefix = "train_v5_3b"
                    elif debug_branch == "v5_3":
                        tag_prefix = "train_v5_3"
                    elif debug_branch == "v5_2":
                        tag_prefix = "train_v5_2"
                    elif model_out.get("global_flow_fw") and model_out.get("local_corr_confidence_fw"):
                        tag_prefix = "train_v6"
                    else:
                        tag_prefix = "train_v5"
                    self.writer.add_scalar(f"{tag_prefix}/slot_mag_mean", slot_mag.mean().item(), step)
                if model_out.get("dense_prior_flow_fw"):
                    dense_prior = model_out["dense_prior_flow_fw"][0].detach()
                    dense_prior_mag = (dense_prior[:, 0] ** 2 + dense_prior[:, 1] ** 2).sqrt()
                    debug_branch = model_out.get("debug_branch", "")
                    if debug_branch == "v5_3b":
                        tag_prefix = "train_v5_3b"
                    elif debug_branch == "v5_3":
                        tag_prefix = "train_v5_3"
                    elif debug_branch == "v5_2":
                        tag_prefix = "train_v5_2"
                    elif model_out.get("global_flow_fw") and model_out.get("local_corr_confidence_fw"):
                        tag_prefix = "train_v6"
                    else:
                        tag_prefix = "train_v5"
                    self.writer.add_scalar(f"{tag_prefix}/dense_prior_mag_mean", dense_prior_mag.mean().item(), step)
                if model_out.get("residual_flow_fw"):
                    residual = model_out["residual_flow_fw"][0].detach()
                    residual_mag = (residual[:, 0] ** 2 + residual[:, 1] ** 2).sqrt()
                    debug_branch = model_out.get("debug_branch", "")
                    if debug_branch == "v5_3b":
                        tag_prefix = "train_v5_3b"
                    elif debug_branch == "v5_3":
                        tag_prefix = "train_v5_3"
                    elif debug_branch == "v5_2":
                        tag_prefix = "train_v5_2"
                    elif model_out.get("global_flow_fw") and model_out.get("local_corr_confidence_fw"):
                        tag_prefix = "train_v6"
                    else:
                        tag_prefix = "train_v5"
                    self.writer.add_scalar(f"{tag_prefix}/residual_mag_mean", residual_mag.mean().item(), step)
                if model_out.get("corr_confidence_fw"):
                    corr_conf = model_out["corr_confidence_fw"][0].detach()
                    self.writer.add_scalar("train_v5/corr_conf_mean", corr_conf.mean().item(), step)
                if model_out.get("match_confidence_fw"):
                    match_conf = model_out["match_confidence_fw"][0].detach()
                    debug_branch = model_out.get("debug_branch", "")
                    if debug_branch == "v5_3b":
                        tag_prefix = "train_v5_3b"
                    elif debug_branch == "v5_3":
                        tag_prefix = "train_v5_3"
                    elif debug_branch == "v5_2":
                        tag_prefix = "train_v5_2"
                    elif model_out.get("global_flow_fw") and model_out.get("local_corr_confidence_fw"):
                        tag_prefix = "train_v6"
                    else:
                        tag_prefix = "train_v5"
                    self.writer.add_scalar(f"{tag_prefix}/match_conf_mean", match_conf.mean().item(), step)
                if model_out.get("global_flow_fw"):
                    global_flow = model_out["global_flow_fw"][0].detach()
                    global_mag = (global_flow[:, 0] ** 2 + global_flow[:, 1] ** 2).sqrt()
                    debug_branch = model_out.get("debug_branch", "")
                    tag_prefix = "train_v5_3b" if debug_branch == "v5_3b" else ("train_v5_3" if debug_branch == "v5_3" else ("train_v5_2" if debug_branch == "v5_2" else "train_v6"))
                    self.writer.add_scalar(f"{tag_prefix}/global_mag_mean", global_mag.mean().item(), step)
                if model_out.get("fused_coarse_flow_fw"):
                    fused = model_out["fused_coarse_flow_fw"][0].detach()
                    fused_mag = (fused[:, 0] ** 2 + fused[:, 1] ** 2).sqrt()
                    debug_branch = model_out.get("debug_branch", "")
                    tag_prefix = "train_v5_3b" if debug_branch == "v5_3b" else ("train_v5_3" if debug_branch == "v5_3" else ("train_v5_2" if debug_branch == "v5_2" else "train_v6"))
                    self.writer.add_scalar(f"{tag_prefix}/fused_coarse_mag_mean", fused_mag.mean().item(), step)
                if model_out.get("global_corr_confidence_fw"):
                    global_conf = model_out["global_corr_confidence_fw"][0].detach()
                    debug_branch = model_out.get("debug_branch", "")
                    tag_prefix = "train_v5_3b" if debug_branch == "v5_3b" else ("train_v5_3" if debug_branch == "v5_3" else ("train_v5_2" if debug_branch == "v5_2" else "train_v6"))
                    self.writer.add_scalar(f"{tag_prefix}/global_conf_mean", global_conf.mean().item(), step)
                if model_out.get("slot_basis_coeffs_fw"):
                    coeff = model_out["slot_basis_coeffs_fw"][0].detach()
                    if coeff.numel() > 0:
                        coeff_energy = coeff.pow(2).mean().item()
                        debug_branch = model_out.get("debug_branch", "")
                        tag_prefix = "train_v5_3b" if debug_branch == "v5_3b" else ("train_v5_3" if debug_branch == "v5_3" else "train_v5")
                        self.writer.add_scalar(f"{tag_prefix}/slot_basis_energy", coeff_energy, step)
                if model_out.get("deform_basis_scale_fw"):
                    deform_scale = model_out["deform_basis_scale_fw"][0].detach().mean().item()
                    debug_branch = model_out.get("debug_branch", "")
                    if debug_branch in {"v5_3", "v5_3b"}:
                        tag_prefix = "train_v5_3b" if debug_branch == "v5_3b" else "train_v5_3"
                        self.writer.add_scalar(f"{tag_prefix}/deform_basis_scale", deform_scale, step)
                if model_out.get("temporal_support_fw"):
                    temporal_support = model_out["temporal_support_fw"][0].detach()
                    debug_branch = model_out.get("debug_branch", "")
                    if debug_branch in {"v5_3", "v5_3b", "v5_2", "v5_1", "v5"}:
                        tag_prefix = "train_v5_3b" if debug_branch == "v5_3b" else ("train_v5_3" if debug_branch == "v5_3" else ("train_v5_2" if debug_branch == "v5_2" else "train_v5"))
                        self.writer.add_scalar(f"{tag_prefix}/temporal_support_mean", temporal_support.mean().item(), step)
                if model_out.get("local_corr_confidence_fw"):
                    local_conf = model_out["local_corr_confidence_fw"][0].detach()
                    self.writer.add_scalar("train_v6/local_conf_mean", local_conf.mean().item(), step)
                    self.writer.add_scalar("train_v6/local_refine_mag_mean", dense_prior_mag.mean().item(), step)
                if model_out.get("slot_visibility_fw"):
                    vis = model_out["slot_visibility_fw"][0].detach()
                    self.writer.add_scalar("train_v6/visible_slot_ratio", vis.mean().item(), step)
                if model_out.get("dense_occlusion_fw"):
                    occ = model_out["dense_occlusion_fw"][0].detach()
                    self.writer.add_scalar("train_v6/occlusion_ratio", occ.mean().item(), step)
        
        # Segment statistics
        if self.log_segment_stats and segment_masks is not None:
            mask = segment_masks.detach()
            # Count actual unique labels (excluding background=0)
            num_segments = len(torch.unique(mask)) - 1  # subtract background
            num_segments = max(num_segments, 0)
            avg_segment_size = (mask > 0.5).float().sum(dim=(-2, -1)).mean().item()
            self.writer.add_scalar("train_segment/num_segments", num_segments, step)
            self.writer.add_scalar("train_segment/avg_size", avg_segment_size, step)
        
        # Visualization
        if self.viz_enable:
            n = min(self.viz_max, clip.shape[0])
            img0 = clip[:n, 0]
            img1 = clip[:n, 1]
            
            grid_img0 = vutils.make_grid(img0, nrow=4, normalize=True)
            grid_img1 = vutils.make_grid(img1, nrow=4, normalize=True)
            self.writer.add_image("train/img0", grid_img0, step)
            self.writer.add_image("train/img1", grid_img1, step)
            
            if flows:
                flow_rgb = _flow_to_rgb(flows[0][:n].clamp(-500, 500))
                grid_flow = vutils.make_grid(flow_rgb, nrow=4)
                self.writer.add_image("train/flow_rgb", grid_flow, step)
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader, epoch: int = 0) -> Dict[str, float]:
        """
        Validate with AnimeRun-compatible metrics.
        
        Metrics:
        - epe: Overall AEPE
        - epe_occ: AEPE on occluded regions
        - epe_nonocc: AEPE on non-occluded regions
        - epe_flat: AEPE on flat regions
        - epe_line: AEPE near contour lines
        - epe_s<10, epe_s10-50, epe_s>50: AEPE by motion magnitude
        - 1px, 3px, 5px: Accuracy thresholds
        """
        self.model.eval()
        device = self.device
        
        # Accumulators for metrics
        epe_all_list = []
        epe_occ_list = []
        epe_nonocc_list = []
        epe_flat_list = []
        epe_line_list = []
        epe_s10_list = []
        epe_s1050_list = []
        epe_s50_list = []
        
        thr1_cnt = thr3_cnt = thr5_cnt = 0
        valid_total = 0
        
        proxy_losses = []
        did_viz = False
        
        for batch in tqdm(val_loader, desc="Validate"):
            batch = _to_device(batch, device)
            clip = batch["clip"]
            
            if clip.dtype == torch.uint8:
                clip = clip.float() / 255.0
            
            B, T, C, H, W = clip.shape
            
            # Get segment masks if SAM enabled
            segment_masks = None
            if self.use_sam and self.sam_guidance is not None:
                # Prefer preloaded masks from dataset for speed/consistency.
                if "sam_masks" in batch:
                    segment_masks = batch["sam_masks"]
                else:
                    segment_masks = self.sam_guidance.extract_segment_masks(clip)
            
            with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                out = self.model(clip, sam_masks=segment_masks)
            
            flows = out["flows"]
            if self.nan_guard_enabled:
                flows, bad = self._sanitize_tensor_list(flows)
                if bad > 0 and self._nan_warn_count < 20:
                    print(
                        f"[NaNGuard][Val] Sanitized {bad} non-finite values "
                        f"at epoch={epoch} step={self.global_step}"
                    )
                    self._nan_warn_count += 1
            
            # Get GT if available
            gt_any = batch.get("flow", batch.get("flow_list", None))
            valid_any = batch.get("valid", None)
            occ_any = batch.get("occ", None)
            line_any = batch.get("line", None)
            
            if gt_any is not None:
                gt_list = [gt_any] if torch.is_tensor(gt_any) else gt_any
                L = min(len(flows), len(gt_list))
                
                for k in range(L):
                    pred = flows[k]
                    gt = gt_list[k]
                    
                    # Resize prediction to GT size
                    if pred.shape[-2:] != gt.shape[-2:]:
                        hp, wp = pred.shape[-2:]
                        hg, wg = gt.shape[-2:]
                        pred = F.interpolate(pred, size=(hg, wg), mode="bilinear", align_corners=True)
                        pred[:, 0] *= wg / float(wp)
                        pred[:, 1] *= hg / float(hp)
                    
                    finite_pix = torch.isfinite(pred).all(dim=1) & torch.isfinite(gt).all(dim=1)
                    epe_map = torch.norm(pred - gt, dim=1)
                    mag = torch.norm(gt, dim=1)
                    
                    # Valid mask
                    if valid_any is None:
                        valid = torch.ones_like(epe_map, dtype=torch.bool)
                    else:
                        valid = resize_mask(valid_any, gt.shape[-2:], device)
                        if valid.dim() == 3 and valid.shape[0] == 1:
                            valid = valid.expand(B, -1, -1)
                    valid = valid & finite_pix
                    if valid.sum().item() == 0:
                        continue
                    
                    # Overall EPE
                    epe_all_list.append(epe_map[valid].cpu().numpy())
                    
                    valid_cnt = valid.sum().item()
                    valid_total += valid_cnt
                    
                    # Thresholds
                    thr1_cnt += ((epe_map < 1.0) & valid).sum().item()
                    thr3_cnt += ((epe_map < 3.0) & valid).sum().item()
                    thr5_cnt += ((epe_map < 5.0) & valid).sum().item()
                    
                    # Occlusion breakdown
                    if occ_any is not None:
                        occ = resize_mask(occ_any, gt.shape[-2:], device)
                        epe_occ = epe_map[(occ == 0) & valid]
                        epe_nonocc = epe_map[(occ == 1) & valid]
                        if epe_occ.numel() > 0:
                            epe_occ_list.append(epe_occ.cpu().numpy())
                        if epe_nonocc.numel() > 0:
                            epe_nonocc_list.append(epe_nonocc.cpu().numpy())
                    
                    # Flat/line breakdown
                    if line_any is not None:
                        line = resize_mask(line_any, gt.shape[-2:], device)
                        epe_flat = epe_map[(line > 0) & valid]
                        epe_line = epe_map[(line == 0) & valid]
                        if epe_flat.numel() > 0:
                            epe_flat_list.append(epe_flat.cpu().numpy())
                        if epe_line.numel() > 0:
                            epe_line_list.append(epe_line.cpu().numpy())
                    
                    # Motion magnitude bins
                    epe_s10 = epe_map[(mag <= 10.0) & valid]
                    epe_s1050 = epe_map[(mag > 10.0) & (mag <= 50.0) & valid]
                    epe_s50 = epe_map[(mag > 50.0) & valid]
                    
                    if epe_s10.numel() > 0:
                        epe_s10_list.append(epe_s10.cpu().numpy())
                    if epe_s1050.numel() > 0:
                        epe_s1050_list.append(epe_s1050.cpu().numpy())
                    if epe_s50.numel() > 0:
                        epe_s50_list.append(epe_s50.cpu().numpy())
            else:
                # No GT: compute proxy loss
                loss = self.base_loss.unsup_forward_only(clip, flows)
                proxy_losses.append(float(loss["total"].detach().item()))
            
            # Visualization (once)
            if self.writer and self.viz_enable and not did_viz:
                n = min(self.viz_max, clip.shape[0])
                grid_img0 = vutils.make_grid(clip[:n, 0], nrow=4, normalize=True)
                grid_img1 = vutils.make_grid(clip[:n, 1], nrow=4, normalize=True)
                self.writer.add_image("val/img0", grid_img0, self.global_step)
                self.writer.add_image("val/img1", grid_img1, self.global_step)
                
                if flows:
                    flow_rgb = _flow_to_rgb(flows[0][:n].clamp(-500, 500))
                    grid_flow = vutils.make_grid(flow_rgb, nrow=4)
                    self.writer.add_image("val/flow_rgb", grid_flow, self.global_step)
                
                did_viz = True
        
        # Aggregate metrics
        metrics: Dict[str, float] = {}
        
        if epe_all_list:
            epe_all = np.concatenate(epe_all_list)
            metrics["epe"] = float(np.mean(epe_all))
        
        if valid_total > 0:
            metrics["1px"] = thr1_cnt / valid_total
            metrics["3px"] = thr3_cnt / valid_total
            metrics["5px"] = thr5_cnt / valid_total
        
        metrics["epe_occ"] = concat_mean(epe_occ_list)
        metrics["epe_nonocc"] = concat_mean(epe_nonocc_list)
        metrics["epe_flat"] = concat_mean(epe_flat_list)
        metrics["epe_line"] = concat_mean(epe_line_list)
        metrics["epe_s<10"] = concat_mean(epe_s10_list)
        metrics["epe_s10-50"] = concat_mean(epe_s1050_list)
        metrics["epe_s>50"] = concat_mean(epe_s50_list)
        
        if proxy_losses:
            metrics["proxy_loss"] = float(np.mean(proxy_losses))
        
        # Log to TensorBoard
        if self.writer:
            for k, v in metrics.items():
                if np.isfinite(v):
                    self.writer.add_scalar(f"val/{k}", v, self.global_step)
        
        # Print summary
        print(f"[Val Epoch {epoch}] " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items() if np.isfinite(v)))
        
        self.model.train()
        return metrics
    
    def _save_checkpoint(self, name: str):
        """Save checkpoint."""
        path = self.workspace / name
        torch.save({
            "epoch": self.current_epoch,
            "step": self.global_step,
            "state_dict": self.model.state_dict(),
            "teacher_state_dict": self.teacher_model.state_dict() if self.teacher_model is not None else None,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "best_metric": self.best_metric,
            "config": self.cfg,
        }, path)
        print(f"Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str, load_optimizer: bool = True):
        """Load checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self._load_module_state(self.model, ckpt["state_dict"], "model")
        if self.teacher_model is not None and ckpt.get("teacher_state_dict") is not None:
            self._load_module_state(self.teacher_model, ckpt["teacher_state_dict"], "teacher")
        
        if load_optimizer and "optimizer" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as exc:
                print(f"[Trainer] Skipping optimizer state from checkpoint: {exc}")
        self._resume_scheduler_state = ckpt.get("scheduler", None)
        
        if "epoch" in ckpt:
            self.current_epoch = ckpt["epoch"]
        if "step" in ckpt:
            self.global_step = ckpt["step"]
        if "best_metric" in ckpt:
            self.best_metric = ckpt["best_metric"]
        
        print(f"Loaded checkpoint: {path}")
