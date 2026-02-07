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
import math
import time
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
from models import AniFlowFormerT, ModelConfig as AFConfig
from models.aniunflow.losses import UnsupervisedFlowLoss as AFTLosses

# Segment-aware imports
from losses.segment_aware_losses import SegmentAwareLossModule, build_segment_aware_losses
from models.aniunflow.sam2_guidance import SAM2GuidanceModule, build_sam2_guidance


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
        
        # Build loss modules
        self.base_loss = AFTLosses(
            alpha_ssim=cfg.get("loss", {}).get("alpha_ssim", 0.2),
            w_smooth=cfg.get("loss", {}).get("w_smooth", 0.1),
            w_cons=cfg.get("loss", {}).get("w_cons", 0.05),
            # Anti-collapse regularization
            w_mag_reg=cfg.get("loss", {}).get("w_mag_reg", 0.01),
            min_flow_mag=cfg.get("loss", {}).get("min_flow_mag", 0.5),
            use_photo_gradient=cfg.get("loss", {}).get("use_photo_gradient", True),
        )
        
        # Warmup settings for occlusion masking
        self.warmup_steps = cfg.get("loss", {}).get("warmup_steps", 0)
        self.disable_occ_during_warmup = cfg.get("loss", {}).get("disable_occ_during_warmup", False)
        
        # Segment-aware losses
        self.use_segment_losses = self._has_segment_losses()
        if self.use_segment_losses:
            self.segment_loss = build_segment_aware_losses(cfg)
        
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
        
        # Scheduler (configured later)
        self.scheduler = None
        self.sched_per_batch = False
        
        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        self.clip_grad = float(optim_cfg.get("clip", 1.0))
        self.accum_steps = int(optim_cfg.get("accum_steps", 1))
        
        # Semi-supervised weight
        self.w_epe_sup = float(cfg.get("loss", {}).get("w_epe_sup", 0.0))
        
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
    
    def _build_model(self) -> nn.Module:
        """Build AniFlowFormer-T model from config."""
        mcfg = self.cfg.get("model", {})
        args = mcfg.get("args", {})
        
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
            # Use V1/V2 model
            return AniFlowFormerT(AFConfig(**args))
    
    def _has_segment_losses(self) -> bool:
        """Check if any segment-aware losses are enabled."""
        loss_cfg = self.cfg.get("loss", {})
        seg_cons = loss_cfg.get("segment_consistency", {}).get("enabled", False)
        boundary = loss_cfg.get("boundary_aware_smooth", {}).get("enabled", False)
        temporal = loss_cfg.get("temporal_memory", {}).get("enabled", False)
        return seg_cons or boundary or temporal
    
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
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=float(self.cfg["optim"]["lr"]),
                steps_per_epoch=max(1, len(train_loader)),
                epochs=epochs,
                pct_start=warmup_epochs / epochs if epochs > 0 else 0.05,
            )
            self.sched_per_batch = True
        else:
            self.scheduler = None
    
    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """Main training loop."""
        epochs = int(self.cfg["optim"]["epochs"])
        self._init_scheduler(train_loader, epochs)
        
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            t0 = time.time()
            
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
            
            print(
                f"[Epoch {epoch:03d}/{epochs}] "
                f"train_loss={train_loss:.4f} "
                f"val_epe={val_metric if val_metric else 'N/A'} "
                f"lr={lr:.2e} time={dt:.1f}s"
            )
            
            if self.writer:
                self.writer.add_scalar("train/loss_epoch", train_loss, epoch)
                self.writer.add_scalar("train/lr_epoch", lr, epoch)
                if val_metric is not None and np.isfinite(val_metric):
                    self.writer.add_scalar("val/epe_epoch", val_metric, epoch)
            
            # Save best
            if val_metric is not None and np.isfinite(val_metric) and val_metric < self.best_metric:
                self.best_metric = val_metric
                self._save_checkpoint("best.pth")
            
            # Periodic checkpoint
            ckpt_cfg = self.cfg.get("ckpt", {})
            if epoch % ckpt_cfg.get("save_every", 5) == 0:
                self._save_checkpoint(f"ckpt_e{epoch:03d}.pth")
        
        print(f"\nTraining complete! Best metric: {self.best_metric:.4f}")
    
    def _train_one_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        accum = {"loss": 0.0, "steps": 0}
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
            if self.use_sam and self.sam_guidance is not None:
                # Use pre-loaded masks if available (faster)
                if "sam_masks" in batch:
                    segment_masks = batch["sam_masks"]
                    # Calculate boundary maps from pre-loaded masks
                    with torch.no_grad():
                         boundary_maps = self.sam_guidance.compute_boundary_maps(segment_masks)
                else:
                    # Fallback to online generation (slower)
                    with torch.no_grad():
                        segment_masks = self.sam_guidance.extract_segment_masks(clip)
                        boundary_maps = self.sam_guidance.compute_boundary_maps(segment_masks)
            
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                # Forward pass
                out_fw = self.model(clip, sam_masks=segment_masks)
                flows_fw = out_fw["flows"]
                
                # Backward pass (reverse clip)
                clip_rev = torch.flip(clip, dims=[1])
                segment_masks_rev = torch.flip(segment_masks, dims=[1]) if segment_masks is not None else None
                out_bw = self.model(clip_rev, sam_masks=segment_masks_rev)
                flows_bw_rev = out_bw["flows"]
                
                # Re-index backward flows
                flows_bw = [flows_bw_rev[len(flows_bw_rev) - 1 - k] for k in range(len(flows_fw))]
                
                # Compute base unsupervised loss
                # Disable occlusion masking during warmup to let flow build up
                use_occ_mask = True
                if self.disable_occ_during_warmup and self.global_step < self.warmup_steps:
                    use_occ_mask = False
                loss_dict = self.base_loss.unsup_bidirectional(clip, flows_fw, flows_bw, use_occ_mask=use_occ_mask)

                total_loss = loss_dict["total"]
                
                # Add segment-aware losses
                if self.use_segment_losses and segment_masks is not None:
                    seg_losses = self.segment_loss(
                        flows_fw, clip, segment_masks, boundary_maps
                    )
                    total_loss = total_loss + seg_losses["total_segment_loss"]
                    loss_dict.update(seg_losses)
                
                # Optional semi-supervised loss
                if self.w_epe_sup > 0 and "flow" in batch:
                    gt = batch["flow"]
                    if len(flows_fw) > 0:
                        pred = flows_fw[0]
                        if pred.shape[-2:] != gt.shape[-2:]:
                            pred = F.interpolate(pred, size=gt.shape[-2:], mode="bilinear", align_corners=True)
                        epe = _masked_epe(pred, gt).mean()
                        total_loss = total_loss + self.w_epe_sup * epe
            
            # Backward
            loss_scaled = total_loss / self.accum_steps
            self.scaler.scale(loss_scaled).backward()
            
            # Optimizer step
            if (batch_idx + 1) % self.accum_steps == 0:
                if self.clip_grad > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                
                if self.sched_per_batch and self.scheduler:
                    self.scheduler.step()
            
            self.global_step += 1
            accum["loss"] += float(total_loss.detach())
            accum["steps"] += 1
            
            pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})
            
            # Logging
            if self.writer and self.global_step % self.log_every == 0:
                self._log_training_step(clip, flows_fw, loss_dict, segment_masks)
        
        return accum
    
    def _log_training_step(
        self,
        clip: torch.Tensor,
        flows: List[torch.Tensor],
        loss_dict: Dict[str, Any],
        segment_masks: Optional[torch.Tensor] = None,
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
        
        # Segment statistics
        if self.log_segment_stats and segment_masks is not None:
            mask = segment_masks.detach()
            num_segments = mask.shape[2]
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
                segment_masks = self.sam_guidance.extract_segment_masks(clip)
            
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = self.model(clip, sam_masks=segment_masks)
            
            flows = out["flows"]
            
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
                    
                    epe_map = torch.norm(pred - gt, dim=1)
                    mag = torch.norm(gt, dim=1)
                    
                    # Valid mask
                    if valid_any is None:
                        valid = torch.ones_like(epe_map, dtype=torch.bool)
                    else:
                        valid = resize_mask(valid_any, gt.shape[-2:], device)
                        if valid.dim() == 3 and valid.shape[0] == 1:
                            valid = valid.expand(B, -1, -1)
                    
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
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "best_metric": self.best_metric,
            "config": self.cfg,
        }, path)
        print(f"Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str, load_optimizer: bool = True):
        """Load checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        
        if load_optimizer and "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        
        if "epoch" in ckpt:
            self.current_epoch = ckpt["epoch"]
        if "step" in ckpt:
            self.global_step = ckpt["step"]
        if "best_metric" in ckpt:
            self.best_metric = ckpt["best_metric"]
        
        print(f"Loaded checkpoint: {path}")
