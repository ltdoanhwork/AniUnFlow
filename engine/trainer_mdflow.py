# file: engine/trainer_mdflow.py
"""
MDFlow (FastFlowNet) Trainer
========================================
Dedicated trainer for MDFlow/FastFlowNet on AnimeRun.
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

# from models.aniunflow.losses import UnsupervisedFlowLoss as AFTLosses # Removed
# from .utils.utils import bilinear_sampler # Removed incorrect import

# Import FastFlowNet directly
try:
    from models.MDFlow.models.FastFlowNet import FastFlowNet
except ImportError:
    import sys
    # Fallback hack if running from scripts/
    sys.path.append(str(Path(__file__).parent.parent))
    from models.MDFlow.models.FastFlowNet import FastFlowNet

def concat_mean(chunks) -> float:
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
    try:
        from utils.flow_viz import flow_to_image as _flow_to_image_np
        imgs = []
        for f in flow.detach().permute(0, 2, 3, 1).cpu().numpy():
            im = _flow_to_image_np(f)
            imgs.append(torch.from_numpy(im).permute(2, 0, 1))
        out = torch.stack(imgs).float() / 255.0
        return out.to(flow.device)
    except Exception:
        # Fallback: simple normalization for visualization if util missing
        return torch.zeros(flow.shape[0], 3, flow.shape[2], flow.shape[3]).to(flow.device)

def resize_mask(mask: Optional[torch.Tensor], target_hw: Tuple[int, int], device: torch.device) -> Optional[torch.Tensor]:
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

# --- Custom Pairwise Loss ---

def shim_warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float().to(x.device)
    vgrid = grid + flo
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    return F.grid_sample(x, vgrid, mode='bilinear', align_corners=True, padding_mode="border")

def gradient(data):
    D_dy = data[:, :, 1:] - data[:, :, :-1]
    D_dx = data[:, :, :, 1:] - data[:, :, :, :-1]
    return D_dx, D_dy

def smooth_grad_1st(flo, image, alpha=10):
    img_dx, img_dy = gradient(image)
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)

    dx, dy = gradient(flo)
    loss_x = weights_x * dx.abs() / 2.0
    loss_y = weights_y * dy.abs() / 2.0
    return loss_x.mean() / 2.0 + loss_y.mean() / 2.0

def SSIM(x, y, md=1):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_x = nn.AvgPool2d(3, 1, 0)(x)
    mu_y = nn.AvgPool2d(3, 1, 0)(y)
    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x = nn.AvgPool2d(3, 1, 0)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(3, 1, 0)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(3, 1, 0)(x * y) - mu_xy
    SSIM_n = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d
    return torch.clamp((1 - SSIM) / 2, 0, 1)

class PairwiseUnsupervisedLoss(nn.Module):
    def __init__(self, alpha_ssim=0.85, w_smooth=10.0):
        super().__init__()
        self.alpha_ssim = alpha_ssim
        self.w_smooth = w_smooth

    def forward(self, img1, img2, flow_fw):
        """
        img1, img2: [B,3,H,W]
        flow_fw: [B,2,h,w] (will be upsampled if needed)
        """
        if flow_fw.shape[-2:] != img1.shape[-2:]:
            flow_fw = F.interpolate(flow_fw, size=img1.shape[-2:], mode="bilinear", align_corners=True)
            # Scaling flow: if flow was computed at 1/4 res, does it need scaling?
            # Usually flow magnitude is pixels.
            # If FastFlowNet output is already scaled to full res pixels but downsampled size, then no mult needed.
            # But usually upscale aligns it.
            # Let's assume flow is in pixels of its own resolution.
            scale_h = img1.shape[2] / flow_fw.shape[2] # wait, standard upsample doesn't change value
            # But "resize flow" usually implies multiplying logic.
            # FastFlowNet returns flow at specific scales.
            # If we upsample flow_1/4 to flow_1, we must multiply by 4.
            # HOWEVER, `upsample_flow` in original loss did: f_up[:,0] *= sx.
            # We should probably do that too.
            scale_x = img1.shape[3] / float(flow_fw.shape[3])
            scale_y = img1.shape[2] / float(flow_fw.shape[2])
            flow_fw = flow_fw * 1.0 # Clone
            flow_fw[:, 0] *= scale_x
            flow_fw[:, 1] *= scale_y

        img2_warped = shim_warp(img2, flow_fw)
        
        # Photometric
        # L1
        diff = torch.abs(img1 - img2_warped)
        l1_loss = diff.mean()
        # SSIM
        ssim_loss = SSIM(img1, img2_warped).mean()
        
        photo_loss = (1 - self.alpha_ssim) * l1_loss + self.alpha_ssim * ssim_loss
        
        # Smoothness
        smooth_loss = smooth_grad_1st(flow_fw, img1)
        
        total = photo_loss + self.w_smooth * smooth_loss
        
        return {
            "total": total,
            "photo": photo_loss,
            "smooth": smooth_loss
        }

class MDFlowTrainer:
    """
    Dedicated Trainer for FastFlowNet.
    """
    
    def __init__(self, cfg: Dict, workspace: Path):
        self.cfg = cfg
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_metric = float("inf")
        self.global_step = 0
        self.current_epoch = 0
        
        # Build FastFlowNet
        print("Building FastFlowNet...")
        mcfg = self.cfg.get("model", {})
        args = mcfg.get("args", {})
        self.model = FastFlowNet(groups=args.get("groups", 3)).to(self.device)
        self.model.train()
        
        # Loss
        # Loss
        self.loss_fn = PairwiseUnsupervisedLoss(
            alpha_ssim=cfg.get("loss", {}).get("alpha_ssim", 0.85), # Standard 0.85 usually
            w_smooth=cfg.get("loss", {}).get("w_smooth", 10.0), # Matches typical UnFlow/ARFlow range
        )
        
        # Optimizer
        optim_cfg = cfg.get("optim", {})
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(optim_cfg.get("lr", 2e-4)),
            weight_decay=float(optim_cfg.get("weight_decay", 1e-4)),
            betas=tuple(optim_cfg.get("betas", [0.9, 0.999])),
        )
        
        # Scheduler
        self.scheduler = None
        self.sched_per_batch = False
        
        self.scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
        self.clip_grad = float(optim_cfg.get("clip", 1.0))
        self.accum_steps = int(optim_cfg.get("accum_steps", 1))
        
        # Logging
        log_cfg = cfg.get("logging", {})
        self.use_tb = log_cfg.get("use_tb", True)
        self.log_every = int(log_cfg.get("log_every", 100))
        self.writer: Optional[SummaryWriter] = None
        if self.use_tb:
            tb_dir = self.workspace / log_cfg.get("tb_dir", "tb")
            tb_dir.mkdir(exist_ok=True, parents=True)
            self.writer = SummaryWriter(str(tb_dir))
            
        # Visualization
        viz_cfg = cfg.get("viz", {})
        self.viz_enable = viz_cfg.get("enable", True)
        self.viz_max = int(viz_cfg.get("max_samples", 8))
        
        # Validation
        val_cfg = cfg.get("validation", {})
        self.val_every_epochs = val_cfg.get("every_n_epochs", 5)

    def _init_scheduler(self, train_loader: DataLoader, epochs: int):
        sched_cfg = self.cfg.get("optim", {}).get("scheduler", {})
        sched_type = sched_cfg.get("type", "cosine").lower()
        self.sched_per_batch = sched_cfg.get("per_batch", False)
        warmup_epochs = int(self.cfg.get("optim", {}).get("warmup_epochs", 0))
        
        if sched_type == "onecycle":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=float(self.cfg["optim"]["lr"]),
                steps_per_epoch=max(1, len(train_loader)),
                epochs=epochs,
                pct_start=warmup_epochs / epochs if epochs > 0 else 0.05,
            )
            self.sched_per_batch = True
        else:
            # Default cosine
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=float(sched_cfg.get("min_lr", 1e-6)),
            )

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        epochs = int(self.cfg["optim"]["epochs"])
        self._init_scheduler(train_loader, epochs)
        
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            self._train_one_epoch(train_loader)
            
            val_metric = None
            if val_loader is not None and epoch % self.val_every_epochs == 0:
                val_metrics = self.validate(val_loader, epoch=epoch)
                val_metric = val_metrics.get("epe", float("inf"))
                
                if self.writer:
                    for k, v in val_metrics.items():
                        self.writer.add_scalar(f"val/{k}", v, epoch)
                
            if self.scheduler and not self.sched_per_batch:
                self.scheduler.step()
                
            # Checkpointing
            if val_metric is not None and val_metric < self.best_metric:
                self.best_metric = val_metric
                self._save_checkpoint("best_mdflow.pth")
                
            ckpt_cfg = self.cfg.get("ckpt", {})
            if epoch % ckpt_cfg.get("save_every", 5) == 0:
                self._save_checkpoint(f"ckpt_mdflow_e{epoch:03d}.pth")

            # Debug: Max steps check
            max_steps = self.cfg.get("debug", {}).get("max_steps", 0)
            if max_steps > 0 and self.global_step >= max_steps:
               print(f"[DEBUG] Training stopped (max_steps={max_steps}).")
               break

    def _train_one_epoch(self, train_loader: DataLoader):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            batch = _to_device(batch, self.device)
            clip = batch["clip"]
            if clip.dtype == torch.uint8:
                clip = clip.float() / 255.0
                
            # FastFlowNet logic:
            # Input: T=3 clip [B, 3, 3, H, W]
            # Use Pair 1: frame 0 -> frame 1
            # Concatenate for FastFlowNet [B, 6, H, W]
            
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                # Forward Image Pair
                img1 = clip[:, 0]
                img2 = clip[:, 1]
                imgs_fw = torch.cat([img1, img2], dim=1)
                
                # Output is tuple of flows (flow2, flow3, flow4, flow5, flow6)
                out_fw = self.model(imgs_fw)
                if not isinstance(out_fw, (list, tuple)):
                    out_fw = [out_fw]
                flows_fw = list(out_fw)
                
                # Backward Image Pair (img2 -> img1)
                imgs_bw = torch.cat([img2, img1], dim=1)
                out_bw = self.model(imgs_bw)
                if not isinstance(out_bw, (list, tuple)):
                    out_bw = [out_bw]
                flows_bw = list(out_bw)
                
                # Multi-scale Unsupervised Loss
                # Unsup loss expects lists of [B, 2, h, w].
                # We iterate scales.
                weights = [1.0, 0.5, 0.25, 0.125, 0.1]
                total_loss = 0.0
                loss_dict = {}
                
                n_scales = min(len(flows_fw), len(flows_bw))
                for i in range(n_scales):
                    # Wrap in list because unsup_bidirectional expects list of T-1 flows
                    fw_i = [flows_fw[i]]
                    bw_i = [flows_bw[i]]
                    
                    loss_res = self.loss_fn(img1, img2, flows_fw[i])
                    
                    w = weights[i] if i < len(weights) else 0.1
                    total_loss += loss_res["total"] * w
                    
                    if i == 0:
                        loss_dict.update(loss_res)
            
            # Optimization
            loss_scaled = total_loss / self.accum_steps
            self.scaler.scale(loss_scaled).backward()
            
            if (batch_idx + 1) % self.accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.sched_per_batch and self.scheduler:
                    self.scheduler.step()
            
            self.global_step += 1
            pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})
            

            if self.writer and self.global_step % self.log_every == 0:
                self._log_step(clip, flows_fw, loss_dict)
            
            # Debug: Max steps
            max_steps = self.cfg.get("debug", {}).get("max_steps", 0)
            if max_steps > 0 and self.global_step >= max_steps:
                print(f"[DEBUG] Reached max_steps ({max_steps}). Stopping epoch.")
                break
                
    def _log_step(self, clip, flows, loss_dict):
        step = self.global_step
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                self.writer.add_scalar(f"train_loss/{k}", v.item(), step)
        if flows:
            # Viz finest flow
            f = flows[0][0].detach() # Batch 0
            if f is not None:
                # Basic viz logic if needed
                pass

    @torch.no_grad()
    def validate(self, val_loader: DataLoader, epoch: int = 0) -> Dict[str, float]:
        self.model.eval()
        
        # Accumulators
        epe_all_list = []
        epe_occ_list = []
        epe_no_occ_list = []
        out_list = [] # For 1px, 3px, 5px
        
        total_samples = 0
        p1_cnt = p3_cnt = p5_cnt = 0
        
        for i, batch in enumerate(tqdm(val_loader, desc="Validate")):
            max_val = self.cfg.get("debug", {}).get("max_val_steps", 0)
            if max_val > 0 and i >= max_val:
                print(f"[DEBUG] Reached max_val_steps ({max_val}). Breaking validation.")
                break
            
            batch = _to_device(batch, self.device)
            clip = batch["clip"]
            if clip.dtype == torch.uint8:
                clip = clip.float() / 255.0
            
            # Predict
            imgs = torch.cat([clip[:, 0], clip[:, 1]], dim=1)
            out = self.model(imgs)
            
            pred = out
            if isinstance(out, (tuple, list)):
                pred = out[0] # use finest
            
            # GT
            gt = batch.get("flow", None)
            valid = batch.get("valid", None) # [B, 1, H, W] or None
            occ = batch.get("occ", None)     # [B, 1, H, W] or None
            
            if gt is not None:
                B, C, H, W = gt.shape
                
                # Resize prediction to GT size
                if pred.shape[-2:] != gt.shape[-2:]:
                    # FastFlowNet output is usually 1/4? If so, we strictly resize.
                    # Standard practice: bilinear upsample, scale flow values by size ratio
                    h_pred, w_pred = pred.shape[-2:]
                    pred = F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=True)
                    pred[:, 0] *= W / float(w_pred)
                    pred[:, 1] *= H / float(h_pred)
                
                # Compute EPE map
                epe_map = torch.norm(pred - gt, dim=1, keepdim=True) # [B, 1, H, W]
                
                # Handle valid mask
                mask = torch.ones_like(epe_map, dtype=torch.bool)
                if valid is not None:
                    # Resize valid mask if needed
                    if valid.shape[-2:] != (H, W):
                        valid = F.interpolate(valid.float(), size=(H, W), mode="nearest").bool()
                    mask = valid.bool()
                
                # Overall EPE
                valid_epe = epe_map[mask]
                if valid_epe.numel() > 0:
                    epe_all_list.append(valid_epe.cpu().numpy())
                    
                    # Accuracy
                    p1_cnt += (valid_epe < 1.0).sum().item()
                    p3_cnt += (valid_epe < 3.0).sum().item()
                    p5_cnt += (valid_epe < 5.0).sum().item()
                    total_samples += valid_epe.numel()

                # Occlusions
                if occ is not None:
                    if occ.shape[-2:] != (H, W):
                         occ = F.interpolate(occ.float(), size=(H, W), mode="nearest")
                    
                    # occ usually: 1=occluded? Or 0=occluded?
                    # In Sintel/KITTI: often provided as "occ" mask where 1 is occlusion?
                    # Let's check typical usage. In AniUnFlow dataset, occ might be 1 for occluded.
                    # Trainer segment aware says: epe_occ = epe[(occ==0) & valid] -> Wait, usually occ=0 means NOT occluded?
                    # Let's check logic: "epe_occ ... occ==0". That implies occ=0 is the "Occ" region? 
                    # OR is it "noc" map (non-occluded)?
                    # Usually datasets provide "noc" (1 where valid and non-occluded).
                    # If the key is 'occ', standard convention is 1=occluded.
                    # SegmentAwareTrainer: "epe_occ = epe_map[(occ == 0) & valid]" -> This suggests 'occ' variable is actually a 'non-occluded' mask (1=visible, 0=occluded)?
                    # Actually standard Sintel SDK returns 'occ'.
                    # Let's stick to generic keys. If "occ" is present:
                    # We will assume key "occ" means 1=occluded, 0=visible (standard name).
                    # OR we can log both and see.
                    # Let's check logic in reference code again:
                    # "epe_occ = epe_map[(occ == 0) & valid]" implies occ=0 is the target region for "epe_occ".
                    # If occ is "occlusions", then occ=1 is occluded. So occ=0 is non-occluded.
                    # So "epe_occ" getting stats from "occ==0"? That sounds backwards for the variable name.
                    # Unless the variable `occ` from batch is actually `noc` (non-occluded)?
                    # Let's assume standard behavior: Log epe_occ (masked by occ==1) and epe_noc (masked by occ==0).
                    # But if the ref code does the opposite, I might invert them.
                    # Ref code: epe_occ = epe[(occ == 0) ...]. 
                    # Ref code: epe_nonocc = epe[(occ == 1) ...].
                    # This implies the batch["occ"] is actually a VALIDITY mask where 1=visible/non-occluded, 0=occluded.
                    # I will mirror this logic.
                    
                    is_occ = (occ < 0.5) & mask
                    is_noc = (occ > 0.5) & mask
                    
                    if is_occ.any():
                        epe_occ_list.append(epe_map[is_occ].cpu().numpy())
                    if is_noc.any():
                        epe_no_occ_list.append(epe_map[is_noc].cpu().numpy())

        # Aggregation
        metrics = {}
        if epe_all_list:
            all_concat = np.concatenate(epe_all_list)
            metrics["epe"] = float(np.mean(all_concat))
            
        if total_samples > 0:
            metrics["1px"] = p1_cnt / total_samples
            metrics["3px"] = p3_cnt / total_samples
            metrics["5px"] = p5_cnt / total_samples
            
        if epe_occ_list:
            metrics["epe_occ"] = float(np.mean(np.concatenate(epe_occ_list)))
        if epe_no_occ_list:
            metrics["epe_noc"] = float(np.mean(np.concatenate(epe_no_occ_list)))
            
        print(f"[Val Epoch {epoch}] EPE: {metrics.get('epe', -1):.4f} | 1px: {metrics.get('1px', -1):.4f}")
        
        self.model.train()
        return metrics

    def _save_checkpoint(self, name: str):
        path = self.workspace / name
        torch.save({
            "epoch": self.current_epoch,
            "state_dict": self.model.state_dict(),
            "best_metric": self.best_metric,
        }, path)
        print(f"Saved checkpoint: {path}")
