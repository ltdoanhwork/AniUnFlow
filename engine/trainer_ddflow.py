# file: engine/trainer_ddflow.py
"""
DDFlow (PWC-Net) Trainer
========================================
Dedicated trainer for DDFlow on AnimeRun (PyTorch Port).
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
from tqdm import tqdm

from models.DDFlow.models.DDFlowNet_pt import DDFlowNet

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
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    return F.grid_sample(x, vgrid, mode='bilinear', align_corners=True, padding_mode="border")

# --- DDFlow Losses (Ported from ddflow_model.py) ---

class DDFlowLoss(nn.Module):
    def __init__(self, use_occlusion=True):
        super().__init__()
        self.use_occlusion = use_occlusion

    def abs_robust_loss(self, diff, mask, q=0.4):
        # diff = tf.pow((tf.abs(diff)+0.01), q)
        diff = torch.pow((torch.abs(diff) + 0.01), q)
        diff = diff * mask
        diff_sum = diff.sum()
        loss_mean = diff_sum / (mask.sum() * 2 + 1e-6)
        return loss_mean

    def census_loss(self, img1, img2_warped, mask, max_distance=3):
        # Implementation of Ternary Transform and Hamming Distance
        # Using simple patch comparison logic if possible or exact conv implementation
        
        def _ternary_transform(image):
            # RGB to Gray
            gray = 0.2989 * image[:, 0:1] + 0.5870 * image[:, 1:2] + 0.1140 * image[:, 2:3]
            intensities = gray * 255.0
            
            # Conv implementation of patches
            # TF: ksizes=[1, patch_size, patch_size, 1], strides=1, padding='SAME'
            # Typically 3x3 or max_distance=3 -> 7x7 patch? 
            # DDFlow says max_distance=3 -> patch_size = 2*3+1 = 7
            patch_size = 7
            out_channels = patch_size * patch_size
            
            # Weights: identity for each pixel in patch
            # We can use unfold
            # Or manually construct conv weights
            # Easier to use unfold for Hamming
            
            # But the TF implementation uses `tf.extract_image_patches` or 1x1 conv simulation?
            # TF Code: 
            # w = np.eye(out_channels).reshape(...)
            # patches = tf.nn.conv2d(..., weights, ...)
            # This creates 'patches' tensor where C dim is flattened patch pixels
            
            # In PyTorch:
            # unfold extract patches
            # [B, C, H, W] -> [B, C*k*k, L] ?
            # Let's use Conv2d with fixed weights to extract patches
            # Input: [B, 1, H, W] -> Output: [B, 49, H, W]
            
            w = torch.eye(out_channels).reshape(out_channels, 1, patch_size, patch_size)
            weights = w.type_as(image)
            patches = F.conv2d(intensities, weights, padding=3) # padding=3 for 7x7 to keep size
            
            transf = patches - intensities
            transf_norm = transf / torch.sqrt(0.81 + torch.square(transf))
            return transf_norm

        def _hamming_distance(t1, t2):
            dist = torch.square(t1 - t2)
            dist_norm = dist / (0.1 + dist)
            dist_sum = torch.sum(dist_norm, dim=1, keepdim=True)
            return dist_sum
            
        t1 = _ternary_transform(img1)
        t2 = _ternary_transform(img2_warped)
        dist = _hamming_distance(t1, t2)
        
        # transform_mask logic (erode borders?)
        # TF: create_mask with max_distance padding
        # We can just ignore border? Or rely on mask
        # If mask provided, multiply.
        return self.abs_robust_loss(dist, mask) # Should we mask borders? Maybe.

    def forward(self, img1, img2, flow_fw, flow_bw):
        """
        DDFlow specific unsuperivsed loss.
        """
        # Warp
        img1_warp = shim_warp(img1, flow_bw) # warp img1 with flow_bw? Wait.
        # ddflow_model.py: 
        # img1_warp = tf_warp(batch_img1, flow_bw['full_res']) -> Warps img1 using flow_bw? 
        # Usually img2_warp = warp(img2, flow_fw) checks consistency with img1.
        # Img1(t) ~= Img2(t+1) warped back.
        # DDFlow TF: img2_warp = tf_warp(batch_img2, flow_fw['full_res'])
        #            loss(img1 - img2_warp) matches.
        
        img2_warp = shim_warp(img2, flow_fw)
        img1_warp = shim_warp(img1, flow_bw) # For consistency check with img2
        
        # Valid masks (Occlusion detection via fwd-bwd check)
        # occlusion() in TF code uses simple range check? 
        # Actually ddflow_model.py calls `occlusion` from utils.
        # It usually implements: |flow_fw + warp(flow_bw)| < threshold
        
        def compute_occ(flow1, flow2): # flow_fw, flow_bw
            # flow2 warped to flow1 frame
            flow2_warped = shim_warp(flow2, flow1)
            flow_diff = flow1 + flow2_warped # should be 0
            mag_sq = torch.sum(flow_diff**2, dim=1, keepdim=True)
            flow_mag_sq = torch.sum(flow1**2, dim=1, keepdim=True) + torch.sum(flow2_warped**2, dim=1, keepdim=True)
            
            # Simple thresholding often used: |diff| > 0.01 * (|f1| + |f2|) + 0.5
            # DDFlow utils.py: 
            # def occlusion(flow_fw, flow_bw):
            #   mag_sq ...
            #   occ_thresh =  0.01 * (mag_sq_fw + mag_sq_bw) + 0.5
            #   occ = mag_sq_diff > occ_thresh
            
            occ_thresh = 0.01 * flow_mag_sq + 0.5
            occ = (mag_sq > occ_thresh).float()
            return occ
        
        occ_fw = compute_occ(flow_fw, flow_bw)
        occ_bw = compute_occ(flow_bw, flow_fw)
        
        mask_fw = 1.0 - occ_fw
        mask_bw = 1.0 - occ_bw
        
        # Losses
        loss_photo = 0.0
        loss_census = 0.0
        
        # Photometric (Abs Robust)
        p_fw = self.abs_robust_loss(img1 - img2_warp, mask_fw)
        p_bw = self.abs_robust_loss(img2 - img1_warp, mask_bw)
        loss_photo = p_fw + p_bw
        
        # Census
        c_fw = self.census_loss(img1, img2_warp, mask_fw)
        c_bw = self.census_loss(img2, img1_warp, mask_bw)
        loss_census = c_fw + c_bw
        
        # Regular DDFlow doesn't use smoothness in main loss? 
        # ddflow_model.py: compute_losses -> abs_robust_mean + census
        # Does it use smoothness? 
        # In `compute_losses` function, it lists: abs_robust, census.
        # And then adds `regularizer_loss` (L2 weights).
        # It seems DDFlow relies on distillation or just these data terms.
        # But 'no_distillation' mode (unsupervised) uses: abs_robust + census.
        
        total = loss_photo + loss_census
        return {
            "total": total,
            "photo": loss_photo,
            "census": loss_census,
            "occ_fw": occ_fw
        }

class DDFlowTrainer:
    def __init__(self, cfg: Dict, workspace: Path):
        self.cfg = cfg
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_metric = float("inf")
        self.global_step = 0
        self.current_epoch = 0
        
        # Model
        print("Building DDFlowNet...")
        self.model = DDFlowNet().to(self.device)
        self.model.train()
        
        # Loss
        self.loss_fn = DDFlowLoss()
        
        # Optimizer
        optim_cfg = cfg.get("optim", {})
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(optim_cfg.get("lr", 2e-4)),
            weight_decay=float(optim_cfg.get("weight_decay", 1e-4))
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        
        # Helpers
        self.accum_steps = int(optim_cfg.get("accum_steps", 1))
        self.log_every = 100
        tb_dir = self.workspace / "tb"
        tb_dir.mkdir(exist_ok=True)
        self.writer = SummaryWriter(str(tb_dir))
        self.val_every_epochs = cfg.get("validation", {}).get("every_n_epochs", 5)

    def save_checkpoint(self, epoch, is_best=False):
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_metric": self.best_metric,
        }
        
        # Save regular checkpoint
        torch.save(state, self.workspace / f"ckpt_ddflow_e{epoch:03d}.pth")
        
        # Save best
        if is_best:
            torch.save(state, self.workspace / "best_ddflow.pth")
            
    def load_checkpoint(self, ckpt_path):
        print(f"Resuming from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        
        # Check if it's a full checkpoint or just model weights (legacy)
        if "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scaler.load_state_dict(checkpoint["scaler"])
            self.current_epoch = checkpoint["epoch"]
            self.global_step = checkpoint["global_step"]
            self.best_metric = checkpoint.get("best_metric", float("inf"))
            print(f"Resumed full state from epoch {self.current_epoch}")
        else:
            self.model.load_state_dict(checkpoint)
            print("Resumed model weights only (legacy checkpoint). Optimizer restarted.")
            # Try to infer epoch from filename
            import re
            match = re.search(r'e(\d+)', Path(ckpt_path).name)
            if match:
                self.current_epoch = int(match.group(1))
                print(f"Inferred epoch {self.current_epoch} from filename")

    def fit(self, train_loader, val_loader=None, resume_path=None):
        if resume_path:
            self.load_checkpoint(resume_path)
            
        start_epoch = self.current_epoch + 1
        epochs = int(self.cfg.get("optim", {}).get("epochs", 50))
        
        print(f"Starting training from epoch {start_epoch} to {epochs}")
        
        for epoch in range(start_epoch, epochs + 1):
            self.current_epoch = epoch
            self._train_one_epoch(train_loader)
            
            # Validation
            if val_loader and epoch % self.val_every_epochs == 0:
                metrics = self.validate(val_loader, epoch=epoch)
                # Log metrics
                if self.writer:
                    for k, v in metrics.items():
                        self.writer.add_scalar(f"val/{k}", v, epoch)
                        
                val_epe = metrics.get("epe", float("inf"))
                is_best = val_epe < self.best_metric
                if is_best:
                    self.best_metric = val_epe
            else:
                is_best = False
            
            # Save checkpoint
            if epoch % 5 == 0 or is_best:
                 self.save_checkpoint(epoch, is_best)

    def _train_one_epoch(self, loader):
        self.model.train()
        pbar = tqdm(loader, desc=f"Epoch {self.current_epoch}")
        
        # Multi-scale weights typically used in PWCNet training 
        # but DDFlow paper might treat sum of scales?
        # DDFlow code: pyramid_processing_bidirection -> returns flow_fw dict (levels)
        # compute_losses uses 'full_res' flow primarily?
        # "img1_warp = tf_warp(batch_img1, flow_bw['full_res']...)"
        # It seems it only supervises the FINAL resolution flow in Unsupervised mode?
        # Re-checking ddflow_model.py...
        # "losses = self.compute_losses(..., flow_fw, ...)"
        # "img1_warp = tf_warp(batch_img1, flow_bw['full_res']...)"
        # It uses ONLY full_res flow for loss calculation in `compute_losses`.
        # Wait, usually PWCNet trains with multi-scale supervision.
        # But `compute_losses` definitely uses `flow_fw['full_res']`.
        # Maybe it strictly supervises the output.
        # I will follow this pattern: Supervise full resolution flow.
        
        for i, batch in enumerate(pbar):
            batch = _to_device(batch, self.device)
            clip = batch["clip"]
            if clip.dtype == torch.uint8: clip = clip.float() / 255.0
            
            # Helper: get pair (0->1)
            img1 = clip[:, 0]
            img2 = clip[:, 1]
            imgs = torch.cat([img1, img2], dim=1) # [B, 6, H, W]
            
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                # Model returns tuple of flows (f2..f6) in training
                # Or we can modify it to return full res? 
                # DDFlowNet_pt returns (f2, f3, f4, f5, f6)
                flows_fw = self.model(imgs) 
                
                # Check DDFlowNet output. It outputs 5 flows.
                # flow2 is the finest (1/4 res).
                # To match DDFlow 'full_res' supervision, we need to upsample flow2 to full res.
                flow_finest = flows_fw[0]
                flow_full = F.interpolate(flow_finest, size=img1.shape[-2:], mode="bilinear", align_corners=True) * 4.0
                
                # We need Backward flow too for occlusion!
                # Run model on reverse pair?
                imgs_bw = torch.cat([img2, img1], dim=1)
                flows_bw = self.model(imgs_bw)
                flow_bw_full = F.interpolate(flows_bw[0], size=img1.shape[-2:], mode="bilinear", align_corners=True) * 4.0
                
                loss_dict = self.loss_fn(img1, img2, flow_full, flow_bw_full)
                loss = loss_dict["total"]
                
                loss_scaled = loss / self.accum_steps
            
            self.scaler.scale(loss_scaled).backward()
            
            if (i+1) % self.accum_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
            if i % self.log_every == 0 and self.writer:
                 self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                 self.writer.add_scalar("train/photo", loss_dict["photo"].item(), self.global_step)
                 self.writer.add_scalar("train/census", loss_dict["census"].item(), self.global_step)
            self.global_step += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    @torch.no_grad()
    def validate(self, loader, epoch=0):
        self.model.eval()
        epe_list = []
        p1_cnt = p3_cnt = 0
        total = 0
        
        for batch in tqdm(loader, desc="Val"):
            batch = _to_device(batch, self.device)
            clip = batch["clip"]
            if clip.dtype == torch.uint8: clip = clip.float() / 255.0
            if clip.dtype == torch.uint8: clip = clip.float() / 255.0
            
            # Dataset returns "flow_list" for sequences, not "flow"
            gt = batch.get("flow", None)
            if gt is None and "flow_list" in batch:
                # flow_list is a list of tensors for consecutive pairs: [flow_01, flow_12...]. 
                # We validate on the first pair (0->1).
                # Depending on collate_fn, flow_list might be a list of batched tensors 
                # or a stacked tensor if custom collate isn't used. 
                # Default collate converts list of lists -> list of stacked tensors.
                flows = batch["flow_list"]
                if isinstance(flows, list) and len(flows) > 0:
                    gt = flows[0] # [B, 2, H, W]
                elif isinstance(flows, torch.Tensor):
                    gt = flows[:, 0] # if stacked [B, T-1, 2, H, W]
            
            if gt is not None:
                gt = gt.to(self.device, non_blocking=True)
            
            imgs = torch.cat([clip[:, 0], clip[:, 1]], dim=1)
            # Eval mode DDFlowNet returns full_res flow
            pred = self.model(imgs)
            
            if gt is not None:
                B, C, H, W = gt.shape
                if pred.shape[-2:] != (H, W):
                    pred = F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=True)
                    # Scale? Eval mode output expects to be roughly aligned.
                    # DDFlowNet_pt eval returns flow2_up_4x.
                    pass
                
                epe_map = torch.norm(pred - gt, dim=1)
                valid = batch.get("valid", torch.ones_like(epe_map).bool())
                if valid.shape[-2:] != (H, W):
                    valid = F.interpolate(valid.float().unsqueeze(1), size=(H, W), mode="nearest").squeeze(1).bool()
                
                valid_epe = epe_map[valid]
                if valid_epe.numel() > 0:
                     epe_list.append(valid_epe.mean().item())
                     p1_cnt += (valid_epe < 1.0).sum().item()
                     p3_cnt += (valid_epe < 3.0).sum().item()
                     total += valid_epe.numel()

        metrics = {}
        if epe_list: metrics["epe"] = np.mean(epe_list)
        if total > 0:
            metrics["1px"] = p1_cnt / total
            metrics["3px"] = p3_cnt / total
        
        print(f"[Epoch {epoch}] Val EPE: {metrics.get('epe', -1):.4f}")
        return metrics
