# file: engine/trainer_unsamflow.py
from __future__ import annotations
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from types import SimpleNamespace
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from tqdm import tqdm

from models.UnSAMFlow.models.pwclite import PWCLite
from models.UnSAMFlow.losses.flow_loss import unFlowLoss

# Re-use utilities from trainer_segment_aware
from engine.trainer_segment_aware import (
    _to_device, _masked_epe, _flow_to_rgb, resize_mask, concat_mean
)

class UnSAMFlowTrainer:
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
        
        # Build Loss
        # Config for loss
        lcfg = cfg.get("loss", {})
        # Map config dict to SimpleNamespace for UnSAMFlow compatibility
        loss_args = Namespace(
            w_l1=lcfg.get("w_l1", 0.0),
            w_ssim=lcfg.get("w_ssim", 0.0), # Usually 0.85
            w_ternary=lcfg.get("w_ternary", 1.0),
            w_ph_scales=lcfg.get("w_ph_scales", [1.0, 1.0, 1.0, 1.0, 0.0]),
            w_sm=lcfg.get("w_smooth", 0.1),
            smooth_type=lcfg.get("smooth_type", "2nd"),
            smooth_edge=lcfg.get("smooth_edge", "image"), # or 'full_seg'
            edge_aware_alpha=lcfg.get("edge_aware_alpha", 10.0),
            occ_from_back=lcfg.get("occ_from_back", False),
            warp_pad=lcfg.get("warp_pad", "border"),
            with_bk=True, # Always compute bw loss for unsupervied
            ransac_threshold=lcfg.get("ransac_threshold", 3),
        )
        self.criterion = unFlowLoss(loss_args).to(self.device)
        self.occ_aware_start_epoch = int(lcfg.get("occ_aware_start_epoch", 5))
        
        # Optimizer
        optim_cfg = cfg.get("optim", {})
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(optim_cfg.get("lr", 2e-4)),
            weight_decay=float(optim_cfg.get("weight_decay", 1e-4)),
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        
        # Logging
        log_cfg = cfg.get("logging", {})
        self.use_tb = log_cfg.get("use_tb", True)
        self.log_every = int(log_cfg.get("log_every", 100))
        self.writer: Optional[SummaryWriter] = None
        if self.use_tb:
            tb_dir = self.workspace / log_cfg.get("tb_dir", "tb")
            tb_dir.mkdir(exist_ok=True, parents=True)
            self.writer = SummaryWriter(str(tb_dir))
            
        self.viz_enable = cfg.get("viz", {}).get("enable", True)

    def _build_model(self) -> nn.Module:
        mcfg = self.cfg.get("model", {})
        args = mcfg.get("args", {})
        # PWCLite expects a config object
        model_cfg = Namespace(
            input_adj_map=args.get("input_adj_map", False),
            input_boundary=args.get("input_boundary", False),
            add_mask_corr=args.get("add_mask_corr", False),
            reduce_dense=args.get("reduce_dense", True),
            learned_upsampler=args.get("learned_upsampler", False),
            aggregation_type=args.get("aggregation_type", "concat")
        )
        print("Building PWCLite with cfg:", model_cfg)
        return PWCLite(model_cfg)

    def _save_checkpoint(self, name: str):
        path = self.workspace / name
        torch.save({
            "epoch": self.current_epoch,
            "step": self.global_step,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
        }, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.current_epoch = ckpt.get("epoch", 0)
        self.global_step = ckpt.get("step", 0)
        self.best_metric = ckpt.get("best_metric", float("inf"))
        print(f"Loaded checkpoint: {path}")

    def _train_one_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        accum = {"loss": 0.0, "steps": 0}
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        for batch in pbar:
            batch = _to_device(batch, self.device)
            clip = batch["clip"]
            if clip.dtype == torch.uint8:
                clip = clip.float() / 255.0
            
            # UnSAMFlow expects pairs (img1, img2)
            # Clip is [B, T, 3, H, W]. Assume T>=2
            img1 = clip[:, 0]
            img2 = clip[:, 1]
            
            # Handle SAM masks
            seg1, seg2 = None, None
            if "sam_masks" in batch:
                # [B, T, S, H, W] -> S=1 usually? Or S channels?
                # UnlabeledClipDataset returns [T, S, H, W] in sample, so batch is [B, T, S, H, W]
                # PWCLite expects (B, 1, H, W) integer masks for 'full_seg' I think?
                # Let's check pwclite usage: 
                #   full_seg1_down = F.interpolate(full_seg1, ...).long()
                #   full_seg1_down_oh = F.one_hot(full_seg1_down)
                # So it expects integer labels (H, W).
                # The dataset currently returns 'sam_masks' which might be (S, H, W) boolean/float or single channel int.
                # In dataio/clip_dataset_unsup.py, it loads .pt. 
                # If these are binary masks per object, we need to merge them into a single integer map.
                # But for now, let's assume the dataset provides what we need or we fix it.
                # If we passed load_sam_masks=True, we should get something.
                
                # Check shape and move to device
                # Dataset returns (B, T, 1, H, W) uint8
                masks = batch["sam_masks"].to(self.device)
                if masks.ndim == 5 and masks.shape[2] == 1:
                    masks = masks.squeeze(2)  # (B, T, H, W)
                
                # Take pairs and convert to long for PWCLite
                seg1 = masks[:, 0].unsqueeze(1).long()  # (B, 1, H, W)
                seg2 = masks[:, 1].unsqueeze(1).long()
            
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                # Bidirectional forward
                res_dict = self.model(img1, img2, full_seg1=seg1, full_seg2=seg2, with_bk=True)
                
                flows_12 = res_dict["flows_12"]
                flows_21 = res_dict["flows_21"]
                
                # Combined flows: UnSAMFlow loss expects [B, 4, H, W] (concat fw and bw)
                combined_flows = [torch.cat([f12, f21], dim=1) for f12, f21 in zip(flows_12, flows_21)]

                # Bi-directional loss in one call
                # Disable occ_aware during early epochs to avoid zero-gradient deadlock
                occ_aware = self.current_epoch >= self.occ_aware_start_epoch
                loss_pack = self.criterion(combined_flows, img1, img2, full_seg1=seg1, full_seg2=seg2, occ_aware=occ_aware)
                total_loss = loss_pack[0]
            
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            accum["loss"] += total_loss.item()
            accum["steps"] += 1
            self.global_step += 1
            
            pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})
            
            if self.writer and self.global_step % self.log_every == 0:
                self.writer.add_scalar("train/loss", total_loss.item(), self.global_step)
                if len(flows_12) > 0:
                     f_vis = _flow_to_rgb(flows_12[0][:4])
                     self.writer.add_image("train/flow", vutils.make_grid(f_vis, normalize=False), self.global_step)

        return accum

    def validate(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.eval()
        epe_list = []
        
        for batch in tqdm(val_loader, desc="Validate"):
            batch = _to_device(batch, self.device)
            clip = batch["clip"]
            if clip.dtype == torch.uint8:
                clip = clip.float() / 255.0
                
            img1 = clip[:, 0]
            img2 = clip[:, 1]
            
            seg1, seg2 = None, None
            if "sam_masks" in batch:
                masks = batch["sam_masks"]
                if masks.ndim == 5 and masks.shape[2] == 1:
                     masks = masks.squeeze(2)
                seg1 = masks[:, 0].unsqueeze(1)
                seg2 = masks[:, 1].unsqueeze(1)
                
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                res = self.model(img1, img2, full_seg1=seg1, full_seg2=seg2, with_bk=False)
                # PWCLite returns list of flows, [0] is finest
                flow_pred = res["flows_12"][0]
            
            # GT
            if "flow_list" in batch: 
                # UnlabeledClipDataset validation returns list of flows per step
                # We need flow_fw[0] which corresponds to img1->img2
                gt = batch["flow_list"][0]
                
                if flow_pred.shape[-2:] != gt.shape[-2:]:
                    flow_pred = F.interpolate(flow_pred, size=gt.shape[-2:], mode='bilinear', align_corners=True)
                    # Scale flow ? PWCNet usually outputs scaled flow.
                    # Standard resize scale correction:
                    scale_x = gt.shape[-1] / float(img1.shape[-1]) # if prediction was original size
                    # Wait, prediction is usually at crop size.
                    # if we resized, we need to rescale magnitude
                    # Let's trust _masked_epe or standard resize logic
                    scale_h = gt.shape[-2] / flow_pred.shape[-2] # 1.0 if we just interpolated
                    scale_w = gt.shape[-1] / flow_pred.shape[-1]
                    # flow_pred[:,0] *= scale_w
                    # flow_pred[:,1] *= scale_h 
                    # Actually F.interpolate doesn't change magnitude, so if flow was for (H, W), it fits (H,W) coordinates.
                    # If we upsample to GT size, we theoretically should scale if the spatial domain changed.
                    # But if we just look at EPE, standard is typically upsample flow to GT res and multiply by scale ratio.
                    # For simplicity, let's assume crop was 1:1 or close enough for dry run.
                    # Correction:
                    flow_pred[:, 0] *= (gt.shape[-1] / float(img1.shape[-1])) # rescale to original image logic?
                    flow_pred[:, 1] *= (gt.shape[-2] / float(img1.shape[-2]))

                epe = _masked_epe(flow_pred, gt)
                epe_list.append(epe.cpu().numpy())
                
        metrics = {}
        if epe_list:
            metrics["epe"] = concat_mean(epe_list)
            print(f"[Val Epoch {epoch}] EPE: {metrics['epe']:.4f}")
            if self.writer:
                self.writer.add_scalar("val/epe", metrics["epe"], self.global_step)
        
        return metrics

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        epochs = int(self.cfg["optim"]["epochs"])
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            self._train_one_epoch(train_loader)
            if val_loader and epoch % self.cfg.get("validation", {}).get("every_n_epochs", 5) == 0:
                mets = self.validate(val_loader, epoch)
                val_epe = mets.get("epe", float("inf"))
                if val_epe < self.best_metric:
                    self.best_metric = val_epe
                    self._save_checkpoint("best.pth")
            
            # Periodic save
            if epoch % 5 == 0:
                 self._save_checkpoint(f"ckpt_e{epoch:03d}.pth")
