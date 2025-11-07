from typing import Dict
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataio.registry import get_dataset_builder
from models.registry import build_model
from losses.unsup_flow_losses import multiscale_loss
from metrics.flow_metrics import *
from utils.loggers import TBLogger
from utils.misc import Saver
from utils.schedule import cosine_lr
from utils.utils import InputPadder
class Trainer:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.saver = Saver(cfg["work_dir"])
        self.logger = TBLogger(self.saver.path("tb"))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # datasets
        build_ds = get_dataset_builder(cfg["dataset"]["name"])
        self.ds_train = build_ds(cfg, split='train')
        self.ds_val = build_ds(cfg, split='val')

        bs = cfg["trainer"]["batch_size"]
        nw = cfg["trainer"]["num_workers"]
        self.train_loader = DataLoader(self.ds_train, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True)
        self.val_loader = DataLoader(self.ds_val, batch_size=bs, shuffle=False, num_workers=nw)

        # model
        self.model = build_model(cfg).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg["opt"]["lr"], weight_decay=cfg["opt"]["weight_decay"])

        self.scaler = torch.cuda.amp.GradScaler(enabled=(cfg["trainer"]["precision"]=="fp16"))
        self.global_step = 0


    def save(self, name, state):
        torch.save(state, self.saver.path(name))

    def train(self):
        epochs = self.cfg["trainer"]["epochs"]
        wts = self.cfg["loss"]["multi_scale_weights"]
        warmup = self.cfg["opt"]["warmup_steps"]
        total_steps = epochs * len(self.train_loader)

        for ep in range(epochs):
            for batch in self.train_loader:
                self.model.train()
                img1 = batch["image1"].to(self.device, non_blocking=True)
                img2 = batch["image2"].to(self.device, non_blocking=True)
                flow = batch["flow"].to(self.device, non_blocking=True)

                # schedule
                lr = cosine_lr(self.global_step, total_steps, self.cfg["opt"]["lr"], warmup)
                for g in self.opt.param_groups:
                    g['lr'] = lr

                self.opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(self.cfg["trainer"]["precision"]=="fp16")):
                    outputs = self.model(img1, img2)
                    loss = multiscale_loss(outputs, flow, wts)

                self.scaler.scale(loss).backward()
                if self.cfg["opt"]["grad_clip"]:
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["opt"]["grad_clip"])
                self.scaler.step(self.opt)
                self.scaler.update()

                self.global_step += 1
                if self.global_step % self.cfg["trainer"]["log_every"] == 0:
                    self.logger.log_scalars({"loss": float(loss), "lr": lr}, self.global_step, prefix="train")

                if self.global_step % self.cfg["trainer"]["val_every"] == 0:
                    self.validate()
                if self.global_step % self.cfg["trainer"]["save_every"] == 0:
                    self.save("last.pth", {"step": self.global_step, "state_dict": self.model.state_dict()})

        # final save
        self.validate()
        self.save("final.pth", {"step": self.global_step, "state_dict": self.model.state_dict()})

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        device = self.device
        metrics_accum = {"epe": 0.0, "1px": 0.0, "3px": 0.0, "5px": 0.0}
        n = 0
        epe_all_list   = []
        epe_occ_list   = []
        epe_nonocc_list= []
        epe_line_list  = []
        epe_flat_list  = []
        epe_s10_list   = []
        epe_s1050_list = []
        epe_s50_list   = []
        for batch in self.val_loader:
            img1 = batch["image1"].to(self.device, non_blocking=True)
            self.padder = InputPadder(img1.shape)
            img2 = batch["image2"].to(self.device, non_blocking=True)
            flow_gt = batch["flow"].to(self.device, non_blocking=True)
            B, _, H, W = flow_gt.shape
            out = self.model(img1, img2)
            flow_pred = out["flow"]

            epe_map = torch.norm(flow_pred - flow_gt, dim=1)         # [B,H,W]
            mag     = torch.norm(flow_gt, dim=1) 

            valid = batch.get("valid", torch.ones((B, H, W), device=device, dtype=torch.bool)).bool()
            occ   = batch.get("occ",   None)  # 1 = non-occluded, 0 = occluded (AnimeRun)
            flat  = batch.get("flat",  None)  # >0 = flat, 0 = line

            epe_all_list.append(epe_map[valid].detach().cpu().numpy())

            if occ is not None:
                occ = occ.to(device)
                epe_occ    = epe_map[(occ == 0) & valid]
                epe_nonocc = epe_map[(occ == 1) & valid]
                if epe_occ.numel()    > 0: epe_occ_list.append(epe_occ.detach().cpu().numpy())
                if epe_nonocc.numel() > 0: epe_nonocc_list.append(epe_nonocc.detach().cpu().numpy())

            if flat is not None:
                flat = flat.to(device)
                epe_flat = epe_map[(flat > 0) & valid]
                epe_line = epe_map[(flat == 0) & valid]
                if epe_flat.numel() > 0: epe_flat_list.append(epe_flat.detach().cpu().numpy())
                if epe_line.numel() > 0: epe_line_list.append(epe_line.detach().cpu().numpy())

            # motion bins theo |flow_gt|
            epe_s10   = epe_map[(mag <= 10.0) & valid]
            epe_s1050 = epe_map[(mag > 10.0) & (mag <= 50.0) & valid]
            epe_s50   = epe_map[(mag > 50.0) & valid]
            if epe_s10.numel()   > 0: epe_s10_list.append(epe_s10.detach().cpu().numpy())
            if epe_s1050.numel() > 0: epe_s1050_list.append(epe_s1050.detach().cpu().numpy())
            if epe_s50.numel()   > 0: epe_s50_list.append(epe_s50.detach().cpu().numpy())

            # ========================== aggregate & log ==========================
            metrics = {}
            if epe_all_list:
                epe_all_np = np.concatenate(epe_all_list)
                metrics["epe"] = float(np.mean(epe_all_np))
                metrics["1px"] = float(np.mean(epe_all_np < 1))
                metrics["3px"] = float(np.mean(epe_all_np < 3))
                metrics["5px"] = float(np.mean(epe_all_np < 5))
            else:
                metrics["epe"] = metrics["1px"] = metrics["3px"] = metrics["5px"] = float("nan")

            # breakdowns (có thể là NaN nếu mask không có trong loader)
            metrics["epe_occ"]     = concat_mean(epe_occ_list)
            metrics["epe_nonocc"]  = concat_mean(epe_nonocc_list)
            metrics["epe_line"]    = concat_mean(epe_line_list)
            metrics["epe_flat"]    = concat_mean(epe_flat_list)
            metrics["epe_s<10"]    = concat_mean(epe_s10_list)
            metrics["epe_s10-50"]  = concat_mean(epe_s1050_list)
            metrics["epe_s>50"]    = concat_mean(epe_s50_list)

            # log tất cả metrics với prefix 'val'
            self.logger.log_scalars(metrics, self.global_step, prefix="val")

            # lưu best theo EPE
            if not hasattr(self, "best_epe"):
                self.best_epe = float("inf")
            if np.isfinite(metrics["epe"]) and metrics["epe"] < self.best_epe:
                self.best_epe = metrics["epe"]
                self.save(
                    "best.pth",
                    {
                        "step": self.global_step,
                        "state_dict": self.model.state_dict(),
                        "best_epe": self.best_epe,
                    },
                )

            return metrics