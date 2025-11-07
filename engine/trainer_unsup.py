# file: engine/trainer_unsup.py
from __future__ import annotations
import math, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

from models.zoo import build_model
from losses.unsup_flow_losses import UnsupervisedFlowLoss
from utils.utils import InputPadder
from tqdm import tqdm

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


def _get_flow_pred(out) -> torch.Tensor:
    """Return final (B,2,H,W) flow tensor from common model outputs."""
    if torch.is_tensor(out):
        return out
    if isinstance(out, dict):
        if "flow" in out:
            return out["flow"]
        if "flows" in out and isinstance(out["flows"], (list, tuple)):
            return out["flows"][-1]
    if isinstance(out, (list, tuple)):
        last = out[-1]
        if torch.is_tensor(last):
            return last
        if isinstance(last, dict) and "flow" in last:
            return last["flow"]
    raise ValueError("Unsupported model output for flow prediction.")
# -----------------------------------------------------------------------------

def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def _pick(batch: Dict[str, torch.Tensor], *names) -> Optional[torch.Tensor]:
    for n in names:
        if n in batch:
            return batch[n]
    return None


def _masked_epe(pred: torch.Tensor, gt: Optional[torch.Tensor], valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    if gt is None:
        gt = torch.zeros_like(pred)
    epe_map = ((pred - gt).pow(2).sum(1).sqrt())  # (B,H,W)
    if valid is not None:
        if valid.dim() == 2:
            valid = valid.unsqueeze(0).expand_as(epe_map)
        eps = 1e-6
        epe = (epe_map * valid).flatten(1).sum(-1) / (valid.flatten(1).sum(-1) + eps)
    else:
        epe = epe_map.flatten(1).mean(-1)
    return epe  # (B,)


# ---------- Flow colorization helpers ----------
def _hsv_to_rgb_torch(h, s, v):
    """h,s,v in [0,1], shape (B,H,W). Returns (B,3,H,W) in [0,1]."""
    i = torch.floor(h * 6)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    i = i.to(torch.int64) % 6
    conds = [
        (v, t, p),
        (q, v, p),
        (p, v, t),
        (p, q, v),
        (t, p, v),
        (v, p, q),
    ]
    r = torch.stack([conds[j][0] for j in i.flatten()], 0).view_as(h)
    g = torch.stack([conds[j][1] for j in i.flatten()], 0).view_as(h)
    b = torch.stack([conds[j][2] for j in i.flatten()], 0).view_as(h)
    return torch.stack([r, g, b], dim=1)  # (B,3,H,W)


def _flow_to_rgb_t(flow: torch.Tensor) -> torch.Tensor:
    """
    flow: (B,2,H,W) -> (B,3,H,W) in [0,1]
    Try utils.flow_viz if available; fallback to HSV mapping.
    """
    try:
        from utils.flow_viz import flow_to_image as _flow_to_image_np
        imgs = []
        for f in flow.detach().permute(0, 2, 3, 1).cpu().numpy():
            im = _flow_to_image_np(f)  # HxWx3 uint8
            imgs.append(torch.from_numpy(im).permute(2, 0, 1))
        out = torch.stack(imgs).float() / 255.0
        return out.to(flow.device)
    except Exception:
        u, v = flow[:, 0], flow[:, 1]
        mag = torch.sqrt(u * u + v * v)
        ang = torch.atan2(v, u)  # [-pi, pi]
        h = (ang + math.pi) / (2 * math.pi)
        # normalize magnitude robustly (95th percentile per-batch)
        m95 = torch.quantile(mag.flatten(1), 0.95, dim=1).view(-1, 1, 1)
        v_ = torch.clamp(mag / (m95 + 1e-6), 0, 1)
        s = torch.ones_like(v_)
        return _hsv_to_rgb_torch(h, s, v_).clamp(0, 1)
# ------------------------------------------------


class UnsupervisedFlowTrainer:
    """
    Adds:
      - TensorBoard scalars + images
      - Save colorized flow PNGs during validate()
    """
    def __init__(self, args, cfg: Dict, workspace: Path):
        self.args = args
        self.cfg = cfg
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_scale = float(cfg.get("data", {}).get("input_scale", 1.0))

        # Model & loss
        self.model = build_model(self.args, cfg["model"]).to(self.device)
        self.model.train()
        self.crit = UnsupervisedFlowLoss(cfg["loss"]).to(self.device)

        # Optim & sched
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg["optim"]["lr"],
            weight_decay=cfg["optim"].get("weight_decay", 0.0),
        )
        if cfg["optim"].get("scheduler", {}).get("type", "cosine") == "cosine":
            T = cfg["optim"]["epochs"]
            self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt, T_max=T, eta_min=cfg["optim"]["scheduler"].get("min_lr", 1e-6)
            )
        else:
            self.sched = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.clip = cfg["optim"].get("clip", 1.0)
        self.gamma = float(cfg.get("loss", {}).get("iter_gamma", 0.8))
        self.w_epe_sup = float(cfg.get("loss", {}).get("w_epe_sup", 1.0))
        self.best_val = float("inf")
        self.global_step = 0

        # TensorBoard
        log_cfg = cfg.get("logging", {})
        self.use_tb = bool(log_cfg.get("use_tb", True))
        self.writer = None
        if self.use_tb:
            tb_dir = self.workspace / str(log_cfg.get("tb_dir", "tb"))
            tb_dir.mkdir(exist_ok=True, parents=True)
            self.writer = SummaryWriter(tb_dir.as_posix())

        # Viz options
        vz = cfg.get("viz", {})
        self.viz_enable = bool(vz.get("enable", True))
        self.viz_max = int(vz.get("max_samples", 8))
        self.viz_dir = self.workspace / str(vz.get("save_dir", "val_vis"))
        self.viz_dir.mkdir(exist_ok=True, parents=True)

    # ------- core fwd/loss -------
    def _forward(self, image1: torch.Tensor, image2: torch.Tensor) -> List[torch.Tensor]:
        out = self.model(image1, image2)
        return out if isinstance(out, list) else [out]

    def _loss_on_preds(self, preds: List[torch.Tensor], batch: Dict) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        total = 0.0
        log = {}
        im1 = _pick(batch, "image1", "img1") * self.input_scale
        im2 = _pick(batch, "image2", "img2") * self.input_scale

        for i, flow12 in enumerate(reversed(preds)):
            w = (self.gamma ** i)
            l, pieces = self.crit(im1, im2, flow12, extra={"iters_left": i})
            total = total + w * l
            for k, v in pieces.items():
                log[f"{k}_it{-i}"] = v.detach()
        log["loss_unsup"] = torch.as_tensor(total).detach()
        return total, log

    # ------- public APIs -------
    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        epochs = int(self.cfg["optim"]["epochs"])
        for ep in range(1, epochs + 1):
            t0 = time.time()
            tr = self._train_one_epoch(train_loader)

            val_epe = None
            if val_loader is not None:
                val_epe = self.validate(val_loader, epoch=ep)

            if self.sched is not None:
                self.sched.step()

            dt = time.time() - t0
            lr = self.opt.param_groups[0]["lr"]
            tr_loss = tr["loss"] / max(1, tr["steps"])
            print(f"[Epoch {ep:03d}/{epochs}] "
                  f"train_loss={tr_loss:.4f} "
                  f"val_aepe={None if val_epe is None else round(val_epe, 4)} "
                  f"lr={lr:.2e} time={dt:.1f}s")

            if self.writer:
                self.writer.add_scalar("train/loss", tr_loss, ep)
                if val_epe is not None:
                    self.writer.add_scalar("val/aepe", val_epe, ep)
                self.writer.add_scalar("train/lr", lr, ep)

            if ep % self.cfg["ckpt"]["save_every"] == 0:
                self._save(ep, val_epe)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        device = self.device

        # Accumulate per-pixel EPE values into lists (numpy) then compute means at the end
        epe_all_list    = []
        epe_occ_list    = []  # occ == 0 (occluded)
        epe_nonocc_list = []  # occ == 1 (non-occluded)
        epe_line_list   = []  # flat == 0
        epe_flat_list   = []  # flat > 0
        epe_s10_list    = []  # |flow| <= 10
        epe_s1050_list  = []  # 10 < |flow| <= 50
        epe_s50_list    = []  # |flow| > 50

        # Also track <1px/<3px/<5px rates on valid pixels
        thr1_cnt = thr3_cnt = thr5_cnt = 0
        valid_cnt_total = 0

        for batch in self.val_loader:
            # ----- fetch & pad -----
            img1 = batch.get("image1", batch.get("img1")).to(device, non_blocking=True)
            img2 = batch.get("image2", batch.get("img2")).to(device, non_blocking=True)
            flow_gt = batch["flow"].to(device, non_blocking=True)  # must exist for val

            # optional padding for RAFT-like models
            try:
                padder = InputPadder(img1.shape)
                img1_p, img2_p = padder.pad(img1, img2)
                out = self.model(img1_p, img2_p)
                flow_pred = _get_flow_pred(out)
                flow_pred = padder.unpad(flow_pred)
            except Exception:
                # fall back if your model already handles padding
                out = self.model(img1, img2)
                flow_pred = _get_flow_pred(out)

            # ----- per-pixel EPE and masks -----
            epe_map = torch.norm(flow_pred - flow_gt, dim=1)  # (B,H,W)
            mag_map = torch.norm(flow_gt, dim=1)               # (B,H,W)

            B, H, W = epe_map.shape
            valid = batch.get("valid", None)
            if valid is None:
                valid = torch.ones((B, H, W), device=device, dtype=torch.bool)
            else:
                valid = valid.to(device).bool()
                if valid.dim() == 2:  # (H,W) -> (B,H,W)
                    valid = valid.unsqueeze(0).expand(B, -1, -1)
                if valid.dim() == 4 and valid.size(1) == 1:
                    valid = valid[:, 0].bool()

            occ = batch.get("occ", None)   # 0 = occluded, 1 = non-occluded (AnimeRun)
            if occ is not None:
                occ = occ.to(device)
                if occ.dim() == 2:
                    occ = occ.unsqueeze(0).expand(B, -1, -1)
                if occ.dim() == 4 and occ.size(1) == 1:
                    occ = occ[:, 0]

            flat = batch.get("flat", None)  # >0 = flat region, 0 = line/edge
            if flat is not None:
                flat = flat.to(device)
                if flat.dim() == 2:
                    flat = flat.unsqueeze(0).expand(B, -1, -1)
                if flat.dim() == 4 and flat.size(1) == 1:
                    flat = flat[:, 0]

            # ----- collect overall -----
            epe_all_list.append(epe_map[valid].detach().cpu().numpy())
            valid_cnt = int(valid.sum().item())
            valid_cnt_total += valid_cnt
            if valid_cnt > 0:
                v_epe = epe_map[valid]
                thr1_cnt += int((v_epe < 1.0).sum().item())
                thr3_cnt += int((v_epe < 3.0).sum().item())
                thr5_cnt += int((v_epe < 5.0).sum().item())

            # ----- occ/nonocc -----
            if occ is not None:
                occ_mask = (occ == 1) & valid
                occed_mask = (occ == 0) & valid
                if occed_mask.any():
                    epe_occ_list.append(epe_map[occed_mask].detach().cpu().numpy())
                if occ_mask.any():
                    epe_nonocc_list.append(epe_map[occ_mask].detach().cpu().numpy())

            # ----- flat/line -----
            if flat is not None:
                flat_mask = (flat > 0) & valid
                line_mask = (flat == 0) & valid
                if flat_mask.any():
                    epe_flat_list.append(epe_map[flat_mask].detach().cpu().numpy())
                if line_mask.any():
                    epe_line_list.append(epe_map[line_mask].detach().cpu().numpy())

            # ----- motion bins by |flow_gt| -----
            bin1 = (mag_map <= 10.0) & valid
            bin2 = (mag_map > 10.0) & (mag_map <= 50.0) & valid
            bin3 = (mag_map > 50.0) & valid
            if bin1.any():
                epe_s10_list.append(epe_map[bin1].detach().cpu().numpy())
            if bin2.any():
                epe_s1050_list.append(epe_map[bin2].detach().cpu().numpy())
            if bin3.any():
                epe_s50_list.append(epe_map[bin3].detach().cpu().numpy())

        # ========================== aggregate ==========================
        metrics = {}
        if epe_all_list:
            all_np = np.concatenate(epe_all_list)
            metrics["epe"] = float(np.mean(all_np))
            metrics["1px"] = float(thr1_cnt / max(1, valid_cnt_total))
            metrics["3px"] = float(thr3_cnt / max(1, valid_cnt_total))
            metrics["5px"] = float(thr5_cnt / max(1, valid_cnt_total))
        else:
            metrics["epe"] = float("nan")
            metrics["1px"] = metrics["3px"] = metrics["5px"] = float("nan")

        # breakdowns
        metrics["epe_occ"]    = concat_mean(epe_occ_list)
        metrics["epe_nonocc"] = concat_mean(epe_nonocc_list)
        metrics["epe_line"]   = concat_mean(epe_line_list)
        metrics["epe_flat"]   = concat_mean(epe_flat_list)
        metrics["epe_s<10"]   = concat_mean(epe_s10_list)
        metrics["epe_s10-50"] = concat_mean(epe_s1050_list)
        metrics["epe_s>50"]   = concat_mean(epe_s50_list)

        # -------- log & save best ----------
        self.logger.log_scalars(metrics, self.global_step, prefix="val")

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

    # ------- internals -------
    def _train_one_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        accum = {"loss": 0.0, "steps": 0}
        for batch in train_loader:
            batch = _to_device(batch, self.device)
            i1 = _pick(batch, "image1", "img1")
            i2 = _pick(batch, "image2", "img2")

            self.opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=True):
                preds = self._forward(i1, i2)
                loss_unsup, _ = self._loss_on_preds(preds, batch)
                if "flow" in batch:
                    epe_b = _masked_epe(preds[-1], batch["flow"], batch.get("valid"))
                    loss = loss_unsup + self.w_epe_sup * epe_b.mean()
                else:
                    loss = loss_unsup

            self.scaler.scale(loss).backward()
            if self.clip is not None:
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.scaler.step(self.opt)
            self.scaler.update()

            accum["loss"] += float(loss.detach().item())
            accum["steps"] += 1
