from __future__ import annotations
import math
import time
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


# ------------------------------ utils ------------------------------ #
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


# ---------- Flow colorization helpers ---------- #
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


# ========================= Trainer ========================= #
class UnsupervisedFlowTrainer:
    """
    Unsupervised optical-flow trainer with:
      - Mixed precision (optional: auto-enabled on CUDA)
      - TensorBoard scalars & images
      - Validation with detailed breakdown metrics
      - Optional PNG export of colorized flow
      - Robust handling of model outputs/padding/types
    """

    def __init__(self, args, cfg: Dict, workspace: Path):
        self.args = args
        self.cfg = cfg
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_scale = float(cfg.get("data", {}).get("input_scale", 1.0))

        # Reproducibility (optional)
        seed = int(cfg.get("optim", {}).get("seed", 0))
        if seed:
            import random
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Model & loss
        self.model = build_model(self.args, cfg["model"]).to(self.device)
        self.model.train()
        self.crit = UnsupervisedFlowLoss(cfg["loss"]).to(self.device)

        # Optimizer
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg["optim"]["lr"],
            weight_decay=cfg["optim"].get("weight_decay", 0.0),
        )

        # Scheduler config is kept; actual scheduler may be created in fit() if per-batch is required
        self.sched = None
        self.sched_cfg = cfg["optim"].get("scheduler", {"type": "cosine"})

        # AMP & training constants
        amp_enabled = torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        self.clip = cfg["optim"].get("clip", 1.0)
        self.gamma = float(cfg.get("loss", {}).get("iter_gamma", 0.8))
        self.w_epe_sup = float(cfg.get("loss", {}).get("w_epe_sup", 1.0))
        self.best_val = float("inf")
        self.best_epe = float("inf")
        self.global_step = 0

        # Gradient accumulation
        self.accum_steps = int(cfg.get("optim", {}).get("accum_steps", 1))
        assert self.accum_steps >= 1

        # TensorBoard
        log_cfg = cfg.get("logging", {})
        self.use_tb = bool(log_cfg.get("use_tb", True))
        self.writer: Optional[SummaryWriter] = None
        if self.use_tb:
            tb_dir = self.workspace / str(log_cfg.get("tb_dir", "tb"))
            tb_dir.mkdir(exist_ok=True, parents=True)
            self.writer = SummaryWriter(tb_dir.as_posix())

        # Visualization options
        vz = cfg.get("viz", {})
        self.viz_enable = bool(vz.get("enable", True))
        self.viz_max = int(vz.get("max_samples", 8))
        self.viz_dir = self.workspace / str(vz.get("save_dir", "val_vis"))
        self.viz_dir.mkdir(exist_ok=True, parents=True)

    # ------- core forward/loss -------
    def _forward(self, image1: torch.Tensor, image2: torch.Tensor) -> List[torch.Tensor]:
        out = self.model(image1, image2)
        return out if isinstance(out, list) else [out]

    def _loss_on_preds(self, preds: List[torch.Tensor], batch: Dict) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        total = 0.0
        log: Dict[str, torch.Tensor] = {}
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

    # ------- checkpoints -------
    def _save(self, epoch: int, val_epe: Optional[float]):
        ckpt = {
            "epoch": epoch,
            "step": self.global_step,
            "state_dict": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "sched": self.sched.state_dict() if self.sched else None,
            "best_epe": getattr(self, "best_epe", float("inf")),
            "cfg": self.cfg,
        }
        path = self.workspace / f"ckpt_e{epoch:03d}.pth"
        torch.save(ckpt, path)

    def save(self, name: str, payload: Dict):
        path = self.workspace / name
        torch.save(payload, path)

    # ------- public APIs -------
    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        epochs = int(self.cfg["optim"]["epochs"])  # total epochs

        # Create scheduler now that we know loader length (if needed)
        sc = self.sched_cfg
        sc_type = sc.get("type", "cosine").lower()
        sc_per_batch = bool(sc.get("per_batch", False))
        if sc_type == "cosine":
            if sc_per_batch:
                T = epochs * max(1, len(train_loader))
                self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.opt, T_max=T, eta_min=sc.get("min_lr", 1e-6)
                )
            else:
                T = epochs
                self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.opt, T_max=T, eta_min=sc.get("min_lr", 1e-6)
                )
        elif sc_type == "onecycle":
            self.sched = torch.optim.lr_scheduler.OneCycleLR(
                self.opt,
                max_lr=self.cfg["optim"]["lr"],
                steps_per_epoch=max(1, len(train_loader)),
                epochs=epochs,
                pct_start=float(sc.get("pct_start", 0.05)),
                anneal_strategy=sc.get("anneal_strategy", "cos"),
                div_factor=float(sc.get("div_factor", 25.0)),
                final_div_factor=float(sc.get("final_div_factor", 1e4)),
            )
            sc_per_batch = True  # OneCycle must step per batch
        else:
            self.sched = None

        for ep in range(1, epochs + 1):
            t0 = time.time()
            tr = self._train_one_epoch(train_loader, step_scheduler_per_batch=sc_per_batch)

            val_epe = None
            if val_loader is not None:
                self.val_loader = val_loader
                metrics = self.validate(epoch=ep)
                val_epe = metrics["epe"]

            # step scheduler per-epoch if configured that way
            if self.sched is not None and not sc_per_batch:
                self.sched.step()

            dt = time.time() - t0
            lr = self.opt.param_groups[0]["lr"]
            tr_loss = tr["loss"] / max(1, tr["steps"])
            print(
                f"[Epoch {ep:03d}/{epochs}] "
                f"train_loss={tr_loss:.4f} "
                f"val_aepe={None if val_epe is None else round(val_epe, 4)} "
                f"lr={lr:.2e} time={dt:.1f}s"
            )

            if self.writer:
                self.writer.add_scalar("train/loss", tr_loss, ep)
                if val_epe is not None:
                    self.writer.add_scalar("val/aepe", val_epe, ep)
                self.writer.add_scalar("train/lr", lr, ep)

            if ep % int(self.cfg.get("ckpt", {}).get("save_every", 1)) == 0:
                self._save(ep, val_epe)

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        device = self.device

        # Accumulate per-pixel EPE values into lists then compute means at the end
        epe_all_list: List[np.ndarray] = []
        epe_occ_list: List[np.ndarray] = []      # occ == 0 (occluded)
        epe_nonocc_list: List[np.ndarray] = []   # occ == 1 (non-occluded)
        epe_line_list: List[np.ndarray] = []     # flat == 0
        epe_flat_list: List[np.ndarray] = []     # flat > 0
        epe_s10_list: List[np.ndarray] = []      # |flow| <= 10
        epe_s1050_list: List[np.ndarray] = []    # 10 < |flow| <= 50
        epe_s50_list: List[np.ndarray] = []      # |flow| > 50

        # Also track <1px/<3px/<5px rates on valid pixels
        thr1_cnt = thr3_cnt = thr5_cnt = 0
        valid_cnt_total = 0

        for batch in tqdm(self.val_loader):
            # ----- fetch & normalize dtype -----
            img1 = batch.get("image1", batch.get("img1")).to(device, non_blocking=True)
            img2 = batch.get("image2", batch.get("img2")).to(device, non_blocking=True)
            if img1.dtype == torch.uint8:
                img1 = img1.float() / 255.0
            if img2.dtype == torch.uint8:
                img2 = img2.float() / 255.0

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
            mag_map = torch.norm(flow_gt, dim=1)              # (B,H,W)

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

            # ----- optional visualization to TB (first few samples) -----
            if self.writer and self.viz_enable:
                with torch.no_grad():
                    # Clamp for robustness before colorization
                    flow_rgb = _flow_to_rgb_t(flow_pred[: self.viz_max].clamp(-500, 500))
                    grid_img1 = vutils.make_grid(
                        img1[: self.viz_max], nrow=min(4, self.viz_max), normalize=True
                    )
                    grid_img2 = vutils.make_grid(
                        img2[: self.viz_max], nrow=min(4, self.viz_max), normalize=True
                    )
                    grid_flow = vutils.make_grid(flow_rgb, nrow=min(4, self.viz_max))
                    self.writer.add_image("val/img1", grid_img1, self.global_step)
                    self.writer.add_image("val/img2", grid_img2, self.global_step)
                    self.writer.add_image("val/flow_rgb", grid_flow, self.global_step)

                # Save PNGs (optional) for external inspection
                try:
                    bsave = min(self.viz_max, flow_rgb.size(0))
                    for b in range(bsave):
                        vutils.save_image(
                            flow_rgb[b],
                            self.viz_dir / f"epoch{epoch:03d}_step{self.global_step:08d}_b{b}.png",
                        )
                except Exception:
                    pass

        # ========================== aggregate ==========================
        metrics: Dict[str, float] = {}
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
        metrics["epe_occ"] = concat_mean(epe_occ_list)
        metrics["epe_nonocc"] = concat_mean(epe_nonocc_list)
        metrics["epe_line"] = concat_mean(epe_line_list)
        metrics["epe_flat"] = concat_mean(epe_flat_list)
        metrics["epe_s<10"] = concat_mean(epe_s10_list)
        metrics["epe_s10-50"] = concat_mean(epe_s1050_list)
        metrics["epe_s>50"] = concat_mean(epe_s50_list)

        # -------- log & save best ----------
        if self.writer:
            for k, v in metrics.items():
                self.writer.add_scalar(f"val/{k}", v, self.global_step)

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
    def _train_one_epoch(self, train_loader: DataLoader, step_scheduler_per_batch: bool = False) -> Dict[str, float]:
        self.model.train()
        accum = {"loss": 0.0, "steps": 0}

        # Zero grad once if using gradient accumulation
        self.opt.zero_grad(set_to_none=True)

        for ib, batch in enumerate(tqdm(train_loader)):
            batch = _to_device(batch, self.device)
            i1 = _pick(batch, "image1", "img1")
            i2 = _pick(batch, "image2", "img2")

            # Ensure floating 0..1 if input came as uint8
            if i1.dtype == torch.uint8:
                i1 = i1.float() / 255.0
            if i2.dtype == torch.uint8:
                i2 = i2.float() / 255.0

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                preds = self._forward(i1, i2)
                loss_unsup, _ = self._loss_on_preds(preds, batch)
                if "flow" in batch:
                    epe_b = _masked_epe(preds[-1], batch["flow"], batch.get("valid"))
                    loss = loss_unsup + self.w_epe_sup * epe_b.mean()
                else:
                    loss = loss_unsup

            # Scale loss if accumulating
            loss_to_backprop = loss / self.accum_steps
            self.scaler.scale(loss_to_backprop).backward()

            # Step when accumulation boundary is reached
            do_step = ((ib + 1) % self.accum_steps == 0)
            if do_step:
                if self.clip is not None:
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad(set_to_none=True)

            if step_scheduler_per_batch and self.sched is not None and do_step:
                self.sched.step()

            self.global_step += 1
            accum["loss"] += float(loss.detach().item())
            accum["steps"] += 1

            # Optional: write a few training scalars per N steps
            if self.writer and (self.global_step % int(self.cfg.get("logging", {}).get("log_every", 200)) == 0):
                self.writer.add_scalar("train/loss_step", float(loss.detach().item()), self.global_step)
                lr_now = self.opt.param_groups[0]["lr"]
                self.writer.add_scalar("train/lr_step", lr_now, self.global_step)

        return accum
