from __future__ import annotations
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from tqdm import tqdm

# Optional: your legacy builder for pair-based models
try:
    from models.zoo import build_model as _build_legacy_model
except Exception:
    _build_legacy_model = None

from models import AniFlowFormerT, ModelConfig as AFConfig
from models import AFTLosses

torch.autograd.set_detect_anomaly(False)


# ------------------------------ small utils ------------------------------ #
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
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, (list, tuple)) and v and torch.is_tensor(v[0]):
            out[k] = [t.to(device, non_blocking=True) for t in v]  # list of tensors (e.g., GT flows per pair)
        else:
            out[k] = v
    return out

def _masked_epe(pred: torch.Tensor, gt: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute AEPE per-batch with optional valid mask. Shape: (B,2,H,W)."""
    epe_map = torch.norm(pred - gt, dim=1)  # (B,H,W)
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
        # robust magnitude normalization
        m95 = torch.quantile(mag.flatten(1), 0.95, dim=1).view(-1, 1, 1)
        v_ = torch.clamp(mag / (m95 + 1e-6), 0, 1)
        s = torch.ones_like(v_)
        return _hsv_to_rgb_torch(h, s, v_).clamp(0, 1)

# ========================= Trainer (clip-based) ========================= #
class UnsupervisedClipTrainer:
    """
    Un/Semi-supervised trainer for clip-based optical flow (AniFlowFormer-T).
    - Mixed precision + gradient accumulation
    - Flexible loss: model.unsup_loss(...) or AFT Losses fallback
    - Validation with AEPE if GT exists; else proxy-loss evaluation
    - TensorBoard scalars/images + PNG flow dumps
    """

    def __init__(self, args, cfg: Dict, workspace: Path):
        self.args = args
        self.cfg = cfg
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_metric = float("inf")  # minimizes (AEPE if available, else total loss)
        self.global_step = 0

        # Reproducibility
        seed = int(cfg.get("optim", {}).get("seed", 0))
        if seed:
            import random
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # ------------------ Build model ------------------ #
        self.model = self._build_model_from_cfg(cfg["model"]).to(self.device)
        self.model.train()

        # Loss fallback (for folderized version)
        self.loss_mod = AFTLosses()

        # ------------------ Optimizer/Scheduler ------------------ #
        o = cfg["optim"]
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(o["lr"]),
            weight_decay=float(o.get("weight_decay", 1e-4)),
            betas=o.get("betas", (0.9, 0.999)),
        )
        self.sched = None
        self.sched_cfg = o.get("scheduler", {"type": "cosine", "per_batch": False})

        # AMP & grad clip/accum
        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        self.clip = float(o.get("clip", 1.0))
        self.accum_steps = int(o.get("accum_steps", 1))
        assert self.accum_steps >= 1

        # Logging
        log_cfg = cfg.get("logging", {})
        self.use_tb = bool(log_cfg.get("use_tb", True))
        self.writer: Optional[SummaryWriter] = None
        if self.use_tb:
            tb_dir = self.workspace / str(log_cfg.get("tb_dir", "tb"))
            tb_dir.mkdir(exist_ok=True, parents=True)
            self.writer = SummaryWriter(tb_dir.as_posix())

        # Visualization
        vz = cfg.get("viz", {})
        self.viz_enable = bool(vz.get("enable", True))
        self.viz_max = int(vz.get("max_samples", 4))
        self.viz_dir = self.workspace / str(vz.get("save_dir", "val_vis"))
        self.viz_dir.mkdir(exist_ok=True, parents=True)

        # Optional semi-supervised term weight (EPE on available GT)
        self.w_epe_sup = float(cfg.get("loss", {}).get("w_epe_sup", 0.0))

        # SAM guidance default
        self.use_sam_default = bool(cfg["model"].get("args", {}).get("use_sam", False))

    # ------------------ Model builder ------------------ #
    def _build_model_from_cfg(self, mcfg: Dict) -> torch.nn.Module:
        name = mcfg.get("name", "")
        args = mcfg.get("args", {})
        if name.lower() in {"aniflowformer-t", "aniflowformer_t", "aniflowformert"}:
            return AniFlowFormerT(AFConfig(**args))
        # fallback to legacy builder
        if _build_legacy_model is not None:
            return _build_legacy_model(self.args, mcfg)
        raise RuntimeError("No valid model builder found for config: {}".format(name))

    # ------------------ Loss wrapper ------------------ #
    def _compute_unsup_losses(self, clip: torch.Tensor, out: Dict[str, List[torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Return a dict with keys: total, photo, smooth, temporal, cycle (when available).
        """
        # flows = out["flows"]
        d = self.loss_mod.unsup_loss(clip, out)
        # Ensure tensors (not floats) for backward
        return {k: (v if torch.is_tensor(v) else torch.as_tensor(v, device=clip.device)) for k, v in d.items()}

    # ------------------ Public APIs ------------------ #
    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        epochs = int(self.cfg["optim"]["epochs"])

        # Scheduler (some policies need loader length)
        self._init_scheduler(train_loader, epochs)

        for ep in range(1, epochs + 1):
            t0 = time.time()
            tr = self._train_one_epoch(train_loader, step_scheduler_per_batch=self._sched_per_batch)
            val_metric = None
            if val_loader is not None:
                metrics = self.validate(val_loader, epoch=ep)
                # prefer AEPE if available, else total proxy loss
                val_metric = metrics.get("epe", None)
                if val_metric is None or not np.isfinite(val_metric):
                    val_metric = metrics.get("total", float("inf"))

            # Step per-epoch scheduler if configured that way
            if self.sched is not None and not self._sched_per_batch:
                self.sched.step()

            dt = time.time() - t0
            lr = self.opt.param_groups[0]["lr"]
            tr_loss = tr["loss"] / max(1, tr["steps"])

            print(
                f"[Epoch {ep:03d}/{epochs}] "
                f"train_loss={tr_loss:.4f} "
                f"val_metric={None if val_metric is None else round(val_metric, 4)} "
                f"lr={lr:.2e} time={dt:.1f}s"
            )

            if self.writer:
                self.writer.add_scalar("train/loss_epoch", tr_loss, ep)
                if val_metric is not None:
                    self.writer.add_scalar("val/metric", val_metric, ep)
                self.writer.add_scalar("train/lr_epoch", lr, ep)

            # Save best
            if val_metric is not None and np.isfinite(val_metric) and val_metric < self.best_metric:
                self.best_metric = val_metric
                self._save("best.pth", {"step": self.global_step, "state_dict": self.model.state_dict(), "best": self.best_metric})

            # Periodic checkpoint
            if (ep % int(self.cfg.get("ckpt", {}).get("save_every", 1))) == 0:
                self._save(f"ckpt_e{ep:03d}.pth", {
                    "epoch": ep,
                    "step": self.global_step,
                    "state_dict": self.model.state_dict(),
                    "opt": self.opt.state_dict(),
                    "sched": self.sched.state_dict() if self.sched else None,
                    "best": self.best_metric,
                    "cfg": self.cfg,
                })

    @torch.no_grad()
    def validate(self, val_loader: DataLoader, epoch: int = 0, fast: bool = False) -> Dict[str, float]:
        self.model.eval()
        device = self.device

        # Tổng hợp an toàn cho nhiều kích thước
        epe_sum = 0.0
        valid_pix = 0
        thr1_cnt = thr3_cnt = thr5_cnt = 0

        proxy_losses = []
        did_viz = False

        for bidx, batch in enumerate(tqdm(val_loader, desc="Validate")):
            batch = _to_device(batch, device)
            clip = batch["clip"]  # (B,T,3,H,W)
            sam_masks = batch.get("sam_masks", None)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = self.model(clip, sam_masks=sam_masks, use_sam=self.use_sam_default)
                losses = self._compute_unsup_losses(clip, out)

            # GT có thể là Tensor (1 pair) hoặc list[Tensor] (mỗi pair)
            gt_any = batch.get("flow", None)
            if gt_any is None:
                gt_any = batch.get("flow_list", None)

            # valid mask có thể là Tensor (áp cho mọi pair) hoặc list[Tensor] tương ứng từng pair
            valid_any = batch.get("valid", None)

            if gt_any is not None:
                pred_list = out["flows"]
                gt_list = [gt_any] if torch.is_tensor(gt_any) else gt_any
                L = min(len(pred_list), len(gt_list))

                for k in range(L):
                    pred = pred_list[k]          # (B,2,hp,wp)
                    gt   = gt_list[k]            # (B,2,hg,wg)

                    # Resize + scale theo từng trục
                    if pred.shape[-2:] != gt.shape[-2:]:
                        hp, wp = pred.shape[-2:]
                        hg, wg = gt.shape[-2:]
                        pred = F.interpolate(pred, size=(hg, wg), mode="bilinear", align_corners=True)
                        sx = wg / float(wp)
                        sy = hg / float(hp)
                        pred[:, 0] *= sx  # u theo W
                        pred[:, 1] *= sy  # v theo H

                    epe_map = torch.norm(pred - gt, dim=1)  # (B,hg,wg)

                    # chọn valid mask
                    if valid_any is None:
                        valid = torch.ones_like(epe_map, dtype=torch.bool)
                    else:
                        if torch.is_tensor(valid_any):
                            valid = valid_any
                        else:
                            valid = valid_any[k]
                        # broadcast nếu cần
                        if valid.dim() == 2:
                            valid = valid.unsqueeze(0).expand_as(epe_map)
                        elif valid.dim() == 3 and valid.size(0) == 1:
                            valid = valid.expand_as(epe_map)
                        valid = valid.bool()

                    # cộng dồn an toàn
                    e = (epe_map * valid).sum().item()
                    n = valid.sum().item()
                    epe_sum += e
                    valid_pix += n

                    # thresholds
                    thr1_cnt += ((epe_map < 1.0) & valid).sum().item()
                    thr3_cnt += ((epe_map < 3.0) & valid).sum().item()
                    thr5_cnt += ((epe_map < 5.0) & valid).sum().item()
            else:
                proxy_losses.append(float(losses["total"].detach().cpu().item()))

            # Viz chỉ 1 lần đầu
            if (self.writer is not None) and self.viz_enable and (not did_viz):
                img0 = clip[:, 0]
                img1 = clip[:, 1]
                grid_img0 = vutils.make_grid(img0[: self.viz_max], nrow=min(4, self.viz_max), normalize=True)
                grid_img1 = vutils.make_grid(img1[: self.viz_max], nrow=min(4, self.viz_max), normalize=True)
                self.writer.add_image("val/img0", grid_img0, self.global_step)
                self.writer.add_image("val/img1", grid_img1, self.global_step)
                if out["flows"]:
                    flow_rgb = _flow_to_rgb_t(out["flows"][0][: self.viz_max].clamp(-500, 500))
                    grid_flow = vutils.make_grid(flow_rgb, nrow=min(4, self.viz_max))
                    self.writer.add_image("val/flow_rgb_pair0", grid_flow, self.global_step)
                did_viz = True

            if fast:
                break  # debug nhanh

        metrics: Dict[str, float] = {}
        if valid_pix > 0:
            metrics["epe"] = epe_sum / valid_pix
            metrics["1px"] = thr1_cnt / valid_pix
            metrics["3px"] = thr3_cnt / valid_pix
            metrics["5px"] = thr5_cnt / valid_pix
        if proxy_losses:
            metrics["total"] = float(np.mean(proxy_losses))

        if self.writer:
            for k, v in metrics.items():
                self.writer.add_scalar(f"val/{k}", v, self.global_step)

        self.model.train()
        return metrics


    # ------------------ internals ------------------ #
    def _init_scheduler(self, train_loader: DataLoader, epochs: int):
        sc = self.sched_cfg
        sc_type = sc.get("type", "cosine").lower()
        self._sched_per_batch = bool(sc.get("per_batch", False))

        if sc_type == "cosine":
            if self._sched_per_batch:
                T = epochs * max(1, len(train_loader))
                self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.opt, T_max=T, eta_min=float(sc.get("min_lr", 1e-6))
                )
            else:
                self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.opt, T_max=epochs, eta_min=float(sc.get("min_lr", 1e-6))
                )
        elif sc_type == "onecycle":
            self.sched = torch.optim.lr_scheduler.OneCycleLR(
                self.opt,
                max_lr=float(self.cfg["optim"]["lr"]),
                steps_per_epoch=max(1, len(train_loader)),
                epochs=epochs,
                pct_start=float(sc.get("pct_start", 0.05)),
                anneal_strategy=sc.get("anneal_strategy", "cos"),
                div_factor=float(sc.get("div_factor", 25.0)),
                final_div_factor=float(sc.get("final_div_factor", 1e4)),
            )
            self._sched_per_batch = True
        else:
            self.sched = None
            self._sched_per_batch = False

    def _train_one_epoch(self, train_loader: DataLoader, step_scheduler_per_batch: bool) -> Dict[str, float]:
        self.model.train()
        accum = {"loss": 0.0, "steps": 0}
        self.opt.zero_grad(set_to_none=True)

        for ib, batch in enumerate(tqdm(train_loader, desc="Train")):
            batch = _to_device(batch, self.device)
            clip = batch["clip"]  # (B,T,3,H,W)
            sam_masks = batch.get("sam_masks", None)
            # Optional: supervised GT flows (per pair or single) for semi-supervision
            flow_gt = batch.get("flow", None)
            flow_gt_list = batch.get("flow_list", None)
            valid_mask = batch.get("valid", None)

            # Type normalization
            if clip.dtype == torch.uint8:
                clip = clip.float() / 255.0

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = self.model(clip, sam_masks=sam_masks, use_sam=self.use_sam_default)
                ldict = self._compute_unsup_losses(clip, out)
                loss = ldict["total"]

                # Optional semi-supervised EPE term on the last predicted pair
                if self.w_epe_sup > 0.0 and (flow_gt is not None or flow_gt_list is not None):
                    pred_list = out["flows"]
                    if flow_gt is not None:
                        gt_use = [flow_gt]
                    else:
                        gt_use = flow_gt_list
                    k = min(len(pred_list), len(gt_use)) - 1
                    if k >= 0:
                        pred = pred_list[k]
                        gt = gt_use[k]
                        if pred.shape[-2:] != gt.shape[-2:]:
                            scale = gt.shape[-1] / float(pred.shape[-1])
                            pred = F.interpolate(pred, size=gt.shape[-2:], mode="bilinear", align_corners=True) * scale
                        epe_b = _masked_epe(pred, gt, valid_mask)
                        loss = loss + self.w_epe_sup * epe_b.mean()

            # Accumulate & step
            loss_to_backprop = loss / self.accum_steps
            self.scaler.scale(loss_to_backprop).backward()

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

            # TB step logging
            if self.writer and (self.global_step % int(self.cfg.get("logging", {}).get("log_every", 200)) == 0):
                self.writer.add_scalar("train/loss_step", float(loss.detach().item()), self.global_step)
                self.writer.add_scalar("train/lr_step", self.opt.param_groups[0]["lr"], self.global_step)

                # Quick viz: first two frames + first predicted flow
                if self.viz_enable:
                    img0 = clip[:, 0]
                    img1 = clip[:, 1]
                    grid_img0 = vutils.make_grid(img0[: self.viz_max], nrow=min(4, self.viz_max), normalize=True)
                    grid_img1 = vutils.make_grid(img1[: self.viz_max], nrow=min(4, self.viz_max), normalize=True)
                    self.writer.add_image("train/img0", grid_img0, self.global_step)
                    self.writer.add_image("train/img1", grid_img1, self.global_step)
                    if out["flows"]:
                        flow_rgb = _flow_to_rgb_t(out["flows"][0][: self.viz_max].clamp(-500, 500))
                        grid_flow = vutils.make_grid(flow_rgb, nrow=min(4, self.viz_max))
                        self.writer.add_image("train/flow_rgb_pair0", grid_flow, self.global_step)

        return accum

    def _save(self, name: str, payload: Dict):
        path = self.workspace / name
        torch.save(payload, path)
