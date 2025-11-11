# file: main.py
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Your dataset & trainer modules (adjust import paths to your project)
from dataio import UnlabeledClipDataset
from engine import UnsupervisedClipTrainer

def build_cfg(args: argparse.Namespace) -> dict:
    return {
        "model": {
            "name": "aniflowformer-t",
            "args": {
                "use_sam": False,
                # TODO: add any required AFConfig args here to match the checkpoint
            },
        },
        "optim": {
            "seed": 1337,
            "epochs": 1,
            "lr": 2e-4,
            "weight_decay": 1e-4,
            "clip": 1.0,
            "accum_steps": 1,
            "scheduler": {"type": "cosine", "per_batch": True, "min_lr": 1e-6},
        },
        "loss": {"w_epe_sup": 0.0},
        "logging": {"use_tb": False, "log_every": 50, "tb_dir": "tb"},
        "viz": {"enable": False, "max_samples": 4, "save_dir": "val_vis"},
        "ckpt": {"save_every": 1},
    }

def build_val_loader(args: argparse.Namespace) -> DataLoader:
    val_ds = UnlabeledClipDataset(
        root=args.data_root,
        T=args.T,
        is_test=True,           # read GT from test/Flow/...
        resize=not args.no_resize,
        keep_aspect=args.keep_aspect,
        pad_mode="reflect",
    )
    return DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

def _strip_module_prefix(sd: dict) -> dict:
    # remove a leading "module." (from DataParallel) if present
    if not sd:
        return sd
    any_module = any(k.startswith("module.") for k in sd.keys())
    return { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() } if any_module else sd

def load_checkpoint_into_trainer(trainer: UnsupervisedClipTrainer, ckpt_path: str, strict: bool = False, resume_optim: bool = False):
    """Load model (and optionally optimizer/scheduler) from a checkpoint."""
    device = trainer.device
    ckpt = torch.load(ckpt_path, map_location=device)

    # Prefer EMA weights if available
    sd = None
    for key in ["ema_state_dict", "ema", "model_ema", "state_dict_ema"]:
        if key in ckpt and isinstance(ckpt[key], dict):
            sd = ckpt[key]; break
    if sd is None:
        # Fallbacks
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            sd = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            sd = ckpt["model"]
        else:
            # assume the whole file is a raw state_dict
            sd = ckpt if isinstance(ckpt, dict) else None
    if sd is None:
        raise RuntimeError(f"Cannot find model state_dict in '{ckpt_path}'")

    sd = _strip_module_prefix(sd)

    # Load into trainer.model
    missing, unexpected = trainer.model.load_state_dict(sd, strict=strict)
    print(f"[CKPT] Loaded weights from: {ckpt_path}")
    if missing:
        print(f"[CKPT][warn] Missing keys ({len(missing)}): {list(missing)[:8]}{' ...' if len(missing)>8 else ''}")
    if unexpected:
        print(f"[CKPT][warn] Unexpected keys ({len(unexpected)}): {list(unexpected)[:8]}{' ...' if len(unexpected)>8 else ''}")

    # Optionally resume optimizer/scheduler
    if resume_optim:
        if "opt" in ckpt and ckpt["opt"] is not None:
            try:
                trainer.opt.load_state_dict(ckpt["opt"])
                print("[CKPT] Optimizer state restored.")
            except Exception as e:
                print(f"[CKPT][warn] Failed to load optimizer state: {e}")
        if "sched" in ckpt and ckpt["sched"] is not None and trainer.sched is not None:
            try:
                trainer.sched.load_state_dict(ckpt["sched"])
                print("[CKPT] Scheduler state restored.")
            except Exception as e:
                print(f"[CKPT][warn] Failed to load scheduler state: {e}")
        if "best" in ckpt:
            trainer.best_metric = float(ckpt["best"])
        if "step" in ckpt:
            trainer.global_step = int(ckpt["step"])

def patch_validate_anisotropic_scale(trainer: UnsupervisedClipTrainer):
    """Optional: fix anisotropic flow scaling inside validate()."""
    import types, torch.nn.functional as F
    def _validate(self, val_loader, epoch: int = 0):
        self.model.eval()
        device = self.device
        epe_all = []; thr1 = thr3 = thr5 = 0; valid_cnt_total = 0
        proxy_losses = []
        from tqdm import tqdm
        for batch in tqdm(val_loader, desc="Validate"):
            batch = _to_device(batch, device)
            clip = batch["clip"]
            sam_masks = batch.get("sam_masks", None)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = self.model(clip, sam_masks=sam_masks, use_sam=self.use_sam_default)
                losses = self._compute_unsup_losses(clip, out)
            gt_any = batch.get("flow", None) or batch.get("flow_list", None)
            if gt_any is not None:
                gt_list = [gt_any] if torch.is_tensor(gt_any) else gt_any
                pred_list = out["flows"]
                L = min(len(pred_list), len(gt_list))
                for k in range(L):
                    pred, gt = pred_list[k], gt_list[k]
                    if pred.shape[-2:] != gt.shape[-2:]:
                        H1, W1 = pred.shape[-2:]; H2, W2 = gt.shape[-2:]
                        pred = F.interpolate(pred, size=(H2, W2), mode="bilinear", align_corners=True)
                        pred[:, 0] *= float(W2) / float(W1)  # scale u
                        pred[:, 1] *= float(H2) / float(H1)  # scale v
                    epe_map = torch.norm(pred - gt, dim=1)
                    epe_all.append(epe_map.detach().cpu().numpy())
                    v = torch.ones_like(epe_map, dtype=torch.bool)
                    valid_cnt_total += int(v.sum().item())
                    thr1 += int((epe_map < 1.0).sum().item())
                    thr3 += int((epe_map < 3.0).sum().item())
                    thr5 += int((epe_map < 5.0).sum().item())
            else:
                proxy_losses.append(float(losses["total"].detach().cpu().item()))
            # NOTE: keep/remove your original 'break' here as you prefer
            break
        metrics = {}
        if epe_all:
            import numpy as np
            metrics["epe"] = float(np.mean(np.concatenate(epe_all)))
            metrics["1px"] = float(thr1 / max(1, valid_cnt_total))
            metrics["3px"] = float(thr3 / max(1, valid_cnt_total))
            metrics["5px"] = float(thr5 / max(1, valid_cnt_total))
        if proxy_losses:
            metrics["total"] = float(np.mean(proxy_losses))
        self.model.train()
        return metrics
    trainer.validate = types.MethodType(_validate, trainer)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--workspace", type=str, default="outputs/aft_eval")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--T", type=int, default=5)
    p.add_argument("--no_resize", action="store_true")
    p.add_argument("--keep_aspect", action="store_true")

    # Checkpoint options
    p.add_argument("--ckpt", type=str, required=True, help="Path to .pth checkpoint to evaluate")
    p.add_argument("--strict", action="store_true", help="Strict key matching when loading state dict")
    p.add_argument("--resume_optim", action="store_true", help="Also load optimizer/scheduler (if present)")
    p.add_argument("--anisotropic_scale_fix", action="store_true", help="Patch validate() to scale u/v separately")

    args = p.parse_args()

    cfg = build_cfg(args)
    Path(args.workspace).mkdir(parents=True, exist_ok=True)

    # Build val loader (reads GT)
    val_loader = build_val_loader(args)

    # Build trainer/model
    trainer = UnsupervisedClipTrainer(args, cfg, Path(args.workspace))

    # (Optional) patch validate with anisotropic flow scaling fix
    if args.anisotropic_scale_fix:
        patch_validate_anisotropic_scale(trainer)

    # --- Load checkpoint weights here ---
    load_checkpoint_into_trainer(trainer, args.ckpt, strict=args.strict, resume_optim=args.resume_optim)

    # Validate
    metrics = trainer.validate(val_loader)
    print("[VAL] metrics:", metrics)

if __name__ == "__main__":
    main()
