#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPFLOW_ROOT = PROJECT_ROOT / "models" / "UPFlow_pytorch"
sys.path.insert(0, str(PROJECT_ROOT))

from dataio import Animerun

# Avoid package-name collision between repository `utils` and UPFlow's `utils`.
sys.modules.pop("utils", None)
sys.modules.pop("utils.tools", None)
sys.path.insert(0, str(UPFLOW_ROOT))
from model.upflow import UPFlow_net


UPFLOW_MEAN = torch.tensor([104.920005, 110.1753, 114.785955], dtype=torch.float32).view(1, 3, 1, 1)
UPFLOW_SCALE = 0.0039216


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: Optional[str]) -> Dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def cfg_get(cfg: Dict, key: str, default):
    cur = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def normalize_upflow(images: torch.Tensor) -> torch.Tensor:
    mean = UPFLOW_MEAN.to(device=images.device, dtype=images.dtype)
    return (images - mean) * UPFLOW_SCALE


def build_upflow_model(force_pytorch_corr: bool = True) -> UPFlow_net:
    net_conf = UPFlow_net.config()
    net_conf.update({"if_use_cor_pytorch": force_pytorch_corr})
    return net_conf()


def collate_metrics(chunks: List[np.ndarray]) -> float:
    if not chunks:
        return float("nan")
    vals = [x for x in chunks if x is not None and x.size > 0]
    if not vals:
        return float("nan")
    return float(np.mean(np.concatenate(vals, axis=0)))


def compute_eval_metrics(model: torch.nn.Module, loader: DataLoader, device: torch.device, max_batches: int = -1) -> Dict[str, float]:
    model.eval()

    epe_all_list: List[np.ndarray] = []
    epe_occ_list: List[np.ndarray] = []
    epe_nonocc_list: List[np.ndarray] = []
    epe_flat_list: List[np.ndarray] = []
    epe_line_list: List[np.ndarray] = []
    epe_s10_list: List[np.ndarray] = []
    epe_s1050_list: List[np.ndarray] = []
    epe_s50_list: List[np.ndarray] = []

    thr1_cnt = 0
    thr3_cnt = 0
    thr5_cnt = 0
    valid_cnt_total = 0

    with torch.no_grad():
        for step, batch in enumerate(tqdm(loader, desc="Validate")):
            if max_batches > 0 and step >= max_batches:
                break

            img1 = batch["image1"].to(device, non_blocking=True)
            img2 = batch["image2"].to(device, non_blocking=True)
            gt = batch["flow"].to(device, non_blocking=True)

            img1 = normalize_upflow(img1)
            img2 = normalize_upflow(img2)
            bsz = img1.shape[0]
            start = torch.zeros((bsz, 2, 1, 1), device=device, dtype=img1.dtype)

            out = model(
                {
                    "im1": img1,
                    "im2": img2,
                    "im1_raw": img1,
                    "im2_raw": img2,
                    "im1_sp": img1,
                    "im2_sp": img2,
                    "start": start,
                    "if_loss": False,
                }
            )
            pred = out["flow_f_out"]

            if pred.shape[-2:] != gt.shape[-2:]:
                hp, wp = pred.shape[-2:]
                hg, wg = gt.shape[-2:]
                pred = torch.nn.functional.interpolate(pred, size=(hg, wg), mode="bilinear", align_corners=True)
                pred[:, 0] *= wg / float(wp)
                pred[:, 1] *= hg / float(hp)

            epe_map = torch.norm(pred - gt, dim=1)
            mag = torch.norm(gt, dim=1)

            valid = batch.get("valid")
            if valid is None:
                valid = torch.ones_like(epe_map, dtype=torch.bool)
            else:
                valid = valid.to(device).bool()
                if valid.dim() == 4 and valid.size(1) == 1:
                    valid = valid[:, 0]
                if valid.dim() == 2:
                    valid = valid.unsqueeze(0).expand_as(epe_map)

            epe_all_list.append(epe_map[valid].detach().cpu().numpy())
            valid_cnt = int(valid.sum().item())
            valid_cnt_total += valid_cnt
            if valid_cnt > 0:
                v = epe_map[valid]
                thr1_cnt += int((v < 1.0).sum().item())
                thr3_cnt += int((v < 3.0).sum().item())
                thr5_cnt += int((v < 5.0).sum().item())

            occ = batch.get("occ")
            if occ is not None:
                occ = occ.to(device)
                if occ.dim() == 4 and occ.size(1) == 1:
                    occ = occ[:, 0]
                if occ.dim() == 2:
                    occ = occ.unsqueeze(0).expand_as(epe_map)
                occ = occ.bool()

                occ_mask = (occ == 0) & valid
                nonocc_mask = (occ == 1) & valid
                if occ_mask.any():
                    epe_occ_list.append(epe_map[occ_mask].detach().cpu().numpy())
                if nonocc_mask.any():
                    epe_nonocc_list.append(epe_map[nonocc_mask].detach().cpu().numpy())

            line = batch.get("line")
            if line is not None:
                line = line.to(device)
                if line.dim() == 4 and line.size(1) == 1:
                    line = line[:, 0]
                if line.dim() == 2:
                    line = line.unsqueeze(0).expand_as(epe_map)

                flat_mask = (line > 0) & valid
                line_mask = (line == 0) & valid
                if flat_mask.any():
                    epe_flat_list.append(epe_map[flat_mask].detach().cpu().numpy())
                if line_mask.any():
                    epe_line_list.append(epe_map[line_mask].detach().cpu().numpy())

            s10 = (mag <= 10.0) & valid
            s1050 = (mag > 10.0) & (mag <= 50.0) & valid
            s50 = (mag > 50.0) & valid
            if s10.any():
                epe_s10_list.append(epe_map[s10].detach().cpu().numpy())
            if s1050.any():
                epe_s1050_list.append(epe_map[s1050].detach().cpu().numpy())
            if s50.any():
                epe_s50_list.append(epe_map[s50].detach().cpu().numpy())

    metrics: Dict[str, float] = {}
    metrics["epe"] = collate_metrics(epe_all_list)
    metrics["1px"] = float(thr1_cnt / max(1, valid_cnt_total))
    metrics["3px"] = float(thr3_cnt / max(1, valid_cnt_total))
    metrics["5px"] = float(thr5_cnt / max(1, valid_cnt_total))
    metrics["epe_occ"] = collate_metrics(epe_occ_list)
    metrics["epe_nonocc"] = collate_metrics(epe_nonocc_list)
    metrics["epe_flat"] = collate_metrics(epe_flat_list)
    metrics["epe_line"] = collate_metrics(epe_line_list)
    metrics["epe_s<10"] = collate_metrics(epe_s10_list)
    metrics["epe_s10-50"] = collate_metrics(epe_s1050_list)
    metrics["epe_s>50"] = collate_metrics(epe_s50_list)

    model.train()
    return metrics


def save_metrics(metrics: Dict[str, float], out_json: Path, out_csv: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, v])


def build_loaders(
    data_root: str,
    val_root: str,
    crop_size: Tuple[int, int],
    stride_min: int,
    stride_max: int,
    resize: bool,
    keep_aspect: bool,
    batch_size: int,
    val_batch_size: int,
    num_workers: int,
    drop_last: bool,
    do_flip: bool,
    grayscale_p: float,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = Animerun(
        root=data_root,
        stride_min=stride_min,
        stride_max=stride_max,
        crop_size=crop_size,
        color_jitter=None,
        do_flip=do_flip,
        grayscale_p=grayscale_p,
        is_test=False,
        resize=resize,
        keep_aspect=keep_aspect,
    )

    val_ds = Animerun(
        root=val_root,
        stride_min=1,
        stride_max=1,
        crop_size=crop_size,
        color_jitter=None,
        do_flip=False,
        grayscale_p=0.0,
        is_test=True,
        resize=resize,
        keep_aspect=keep_aspect,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UPFlow on AnimeRun and report AniUnFlow metrics")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--workspace", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--val_root", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--val_batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--crop_h", type=int, default=None)
    parser.add_argument("--crop_w", type=int, default=None)
    parser.add_argument("--resize", action="store_true")
    parser.add_argument("--no_resize", action="store_true")
    parser.add_argument("--keep_aspect", action="store_true")
    parser.add_argument("--drop_last", action="store_true")

    parser.add_argument("--max_steps_per_epoch", type=int, default=-1)
    parser.add_argument("--eval_max_batches", type=int, default=-1)
    parser.add_argument("--final_eval_max_batches", type=int, default=-1)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    data_root = str(args.data_root or cfg_get(cfg, "data.train_root", "data/AnimeRun_v2"))
    val_root = str(args.val_root or cfg_get(cfg, "data.val_root", data_root))

    crop_default = cfg_get(cfg, "data.crop_size", [256, 512])
    crop_h = int(args.crop_h if args.crop_h is not None else crop_default[0])
    crop_w = int(args.crop_w if args.crop_w is not None else crop_default[1])
    crop_size = (crop_h, crop_w)

    resize_default = bool(cfg_get(cfg, "data.resize", True))
    resize = False if args.no_resize else (True if args.resize else resize_default)
    keep_aspect = bool(args.keep_aspect or cfg_get(cfg, "data.keep_aspect", False))

    batch_size = int(args.batch_size if args.batch_size is not None else cfg_get(cfg, "data.batch_size", 8))
    val_batch_size = int(args.val_batch_size if args.val_batch_size is not None else max(1, batch_size // 2))
    num_workers = int(args.num_workers if args.num_workers is not None else cfg_get(cfg, "data.num_workers", 4))
    drop_last = bool(args.drop_last or cfg_get(cfg, "data.drop_last", True))

    stride_min = int(cfg_get(cfg, "data.stride_min", 1))
    stride_max = int(cfg_get(cfg, "data.stride_max", 3))
    do_flip = bool(cfg_get(cfg, "data.do_flip", True))
    grayscale_p = float(cfg_get(cfg, "data.grayscale_p", 0.0))

    epochs = int(args.epochs if args.epochs is not None else cfg_get(cfg, "optim.epochs", 60))
    lr = float(args.lr if args.lr is not None else cfg_get(cfg, "optim.lr", 1e-4))
    weight_decay = float(args.weight_decay if args.weight_decay is not None else cfg_get(cfg, "optim.weight_decay", 1e-4))

    workspace = Path(args.workspace or cfg_get(cfg, "workspace", "workspaces/upflow_animerun"))
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "config_effective.json").write_text(
        json.dumps(
            {
                "data_root": data_root,
                "val_root": val_root,
                "crop_size": crop_size,
                "resize": resize,
                "keep_aspect": keep_aspect,
                "batch_size": batch_size,
                "val_batch_size": val_batch_size,
                "num_workers": num_workers,
                "drop_last": drop_last,
                "stride_min": stride_min,
                "stride_max": stride_max,
                "epochs": epochs,
                "lr": lr,
                "weight_decay": weight_decay,
                "seed": args.seed,
                "force_pytorch_corr": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[UPFlow Train] device={device}, workspace={workspace}")
    print(f"[UPFlow Train] data_root={data_root}, val_root={val_root}")
    print(f"[UPFlow Train] crop_size={crop_size}, resize={resize}, keep_aspect={keep_aspect}")

    train_loader, val_loader = build_loaders(
        data_root=data_root,
        val_root=val_root,
        crop_size=crop_size,
        stride_min=stride_min,
        stride_max=stride_max,
        resize=resize,
        keep_aspect=keep_aspect,
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        num_workers=num_workers,
        drop_last=drop_last,
        do_flip=do_flip,
        grayscale_p=grayscale_p,
    )

    model = build_upflow_model(force_pytorch_corr=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)

    start_epoch = 1
    best_epe = float("inf")

    if args.resume:
        resume_path = Path(args.resume)
        ckpt = torch.load(resume_path, map_location=device)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_epe = float(ckpt.get("best_epe", best_epe))
        print(f"[UPFlow Train] resumed from {resume_path}, start_epoch={start_epoch}")

    history: List[Dict] = []

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_steps = 0

        pbar = tqdm(train_loader, desc=f"Train epoch {epoch}/{epochs}")
        for step, batch in enumerate(pbar, start=1):
            if args.max_steps_per_epoch > 0 and step > args.max_steps_per_epoch:
                break

            img1 = batch["image1"].to(device, non_blocking=True)
            img2 = batch["image2"].to(device, non_blocking=True)
            img1 = normalize_upflow(img1)
            img2 = normalize_upflow(img2)

            bsz = img1.shape[0]
            start = torch.zeros((bsz, 2, 1, 1), device=device, dtype=img1.dtype)

            output = model(
                {
                    "im1": img1,
                    "im2": img2,
                    "im1_raw": img1,
                    "im2_raw": img2,
                    "im1_sp": img1,
                    "im2_sp": img2,
                    "start": start,
                    "if_loss": True,
                }
            )

            loss_terms = []
            for key in ("photo_loss", "smooth_loss", "census_loss", "msd_loss", "eq_loss", "oi_loss"):
                val = output.get(key)
                if torch.is_tensor(val):
                    loss_terms.append(val.mean())

            if not loss_terms:
                raise RuntimeError("UPFlow forward did not produce any loss term.")

            loss = torch.stack(loss_terms).sum()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_v = float(loss.detach().item())
            epoch_loss_sum += loss_v
            epoch_steps += 1
            pbar.set_postfix(loss=f"{loss_v:.4f}")

        train_loss = epoch_loss_sum / max(1, epoch_steps)

        val_metrics = None
        if epoch % max(1, args.val_every) == 0:
            val_metrics = compute_eval_metrics(model, val_loader, device, max_batches=args.eval_max_batches)
            epe_str = val_metrics.get("epe", float("nan"))
            print(f"[Epoch {epoch}] train_loss={train_loss:.6f} val_epe={epe_str:.6f}")
        else:
            print(f"[Epoch {epoch}] train_loss={train_loss:.6f}")

        record = {"epoch": epoch, "train_loss": train_loss, "val_metrics": val_metrics}
        history.append(record)

        if val_metrics is not None:
            val_epe = val_metrics.get("epe", float("inf"))
            if np.isfinite(val_epe) and val_epe < best_epe:
                best_epe = float(val_epe)
                best_path = workspace / "best_upflow_animerun.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_epe": best_epe,
                        "val_metrics": val_metrics,
                    },
                    best_path,
                )
                print(f"[UPFlow Train] saved best checkpoint: {best_path}")

        if epoch % max(1, args.save_every) == 0 or epoch == epochs:
            ckpt_path = workspace / f"ckpt_upflow_e{epoch:03d}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_epe": best_epe,
                    "val_metrics": val_metrics,
                },
                ckpt_path,
            )
            print(f"[UPFlow Train] saved checkpoint: {ckpt_path}")

        with open(workspace / "train_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    final_metrics = compute_eval_metrics(model, val_loader, device, max_batches=args.final_eval_max_batches)
    save_metrics(final_metrics, workspace / "metrics_final.json", workspace / "metrics_final.csv")
    print("[UPFlow Train] final eval metrics:")
    for k, v in final_metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
