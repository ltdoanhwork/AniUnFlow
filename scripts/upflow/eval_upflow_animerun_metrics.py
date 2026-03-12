#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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


def evaluate_metrics(model: torch.nn.Module, loader: DataLoader, device: torch.device, max_batches: int = -1) -> Dict[str, float]:
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
        for step, batch in enumerate(tqdm(loader, desc="Evaluate")):
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate UPFlow on AnimeRun and export AniUnFlow metrics")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--crop_h", type=int, default=None)
    parser.add_argument("--crop_w", type=int, default=None)
    parser.add_argument("--resize", action="store_true")
    parser.add_argument("--no_resize", action="store_true")
    parser.add_argument("--keep_aspect", action="store_true")
    parser.add_argument("--out_json", type=str, default=None)
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--max_batches", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    data_root = str(args.data_root or cfg_get(cfg, "data.val_root", cfg_get(cfg, "data.train_root", "data/AnimeRun_v2")))
    crop_default = cfg_get(cfg, "data.crop_size", [256, 512])
    crop_h = int(args.crop_h if args.crop_h is not None else crop_default[0])
    crop_w = int(args.crop_w if args.crop_w is not None else crop_default[1])
    crop_size = (crop_h, crop_w)

    resize_default = bool(cfg_get(cfg, "data.resize", True))
    resize = False if args.no_resize else (True if args.resize else resize_default)
    keep_aspect = bool(args.keep_aspect or cfg_get(cfg, "data.keep_aspect", False))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = Animerun(
        root=data_root,
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
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = build_upflow_model(force_pytorch_corr=True).to(device)

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state = ckpt["model"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
    model.load_state_dict(state, strict=False)

    metrics = evaluate_metrics(model, loader, device, max_batches=args.max_batches)

    out_json = Path(args.out_json) if args.out_json else ckpt_path.parent / f"metrics_{ckpt_path.stem}.json"
    out_csv = Path(args.out_csv) if args.out_csv else ckpt_path.parent / f"metrics_{ckpt_path.stem}.csv"
    save_metrics(metrics, out_json, out_csv)

    print(f"[UPFlow Eval] ckpt={ckpt_path}")
    print(f"[UPFlow Eval] data_root={data_root}, crop_size={crop_size}, resize={resize}, keep_aspect={keep_aspect}")
    print(f"[UPFlow Eval] saved json: {out_json}")
    print(f"[UPFlow Eval] saved csv : {out_csv}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
