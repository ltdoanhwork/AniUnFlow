#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, os, glob, csv, math
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn.functional as F

# Model (your clip-based model)
from models import AniFlowFormerT, ModelConfig

# Color-wheel viz (Middlebury)
from utils.flow_viz import flow_to_image as flowviz_rgb, compute_flow_magnitude_radmax  # returns RGB uint8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- IO utils ----------
def list_images(folder: str) -> List[str]:
    files = []
    for e in ("*.png", "*.jpg", "*.jpeg"):
        files += glob.glob(os.path.join(folder, e))
    files.sort()
    return files

def load_image_rgb(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.uint8)
    ten = torch.from_numpy(arr).permute(2, 0, 1).float()
    return ten  # [3,H,W] 0..255

def pad_to_multiple(img: torch.Tensor, mult: int = 16) -> Tuple[torch.Tensor, Tuple[int,int,int,int]]:
    _, H, W = img.shape
    pad_h = (mult - (H % mult)) % mult
    pad_w = (mult - (W % mult)) % mult
    pad = (0, pad_w, 0, pad_h)  # left,right,top,bottom
    out = F.pad(img, pad, mode="replicate")
    return out, (0, pad_w, 0, pad_h)

def unpad(x: torch.Tensor, pad: Tuple[int,int,int,int]) -> torch.Tensor:
    l, r, t, b = pad
    if x.dim() == 4:
        return x[..., :x.shape[-2]-b, :x.shape[-1]-r]
    return x[..., :x.shape[-2]-b, :x.shape[-1]-r]


# ---------- GT flow loaders ----------
def _read_flo(path: str) -> np.ndarray:
    # Middlebury .flo
    with open(path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)[0]
        if magic != 202021.25:
            raise RuntimeError("Invalid .flo file %s" % path)
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2*w*h)
    flow = np.reshape(data, (h, w, 2))
    return flow

def _read_exr(path: str) -> np.ndarray:
    # Try OpenCV EXR (requires OpenCV built with OpenEXR/Imath)
    # Expect 2-channel or 3-channel; if 3, use first two.
    exr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if exr is None:
        raise RuntimeError(f"Failed to read EXR: {path}")
    if exr.ndim == 2:
        # Single-channel? Not expected; duplicate to 2
        exr = np.stack([exr, np.zeros_like(exr)], axis=-1)
    elif exr.shape[2] >= 2:
        exr = exr[..., :2]
    exr = exr.astype(np.float32)
    return exr

def _read_npy(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 3 and arr.shape[2] >= 2:
        return arr[..., :2].astype(np.float32)
    if arr.ndim == 2:
        return np.stack([arr, np.zeros_like(arr)], axis=-1).astype(np.float32)
    raise RuntimeError(f"Invalid npy flow shape: {arr.shape}")

def load_gt_flow(base_dir: str, stem: str, pattern: str,
                 explicit_ext: Optional[str]) -> Optional[np.ndarray]:
    """
    base_dir: folder of GT flow (forward)
    stem: frame stem (e.g., 'Image0001')
    pattern: Python format or template; tokens: {stem}, {index}, {index1}
        - {stem}: exact frame stem without extension
        - {index}: numeric found in stem if any (or None)
        - {index1}: index+1 (for patterns like Image0001_1.exr)
    explicit_ext: force extension ('.flo' / '.exr' / '.npy') or None -> try in order
    """
    # Try to extract integer index from stem tail
    idx = None
    import re
    m = re.search(r'(\d+)$', stem)
    if m:
        idx = int(m.group(1))

    def render(ext: str):
        s = pattern
        s = s.replace("{stem}", stem)
        if idx is not None:
            s = s.replace("{index}", f"{idx}")
            s = s.replace("{index1}", f"{idx+1}")
        return os.path.join(base_dir, s if s.endswith(ext) else s + ext)

    tried = []

    def try_one(path: str, ext: str):
        if not os.path.isfile(path):
            return None
        if ext == ".flo":
            return _read_flo(path)
        if ext == ".exr":
            return _read_exr(path)
        if ext == ".npy":
            return _read_npy(path)
        return None

    if explicit_ext:
        p = render(explicit_ext)
        tried.append(p)
        gt = try_one(p, explicit_ext)
        if gt is not None:
            return gt

    for ext in (".flo", ".exr", ".npy"):
        p = render(ext)
        tried.append(p)
        gt = try_one(p, ext)
        if gt is not None:
            return gt

    print("[Warn] GT not found. Tried:", tried)
    return None


# ---------- Debug helpers ----------
def debug_flow_stats(flow, name: str):
    """Print basic stats for a flow tensor/array, for debugging.
    flow: Tensor with shape [..., 2, H, W] or [H, W, 2], or numpy array
    """
    if isinstance(flow, torch.Tensor):
        f = flow.detach().cpu()
    else:
        # numpy or already on CPU
        f = torch.from_numpy(flow) if isinstance(flow, np.ndarray) else flow

    print(f"[DEBUG] {name}:")
    print(f"  shape = {tuple(f.shape)}")

    # Try to interpret as (..., 2, H, W) or (H, W, 2)
    if f.ndim == 4 and f.shape[-3] == 2:
        # [..., 2, H, W]
        u = f[..., 0, :, :]
        v = f[..., 1, :, :]
    elif f.ndim == 3 and f.shape[-1] == 2:
        # [H, W, 2]
        u = f[..., 0]
        v = f[..., 1]
    else:
        print(f"  (not a standard flow shape, skip detailed stats)")
        return

    mag = torch.sqrt(u**2 + v**2)
    print(f"  u: min={u.min():.4f}, max={u.max():.4f}, mean={u.mean():.4f}, std={u.std():.4f}")
    print(f"  v: min={v.min():.4f}, max={v.max():.4f}, mean={v.mean():.4f}, std={v.std():.4f}")
    print(f"  |flow|: min={mag.min():.4f}, max={mag.max():.4f}, mean={mag.mean():.4f}, std={mag.std():.4f}")


# ---------- Viz helpers ----------
def epe_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    # pred, gt: HxWx2 float32
    epe = np.linalg.norm(pred - gt, axis=2)
    return epe

def epe_color(epe: np.ndarray) -> np.ndarray:
    # Normalize by p95 for stable colormap
    p95 = np.percentile(epe, 95.0) + 1e-6
    x = np.clip(epe / p95, 0, 1)
    x8 = (x * 255.0).astype(np.uint8)
    cm = cv2.applyColorMap(x8, cv2.COLORMAP_JET)  # BGR
    return cm

def stack_v(*imgs_rgb: np.ndarray) -> np.ndarray:
    # all RGB uint8 of same width
    ws = [im.shape[1] for im in imgs_rgb]
    w = min(ws)
    rs = [cv2.resize(im, (w, int(im.shape[0]*w/im.shape[1])), interpolation=cv2.INTER_AREA)
          if im.shape[1] != w else im for im in imgs_rgb]
    return np.concatenate(rs, axis=0)


# ---------- Model ----------
def build_model(ckpt_path: str, use_sam: bool = False) -> AniFlowFormerT:
    cfg = ModelConfig(use_sam=use_sam)
    model = AniFlowFormerT(cfg).to(DEVICE)
    model.eval()

    if ckpt_path and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        state = ckpt.get("state_dict", ckpt.get("model", ckpt))
        new_state = {}
        for k, v in state.items():
            nk = k[7:] if k.startswith("module.") else k
            new_state[nk] = v
        missing, unexpected = model.load_state_dict(new_state, strict=False)
        if missing:   print("[Warn] Missing keys:", len(missing))
        if unexpected: print("[Warn] Unexpected keys:", len(unexpected))
    else:
        print("[Info] No checkpoint. Using random init.")
    return model


# ---------- Core compare ----------
@torch.no_grad()
def run_compare(model: AniFlowFormerT, frames_dir: str, gt_dir: str, out_dir: str,
                T: int, resize_h: int, resize_w: int,
                gt_pattern: str, gt_ext: Optional[str]):

    os.makedirs(out_dir, exist_ok=True)
    files = list_images(frames_dir)
    if len(files) < 2:
        raise RuntimeError("Need >=2 frames.")

    # cache padded frames
    imgs = []
    pads = None
    for p in files:
        im = load_image_rgb(p)
        if resize_h and resize_w:
            im = F.interpolate(im.unsqueeze(0), size=(resize_h, resize_w),
                               mode="bilinear", align_corners=True)[0]
        im, pad = pad_to_multiple(im, 16)
        if pads is None:
            pads = pad
        imgs.append(im)

    # metrics file
    csv_path = os.path.join(out_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as fcsv:
        wr = csv.writer(fcsv)
        wr.writerow(["pair_idx", "frame0", "frame1", "AEPE", "1px", "3px", "5px", "H", "W"])

        # sliding window
        for s in range(0, len(imgs)-1):
            sub = imgs[s:s+T]  # may be shorter at tail
            if len(sub) < 2:
                break

            # Debug info
            print(f"\n[PAIR {s}] -----------------------------------")
            print(f"  Using frames: {Path(files[s]).name} -> {Path(files[s+1]).name}")
            print(f"  Subclip length T' = {len(sub)}")

            clip = torch.stack(sub, dim=0).unsqueeze(0).to(DEVICE) / 255.0  # [1,T',3,H,W]
            print(f"  clip.shape = {tuple(clip.shape)}, clip.min={clip.min():.4f}, clip.max={clip.max():.4f}")

            out = model(clip)

            # Debug model output
            if isinstance(out, dict):
                print(f"  [DEBUG] model output keys: {list(out.keys())}")
                if "flows" in out:
                    flows = out["flows"]
                    if isinstance(flows, (list, tuple)):
                        print(f"  flows is list/tuple of length {len(flows)}")
                        for i, f in enumerate(flows):
                            print(f"    flows[{i}].shape = {tuple(f.shape)}")
                        pred = flows[0]
                    else:
                        print(f"  flows is Tensor with shape {tuple(flows.shape)}")
                        pred = flows[0] if flows.ndim == 4 else flows
                else:
                    print(f"  [WARN] 'flows' not in model output, available keys: {list(out.keys())}")
                    continue
            else:
                print(f"  [WARN] model output is not a dict. type: {type(out)}")
                continue

            # Debug raw model output flow
            debug_flow_stats(pred, "pred_raw (model output)")

            # upsample & rescale to image size
            H0, W0 = imgs[s].shape[-2:]
            h, w = pred.shape[-2:]
            f_up = F.interpolate(pred, size=(H0, W0), mode="bilinear", align_corners=True)
            sx = W0 / float(w); sy = H0 / float(h)
            f_up[:, 0] *= sx; f_up[:, 1] *= sy
            debug_flow_stats(f_up, "pred_upsampled_scaled")

            f_up = unpad(f_up, pads)  # [1,2,H,W]
            debug_flow_stats(f_up, "pred_unpadded")

            flow_pred = f_up[0].permute(1,2,0).detach().cpu().numpy().astype(np.float32)  # HxWx2
            debug_flow_stats(flow_pred, "flow_pred_np (final)")

            # build GT path from first frame stem
            stem0 = Path(files[s]).stem
            gt = load_gt_flow(gt_dir, stem0, pattern=gt_pattern, explicit_ext=gt_ext)
            if gt is None:
                print(f"[Warn] GT not found for {stem0}. Skip metrics/viz.")
                continue

            print(f"  Loaded GT: gt.shape = {gt.shape}, gt.dtype = {gt.dtype}")

            # match sizes and ensure correct shape
            if gt.shape[:2] != flow_pred.shape[:2]:
                print(f"  [WARN] GT size {gt.shape[:2]} != pred size {flow_pred.shape[:2]}, resizing GT...")
                gt = cv2.resize(gt, (flow_pred.shape[1], flow_pred.shape[0]), interpolation=cv2.INTER_NEAREST)

            debug_flow_stats(gt, "gt_flow_np")

            # metrics
            diff = flow_pred - gt
            epe = np.linalg.norm(diff, axis=2)
            aepe = float(np.mean(epe))
            c1 = float((epe < 1.0).mean())
            c3 = float((epe < 3.0).mean())
            c5 = float((epe < 5.0).mean())
            print(f"  Metrics: AEPE={aepe:.4f}, 1px={c1:.4f}, 3px={c3:.4f}, 5px={c5:.4f}")

            # === FIX: Use shared normalization for GT and Pred ===
            # Compute shared rad_max from both GT and Pred
            shared_rad_max = compute_flow_magnitude_radmax([gt, flow_pred], robust_percentile=95)
            print(f"  shared_rad_max = {shared_rad_max:.4f}")
            
            # === FIX: Proper image cropping (no black bar) ===
            img_crop = imgs[s].unsqueeze(0)   # [1,3,H_padded, W_padded]
            img_crop = unpad(img_crop, pads)  # [1,3,H,W] unpadded
            # Additional crop to match flow size
            H_flow, W_flow = flow_pred.shape[:2]
            img_crop = img_crop[..., :H_flow, :W_flow]  # safety crop
            img_np = img_crop[0].permute(1,2,0).cpu().numpy().astype(np.uint8)
            
            # === Use shared rad_max for both flows ===
            pred_rgb = flowviz_rgb(flow_pred, rad_max=shared_rad_max)                 # RGB
            gt_rgb   = flowviz_rgb(gt, rad_max=shared_rad_max)                        # RGB
            epe_bgr  = epe_color(epe)                         # BGR
            epe_rgb  = epe_bgr[..., ::-1]

            panel = stack_v(img_np, pred_rgb, gt_rgb, epe_rgb)  # RGB
            out_png = os.path.join(out_dir, f"{s:06d}.png")
            cv2.imwrite(out_png, panel[..., ::-1])  # save as BGR

            wr.writerow([s, Path(files[s]).name, Path(files[s+1]).name, aepe, c1, c3, c5, panel.shape[0], panel.shape[1]])
            print(f"[{s}] AEPE={aepe:.3f}  1px={c1:.3f}  3px={c3:.3f}  5px={c5:.3f} -> {out_png}")

    print(f"[OK] Saved per-pair viz PNGs and metrics CSV at: {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=False, help="Path to checkpoint .pth")
    ap.add_argument("--frames", type=str, required=True, help="Folder of RGB frames")
    ap.add_argument("--gt_dir", type=str, required=True, help="Folder of forward GT flows")
    ap.add_argument("--out_dir", type=str, required=True, help="Output folder for viz+metrics")
    ap.add_argument("--T", type=int, default=5, help="Clip length (>=2)")
    ap.add_argument("--resize_h", type=int, default=368)
    ap.add_argument("--resize_w", type=int, default=768)
    # Pattern tips:
    #  - Example 1: pattern="{stem}" + ext ".flo" -> Image0001.flo
    #  - Example 2: pattern="{stem}_1" + ext ".exr" -> Image0001_1.exr
    #  - Example 3: pattern="Image{index:04d}_1" (pre-format before +ext) -> not supported here; use {index} tokens.
    ap.add_argument("--gt_pattern", type=str, default="{stem}", help="Filename pattern without extension (uses {stem}, {index}, {index1})")
    ap.add_argument("--gt_ext", type=str, default=None, choices=[None, ".flo", ".exr", ".npy"], help="Force GT extension or try order .flo,.exr,.npy")
    ap.add_argument("--use_sam", action="store_true")
    args = ap.parse_args()

    model = build_model(args.ckpt or "", use_sam=args.use_sam)
    run_compare(model, args.frames, args.gt_dir, args.out_dir,
                T=max(2, args.T), resize_h=args.resize_h, resize_w=args.resize_w,
                gt_pattern=args.gt_pattern, gt_ext=args.gt_ext)

if __name__ == "__main__":
    main()
"""
python debug_compare_flow.py \
  --ckpt /home/serverai/ltdoanh/AniUnFlow/workspace_aft/best.pth \
  --frames /home/serverai/ltdoanh/AniUnFlow/data/AnimeRun_v2/train/Frame_Anime/agent_basement2_weapon_approach/color_1 \
  --gt_dir /home/serverai/ltdoanh/AniUnFlow/data/AnimeRun_v2/train/Flow/agent_basement2_weapon_approach/forward \
  --out_dir /home/serverai/ltdoanh/AniUnFlow/debug_viz \
  --T 5 --resize_h 368 --resize_w 768 \
  --gt_pattern "Image{stem}" --gt_ext ".exr"

"""