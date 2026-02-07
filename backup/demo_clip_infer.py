#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo inference for AniFlowFormer-T (clip-based, one-pass).
- Đọc thư mục ảnh (PNG/JPG), sắp xếp theo tên
- Chia sliding window độ dài T (stride 1), mỗi window -> model(clip)
- Lấy flow cho cặp (t -> t+1) từ window bắt đầu tại t
- Xuất video: ghép dọc [ảnh; flow-color] cho từng cặp liên tiếp

Example:
python demo_clip_infer.py \
  --ckpt outputs/aniflowformer_t_linkto_anime/best.pth \
  --frames /path/to/frames \
  --out outputs/flow_vis.mp4 \
  --T 5 --fps 24 --resize_h 368 --resize_w 768
"""
from __future__ import annotations

import argparse, os, glob
import numpy as np
from typing import List, Tuple
from PIL import Image

import torch
import torch.nn.functional as F
import cv2
from utils.flow_viz import flow_to_image
# ---- import your model ----
from models import AniFlowFormerT, ModelConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------- utils -------------
def list_images(folder: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg")
    files = []
    for e in exts:
        files += glob.glob(os.path.join(folder, e))
    files.sort()
    return files

def load_image_rgb(path: str) -> torch.Tensor:
    """Return tensor [3,H,W] float32 in [0,255]."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.uint8)
    ten = torch.from_numpy(arr).permute(2, 0, 1).float()
    return ten

def pad_to_multiple(img: torch.Tensor, mult: int = 16) -> Tuple[torch.Tensor, Tuple[int,int,int,int]]:
    """Pad [C,H,W] to make H,W divisible by mult. Returns padded tensor and pad (l,r,t,b)."""
    _, H, W = img.shape
    pad_h = (mult - (H % mult)) % mult
    pad_w = (mult - (W % mult)) % mult
    pad = (0, pad_w, 0, pad_h)  # left,right,top,bottom in F.pad (W then H)
    out = F.pad(img, pad, mode="replicate")
    return out, (0, pad_w, 0, pad_h)

def unpad_flow(flow: torch.Tensor, pad: Tuple[int,int,int,int]) -> torch.Tensor:
    """Undo pad for [1,2,H,W] or [2,H,W]."""
    l, r, t, b = pad
    if flow.dim() == 4:
        _, _, H, W = flow.shape
        return flow[..., :H-b, :W-r]
    else:
        H, W = flow.shape[-2:]
        return flow[..., :H-b, :W-r]

def make_vis_frame(img: torch.Tensor, flo: torch.Tensor) -> np.ndarray:
    """
    img: [1,3,H,W] float(0..255) or (0..1)
    flo: [1,2,H,W] float (pixels)
    return: np.uint8 BGR, stacked vertically [image; flow]
    
    Uses Middlebury color-wheel for flow visualization (symmetric clipping, 95th percentile normalization).
    """
    # to uint8 RGB
    im = img[0]
    if im.max() <= 1.001:
        im = (im * 255.0).clamp(0,255)
    img_np = im.permute(1,2,0).detach().cpu().numpy().astype(np.uint8)

    flow_np = flo[0].permute(1,2,0).detach().cpu().numpy()

    # quick stats for debugging near-constant/NaN flows
    def _stat(name, t: torch.Tensor | np.ndarray):
        try:
            if isinstance(t, np.ndarray):
                arr = t
                has_nan = np.isnan(arr).any()
                mn = float(np.nanmin(arr))
                mx = float(np.nanmax(arr))
                mean = float(np.nanmean(arr))
            else:
                arr = t.detach().cpu()
                has_nan = torch.isnan(arr).any().item()
                mn = float(torch.nanmin(arr))
                mx = float(torch.nanmax(arr))
                mean = float(torch.nanmean(arr))
            print(f"{name}: min={mn:.4f} max={mx:.4f} has_nan={has_nan} mean={mean:.4f}")
        except Exception:
            pass

    _stat("flow_u", flow_np[..., 0])
    _stat("flow_v", flow_np[..., 1])

    # Use Middlebury color-wheel visualization directly
    flow_color = flow_to_image(flow_np)  # RGB uint8 [H,W,3]

    vis_rgb = np.concatenate([img_np, flow_color], axis=0)
    vis_bgr = vis_rgb[..., ::-1]
    return vis_bgr


# ------------- model I/O -------------
def build_model(ckpt_path: str, use_sam: bool = False) -> AniFlowFormerT:
    cfg = ModelConfig(use_sam=use_sam)
    model = AniFlowFormerT(cfg).to(DEVICE)
    model.eval()

    if ckpt_path and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        # accept state under various keys
        state = ckpt.get("state_dict", ckpt.get("model", ckpt))
        # strip "module." if any
        new_state = {}
        for k,v in state.items():
            nk = k[7:] if k.startswith("module.") else k
            new_state[nk] = v
        missing, unexpected = model.load_state_dict(new_state, strict=False)
        if missing:
            print("[Warn] Missing keys:", len(missing))
        if unexpected:
            print("[Warn] Unexpected keys:", len(unexpected))
    else:
        print("[Info] Run with random-init weights (no checkpoint provided).")
    return model


# ------------- inference core -------------
@torch.no_grad()
def run_folder(model: AniFlowFormerT, frame_dir: str, out_path: str,
               T: int = 5, resize_h: int | None = None, resize_w: int | None = None,
               fps: float = 24.0):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    files = list_images(frame_dir)
    if len(files) < 2:
        raise RuntimeError(f"Found {len(files)} image(s) in {frame_dir}. Need >= 2.")

    # load and (optional) resize/pad first frame to allocate writer
    img0 = load_image_rgb(files[0])  # [3,H,W] in 0..255
    if resize_h and resize_w:
        img0 = F.interpolate(img0.unsqueeze(0), size=(resize_h, resize_w),
                             mode="bilinear", align_corners=True)[0]
    img0, pad = pad_to_multiple(img0, mult=16)

    # one dry forward to get vis size
    if len(files) >= T:
        clip = [img0]  # already padded
        for p in files[1:T]:
            im = load_image_rgb(p)
            if resize_h and resize_w:
                im = F.interpolate(im.unsqueeze(0), size=(resize_h, resize_w),
                                   mode="bilinear", align_corners=True)[0]
            im, _ = pad_to_multiple(im, mult=16)
            clip.append(im)
        clip_t = torch.stack(clip, dim=0)  # [T,3,H,W]
        clip_t = clip_t.unsqueeze(0).to(DEVICE) / 255.0  # [1,T,3,H,W] in 0..1
        out = model(clip_t)
        flows = out["flows"]  # list of [B,2,H4,W4], length = T-1

        # upsample flow to image size and unpad for vis
        flow_s = flows[0]
        f_up = F.interpolate(flow_s, size=img0.shape[-2:], mode="bilinear",
                             align_corners=True)
        # correct scaling: scale u by W ratio and v by H ratio
        H0, W0 = img0.shape[-2:]
        h_pred, w_pred = flow_s.shape[-2:]
        sx = W0 / float(w_pred)
        sy = H0 / float(h_pred)
        f_up[:, 0] *= sx
        f_up[:, 1] *= sy
        f_up = unpad_flow(f_up, pad)
        img0_unpad = img0.unsqueeze(0) / 255.0
        img0_unpad = img0_unpad[..., :f_up.shape[-2], :f_up.shape[-1]]  # safety crop
        first_bgr = make_vis_frame(img0_unpad*255.0, f_up)

    # end if len(files) >= T
    else:
        # just pair the first two
        img1 = load_image_rgb(files[1])
        if resize_h and resize_w:
            img1 = F.interpolate(img1.unsqueeze(0), size=(resize_h, resize_w),
                                 mode="bilinear", align_corners=True)[0]
        img1, _ = pad_to_multiple(img1, mult=16)
        clip_t = torch.stack([img0, img1], dim=0).unsqueeze(0).to(DEVICE) / 255.0
        out = model(clip_t)
        flow_s = out["flows"][0]
        f_up = F.interpolate(flow_s, size=img0.shape[-2:], mode="bilinear",
                             align_corners=True)
        H0, W0 = img0.shape[-2:]
        h_pred, w_pred = flow_s.shape[-2:]
        sx = W0 / float(w_pred)
        sy = H0 / float(h_pred)
        f_up[:, 0] *= sx
        f_up[:, 1] *= sy
        f_up = unpad_flow(f_up, pad)
        first_bgr = make_vis_frame((img0.unsqueeze(0)/255.0)*255.0, f_up)

    H_vis, W_vis = first_bgr.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W_vis, H_vis))
    writer.write(first_bgr)

    # sliding window T with stride 1; for each start s, take flows[0] as (s->s+1)
    imgs: List[torch.Tensor] = [img0]  # padded images cache
    for p in files[1:]:
        im = load_image_rgb(p)
        if resize_h and resize_w:
            im = F.interpolate(im.unsqueeze(0), size=(resize_h, resize_w),
                               mode="bilinear", align_corners=True)[0]
        im, _ = pad_to_multiple(im, mult=16)
        imgs.append(im)

    # process pairs
    for s in range(0, len(imgs)-1):
        # window [s : s+T]
        sub = imgs[s:s+T]
        if len(sub) < 2:
            break
        clip = torch.stack(sub, dim=0).unsqueeze(0).to(DEVICE) / 255.0  # [1,T',3,H,W]
        out = model(clip)
        flows = out["flows"]           # length = T'-1
        flow_s = flows[0]              # [1,2,h,w] for pair (s->s+1)
        # upsample & unpad to original (unpad size equals first img unpad)
        f_up = F.interpolate(flow_s, size=imgs[s].shape[-2:], mode="bilinear",
                             align_corners=True)
        H0, W0 = imgs[s].shape[-2:]
        h_pred, w_pred = flow_s.shape[-2:]
        sx = W0 / float(w_pred)
        sy = H0 / float(h_pred)
        f_up[:, 0] *= sx
        f_up[:, 1] *= sy
        f_up = unpad_flow(f_up, pad)

        img_vis = (imgs[s].unsqueeze(0)/255.0)
        img_vis = img_vis[..., :f_up.shape[-2], :f_up.shape[-1]]
        frame_bgr = make_vis_frame(img_vis*255.0, f_up)
        if frame_bgr.shape[0] != H_vis or frame_bgr.shape[1] != W_vis:
            frame_bgr = cv2.resize(frame_bgr, (W_vis, H_vis), interpolation=cv2.INTER_AREA)
        writer.write(frame_bgr)

    writer.release()
    print(f"[OK] Saved optical-flow video to: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=False, help="Path to checkpoint .pth (state_dict or payload)")
    ap.add_argument("--frames", type=str, required=True, help="Folder containing frames (png/jpg)")
    ap.add_argument("--out", type=str, default="outputs/flow_vis.mp4", help="Output video path")
    ap.add_argument("--T", type=int, default=5, help="Clip length (>=2). If fewer frames remain, will run shorter clip.")
    ap.add_argument("--fps", type=float, default=24.0, help="FPS for output video")
    ap.add_argument("--resize_h", type=int, default=368, help="Resize H before pad (keep_aspect=false)")
    ap.add_argument("--resize_w", type=int, default=768, help="Resize W before pad (keep_aspect=false)")
    ap.add_argument("--use_sam", action="store_true", help="Enable SAM-guided path (requires masks, not used in this demo)")
    args = ap.parse_args()

    model = build_model(args.ckpt or "", use_sam=args.use_sam)
    run_folder(model, args.frames, args.out, T=max(2, args.T),
               resize_h=args.resize_h, resize_w=args.resize_w, fps=args.fps)


if __name__ == "__main__":
    main()

"""
python demo_clip_infer.py \
  --ckpt /home/serverai/ltdoanh/AniUnFlow/linktoanime/best.pth \
  --frames /home/serverai/ltdoanh/AniUnFlow/runs/01_visual_view/01_1_Rendering \
  --out /home/serverai/ltdoanh/AniUnFlow/linktoanime/flow_vis.mp4 \
  --T 5 \
  --fps 24 \
  --resize_h 368 --resize_w 768

python demo_clip_infer.py \
  --ckpt /home/serverai/ltdoanh/AniUnFlow/workspace_aft/best.pth\
  --frames /home/serverai/ltdoanh/AniUnFlow/data/AnimeRun_v2/train/Frame_Anime/agent_basement2_weapon_approach/color_1 \
  --out /home/serverai/ltdoanh/AniUnFlow/linktoanime/flow_vis.mp4 \
  --T 5 \
  --fps 24 \
  --resize_h 368 --resize_w 768

"""