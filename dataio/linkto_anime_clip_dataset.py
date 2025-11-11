# file: data/linkto_anime_clip_dataset.py
from __future__ import annotations
import os, re, random, cv2, numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image

# ------------------------------ io helpers ------------------------------ #
_VALID_IMG_EXT = {".png", ".jpg", ".jpeg"}
_VALID_EXR_EXT = {".exr"}

def _list_dir_sorted(p: Path) -> List[Path]:
    return sorted([q for q in p.iterdir() if q.is_dir()])

def _list_files_sorted(p: Path, exts: set) -> List[Path]:
    return sorted([q for q in p.iterdir() if q.is_file() and q.suffix.lower() in exts])

def _find_subdir_contains(parent: Path, keyword: str) -> Optional[Path]:
    """Find first immediate subdir whose name contains keyword (case-insensitive)."""
    kw = keyword.lower()
    for d in _list_dir_sorted(parent):
        if kw in d.name.lower():
            return d
    return None

def _read_image(path: Path) -> Image.Image:
    # Pillow reads PNG/JPG reliably; if you already have frame_utils.read_gen, swap in here
    return Image.open(str(path)).convert("RGB")

def _read_exr_flow(path: Path) -> np.ndarray:
    """
    Read .exr flow into float32 (H,W,2).
    Tries OpenEXR → OpenCV → imageio.v3. Takes first two channels as (u,v).
    """
    # 1) OpenEXR (if available)
    try:
        import OpenEXR, Imath, array
        exr = OpenEXR.InputFile(str(path))
        print("Reading EXR flow with OpenEXR:", path)
        dw = exr.header()['dataWindow']
        W = dw.max.x - dw.min.x + 1
        H = dw.max.y - dw.min.y + 1
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        # common channel names
        for chans in (('R','G'), ('X','Y'), ('U','V'), ('u','v')):
            try:
                R = np.frombuffer(exr.channel(chans[0], pt), dtype=np.float32).reshape(H, W)
                G = np.frombuffer(exr.channel(chans[1], pt), dtype=np.float32).reshape(H, W)
                return np.stack([R, G], axis=-1)
            except Exception:
                continue
        # fallback read all channels then slice first two
        ch_names = list(exr.header()['channels'].keys())
        ch_names.sort()
        planes = [np.frombuffer(exr.channel(c, pt), dtype=np.float32).reshape(H, W) for c in ch_names[:2]]
        return np.stack(planes, axis=-1).astype(np.float32)
    except Exception:
        pass

    # 2) OpenCV (>=4.5 builds usually support EXR)
    try:
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)  # float32, channels 2/3/4
        if arr is None:
            raise RuntimeError("cv2 failed")
        if arr.ndim == 2:  # single-channel: impossible for flow; duplicate to (2,)
            arr = np.stack([arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] >= 2:
            # OpenCV returns BGR(A); take first two channels as (u,v).
            arr = arr[:, :, :2]
        return arr.astype(np.float32)
    except Exception:
        pass

    # 3) imageio.v3
    try:
        import imageio.v3 as iio
        arr = iio.imread(str(path))  # float array
        if arr.ndim == 2:
            arr = np.stack([arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] >= 2:
            arr = arr[:, :, :2]
        return arr.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Cannot read EXR flow: {path} ({e})")

# ------------------------------ resize helpers ------------------------------ #
def _choose_interp(h0, w0, H, W):
    return cv2.INTER_AREA if (H < h0 or W < w0) else cv2.INTER_LINEAR

def _resize_img_direct(img_np: np.ndarray, H: int, W: int) -> np.ndarray:
    h0, w0 = img_np.shape[:2]
    return cv2.resize(img_np, (W, H), interpolation=_choose_interp(h0, w0, H, W))

def _resize_img_keep_aspect_and_pad(img_np: np.ndarray, H: int, W: int, pad_mode: str) -> np.ndarray:
    h0, w0 = img_np.shape[:2]
    scale = min(W / float(w0), H / float(h0))
    newW, newH = max(1, int(round(w0 * scale))), max(1, int(round(h0 * scale)))
    img_rs = cv2.resize(img_np, (newW, newH), interpolation=_choose_interp(h0, w0, newH, newW))
    pad_h, pad_w = H - newH, W - newW
    top, left = pad_h // 2, pad_w // 2
    bottom, right = pad_h - top, pad_w - left
    border_mode = cv2.BORDER_REFLECT_101 if pad_mode == "reflect" else cv2.BORDER_CONSTANT
    return cv2.copyMakeBorder(img_rs, top, bottom, left, right, border_mode, value=0)

def _resize_flow_direct(flow: np.ndarray, H: int, W: int) -> np.ndarray:
    H0, W0 = flow.shape[:2]
    fx, fy = W / float(W0), H / float(H0)
    flow_rs = cv2.resize(flow, (W, H), interpolation=cv2.INTER_LINEAR)
    flow_rs[..., 0] *= fx
    flow_rs[..., 1] *= fy
    return flow_rs

def _resize_flow_keep_aspect_and_pad(flow: np.ndarray, H: int, W: int, pad_mode: str) -> np.ndarray:
    H0, W0 = flow.shape[:2]
    scale = min(W / float(W0), H / float(H0))
    newW, newH = max(1, int(round(W0 * scale))), max(1, int(round(H0 * scale)))
    flow_rs = cv2.resize(flow, (newW, newH), interpolation=cv2.INTER_LINEAR)
    flow_rs *= scale
    pad_h, pad_w = H - newH, W - newW
    top, left = pad_h // 2, pad_w // 2
    bottom, right = pad_h - top, pad_w - left
    border_mode = cv2.BORDER_CONSTANT  # 0 motion outside
    return cv2.copyMakeBorder(flow_rs, top, bottom, left, right, border_mode, value=0.0)

# ------------------------------ scan split ------------------------------ #
def _scan_linktoanime_split(root: Path, split: str) -> List[Dict[str, Any]]:
    """
    Return list of sequences for a split (train/val/test).
    Each item: {"frames": [Path...], "flow_fw": [Path...]}.
    We treat each 'visual_view' (or similar) as one sequence.
    """
    split_dir = root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split not found: {split_dir}")

    seqs: List[Dict[str, Any]] = []
    # walk scenes: e.g., color_line_optic_01_25, 01 model, 01 visual_view, ...
    for scene_dir in _list_dir_sorted(split_dir):
        # some scenes have sub-buckets (e.g., '01 model', '02 model', or directly view dirs)
        stack = []
        subdirs = _list_dir_sorted(scene_dir)
        if not subdirs:
            stack = [scene_dir]
        else:
            # flatten two levels: scene_dir/* (e.g., '01 model') then inside that, views
            for d1 in subdirs:
                views = [d1] if not _list_dir_sorted(d1) else _list_dir_sorted(d1)
                for view_dir in views:
                    stack.append(view_dir)

        for view_dir in stack:
            # find rendering & flow dirs inside this view_dir
            rend = _find_subdir_contains(view_dir, "Rendering")
            flow = _find_subdir_contains(view_dir, "Optical_flow_exr")
            if rend is None or flow is None:
                # some layouts may nest one more level; scan deeper once
                for d2 in _list_dir_sorted(view_dir):
                    rend = rend or _find_subdir_contains(d2, "Rendering")
                    flow = flow or _find_subdir_contains(d2, "Optical_flow_exr")
                if rend is None or flow is None:
                    continue  # skip if incomplete

            frames = _list_files_sorted(rend, _VALID_IMG_EXT)
            flows  = _list_files_sorted(flow, _VALID_EXR_EXT)
            if len(frames) < 3 or len(flows) < 1:
                continue

            # Align lengths (assume flow[k] = frames[k] -> frames[k+1])
            max_pairs = min(len(frames) - 1, len(flows))
            frames = frames[: max_pairs + 1]
            flows  = flows[: max_pairs]

            if max_pairs >= 1:
                seqs.append({"frames": frames, "flow_fw": flows})
    return seqs

# ------------------------------ dataset ------------------------------ #
class LinkToAnimeClipDataset(Dataset):
    """
    Clip dataset compatible with AnimeRun_v2 API.
    - Train: images only (no GT).
    - Test/Val (is_test=True): images + GT forward flows from .exr, returns 'flow_list' (length T-1).

    Output:
      sample = {
        "clip": FloatTensor[T,3,H,W] in [0,1],
        "stride": int,
        "center": int,
        "seq_id": int,
        # test only:
        "flow_list": List[FloatTensor[2,H,W]]
      }
    """
    def __init__(self, root: str,
                 split: str = "train",
                 T: int = 5,
                 stride_min: int = 1, stride_max: int = 2,
                 crop_size: Tuple[int,int] = (368, 768),
                 resize: bool = True,
                 keep_aspect: bool = False,
                 pad_mode: str = "reflect",
                 color_jitter: Optional[Tuple[float,float,float,float]] = (0.1,0.1,0.1,0.02),
                 do_flip: bool = True,
                 grayscale_p: float = 0.0,
                 limit: Optional[int] = None
                 ):
        assert T >= 3 and T % 2 == 1
        self.root = Path(root)
        self.split = split
        self.T = T
        self.L = T // 2

        self.is_test = split.lower() in {"val", "test"}
        self.smin = 1 if self.is_test else stride_min
        self.smax = 1 if self.is_test else stride_max

        self.H, self.W = crop_size
        self.resize_enable = resize
        self.keep_aspect = keep_aspect
        assert pad_mode in ("reflect","constant")
        self.pad_mode = pad_mode

        self.do_flip = do_flip and (not self.is_test)
        self.color_jitter = color_jitter if (not self.is_test) else None
        self.gray_p = grayscale_p if (not self.is_test) else 0.0

        # scan sequences from split
        self.seqs = _scan_linktoanime_split(self.root, self.split)
        print(f"[{self.split}] Found {len(self.seqs)} sequences under {self.root/self.split}")
        if limit is not None:
            self.seqs = self.seqs[:limit]
            print(f"  - limit applied, using first {limit} sequences")
        # build index
        self.index: List[Tuple[int,int,int]] = []
        for sid, seq in enumerate(self.seqs):
            n = len(seq["frames"])
            for s in range(self.smin, self.smax+1):
                for t in range(self.L * s, n - self.L * s):
                    self.index.append((sid, t, s))

    def __len__(self): return len(self.index)

    # augment (train only)
    def _augment_clip(self, imgs: List[Image.Image]) -> List[Image.Image]:
        if self.do_flip and random.random() < 0.5:
            imgs = [TF.hflip(x) for x in imgs]
        if self.color_jitter is not None:
            b,c,s,h = self.color_jitter
            db = (random.random()*2-1)*b
            dc = (random.random()*2-1)*c
            ds = (random.random()*2-1)*s
            dh = (random.random()*2-1)*h
            imgs = [TF.adjust_brightness(x, 1+db) for x in imgs]
            imgs = [TF.adjust_contrast(x,  1+dc) for x in imgs]
            imgs = [TF.adjust_saturation(x,1+ds) for x in imgs]
            imgs = [TF.adjust_hue(x,       dh)   for x in imgs]
        if random.random() < self.gray_p:
            imgs = [TF.rgb_to_grayscale(x, num_output_channels=3) for x in imgs]
        return imgs

    def _resize_image(self, img_np: np.ndarray) -> np.ndarray:
        if not self.resize_enable: return img_np
        if self.keep_aspect: return _resize_img_keep_aspect_and_pad(img_np, self.H, self.W, self.pad_mode)
        return _resize_img_direct(img_np, self.H, self.W)

    def _resize_flow(self, flow_np: np.ndarray) -> np.ndarray:
        if not self.resize_enable: return flow_np
        if self.keep_aspect: return _resize_flow_keep_aspect_and_pad(flow_np, self.H, self.W, self.pad_mode)
        return _resize_flow_direct(flow_np, self.H, self.W)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sid, t, s = self.index[idx]
        seq = self.seqs[sid]
        frames = seq["frames"]
        picks = [t + k*s for k in range(-self.L, self.L+1)]

        # images
        imgs_pil = [_read_image(frames[p]) for p in picks]
        if not self.is_test:
            imgs_pil = self._augment_clip(imgs_pil)
        imgs_np = [np.asarray(im)[..., :3].astype(np.uint8) for im in imgs_pil]
        imgs_np = [self._resize_image(x) for x in imgs_np]
        imgs_t = [torch.from_numpy(x).permute(2,0,1).float()/255.0 for x in imgs_np]
        clip = torch.stack(imgs_t, dim=0)  # (T,3,H,W)

        sample: Dict[str, Any] = {"clip": clip, "stride": s, "center": t, "seq_id": sid}

        # attach GT on val/test
        if self.is_test:
            flows = seq["flow_fw"]  # assume flows[k] = frames[k] -> frames[k+1]
            gt_list: List[torch.Tensor] = []
            for i in range(self.T - 1):
                pair_start_idx = min(picks[i], len(flows)-1)  # safety
                f_np = _read_exr_flow(flows[pair_start_idx])  # (H,W,2)
                f_np = self._resize_flow(f_np)
                gt_list.append(torch.from_numpy(f_np).permute(2,0,1).float())
            sample["flow_list"] = gt_list

        return sample
