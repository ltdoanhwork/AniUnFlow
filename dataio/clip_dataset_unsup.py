# file: data/animerun_clip_dataset.py
from __future__ import annotations
import os, cv2, random
from glob import glob
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image
from utils import frame_utils

# ------------------------------ helpers ------------------------------ #
_VALID_IMG_EXT = {".png", ".jpg", ".jpeg"}
_VALID_FLOW_EXT = {".flo", ".png", ".pfm", ".npy"}  # support common formats

def _list_dir_sorted(p: Path) -> List[Path]:
    return sorted([q for q in p.iterdir() if q.is_dir()])

def _list_files_sorted(p: Path, exts: set) -> List[Path]:
    return sorted([q for q in p.iterdir() if q.is_file() and q.suffix.lower() in exts])

def _read_image(path: Path) -> Image.Image:
    return frame_utils.read_gen(str(path))  # returns PIL.Image for images in most RAFT forks

def _read_flow_any(path: Path) -> np.ndarray:
    """
    Read optical flow into float32 array (H,W,2).
    Supports .flo (Middlebury), .png (KITTI), .pfm, .npy (HxWx2).
    """
    ext = path.suffix.lower()
    if ext == ".flo":
        flow = frame_utils.readFlow(str(path))  # Middlebury .flo -> (H,W,2)
        return flow.astype(np.float32)
    if ext == ".png":
        # KITTI .png returns (flow, valid)
        flow, _ = frame_utils.readFlowKITTI(str(path))
        return flow.astype(np.float32)
    if ext == ".pfm":
        data, _ = frame_utils.readPFM(str(path))  # (H,W,2)
        return data.astype(np.float32)
    if ext == ".npy":
        arr = np.load(str(path))
        if arr.ndim == 3 and arr.shape[-1] >= 2:
            return arr[..., :2].astype(np.float32)
    # Fallback: try generic reader (some forks overload read_gen for flow)
    out = frame_utils.read_gen(str(path))
    if isinstance(out, tuple) and len(out) >= 1:
        out = out[0]
    if out.ndim == 3 and out.shape[-1] >= 2:
        return out[..., :2].astype(np.float32)
    raise ValueError(f"Unsupported flow file: {path}")

def _scan_sequences_train(frame_root: Path,
                          frame_subdir_glob: str = "color_*",
                          min_frames: int = 3,
                          merge_color_tracks: bool = False) -> List[List[Path]]:
    """
    Train split: frames only. Return list of frame-path lists (one list per sequence).
    """
    if not frame_root.exists():
        raise FileNotFoundError(f"Frame root not found: {frame_root}")

    seqs: List[List[Path]] = []
    for scene_dir in _list_dir_sorted(frame_root):
        color_dirs = sorted([d for d in scene_dir.glob(frame_subdir_glob) if d.is_dir()])
        if color_dirs:
            if merge_color_tracks:
                frames = []
                for d in color_dirs:
                    frames += _list_files_sorted(d, _VALID_IMG_EXT)
                if len(frames) >= min_frames:
                    seqs.append(frames)
            else:
                for d in color_dirs:
                    frames = _list_files_sorted(d, _VALID_IMG_EXT)
                    if len(frames) >= min_frames:
                        seqs.append(frames)
        else:
            frames = _list_files_sorted(scene_dir, _VALID_IMG_EXT)
            if len(frames) >= min_frames:
                seqs.append(frames)
    return seqs

def _scan_sequences_test(frame_root: Path, flow_root: Path) -> List[Dict[str, Any]]:
    """
    Test split: return a list of dicts, one per SCENE.
    Each dict contains:
      - "frames": List[Path] from Frame_Anime/<scene>/original
      - "flow_fw": List[Path] from Flow/<scene>/forward (assumed to align with consecutive pairs)
      - (optional) "flow_bw": List[Path] from Flow/<scene>/backward
    Assumes per-scene frames are consistent sizes and flows are for stride=1 consecutive pairs.
    """
    if not frame_root.exists():
        raise FileNotFoundError(f"Test frame root not found: {frame_root}")
    if not flow_root.exists():
        raise FileNotFoundError(f"Test flow root not found: {flow_root}")

    seqs: List[Dict[str, Any]] = []
    for scene_dir in _list_dir_sorted(frame_root):
        # use 'original' track to align with GT
        frames_dir = scene_dir / "original"
        if not frames_dir.exists():
            # fallback: if no 'original', use images directly under scene
            frames = _list_files_sorted(scene_dir, _VALID_IMG_EXT)
        else:
            frames = _list_files_sorted(frames_dir, _VALID_IMG_EXT)
        if len(frames) < 3:
            continue

        scene_name = scene_dir.name
        flow_scene_dir = flow_root / scene_name
        fw_dir = flow_scene_dir / "forward"
        bw_dir = flow_scene_dir / "backward"

        flow_fw = _list_files_sorted(fw_dir, _VALID_FLOW_EXT) if fw_dir.exists() else []
        flow_bw = _list_files_sorted(bw_dir, _VALID_FLOW_EXT) if bw_dir.exists() else []

        # We expect len(flow_fw) >= len(frames)-1 for full coverage. If not, we clip to the min.
        if not flow_fw:
            print(f"[WARN] No forward flows for scene '{scene_name}'. Skipping.")
            continue

        # Clip to the shortest between (frames-1) and flow_fw length:
        max_pairs = min(len(frames) - 1, len(flow_fw))
        if max_pairs < len(frames) - 1:
            # If flows are shorter, also clip frames to match.
            frames = frames[: max_pairs + 1]
            flow_fw = flow_fw[: max_pairs]
            if flow_bw:
                flow_bw = flow_bw[: max_pairs]

        seqs.append({"frames": frames, "flow_fw": flow_fw, "flow_bw": flow_bw})
    return seqs

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
    """Resize flow to (H,W) and scale vectors accordingly."""
    H0, W0 = flow.shape[:2]
    fx, fy = W / float(W0), H / float(H0)
    flow_rs = cv2.resize(flow, (W, H), interpolation=cv2.INTER_LINEAR)
    flow_rs[..., 0] *= fx
    flow_rs[..., 1] *= fy
    return flow_rs

def _resize_flow_keep_aspect_and_pad(flow: np.ndarray, H: int, W: int, pad_mode: str) -> np.ndarray:
    """Keep-aspect resize and pad. Scale vectors by isotropic 'scale' and pad zeros."""
    H0, W0 = flow.shape[:2]
    scale = min(W / float(W0), H / float(H0))
    newW, newH = max(1, int(round(W0 * scale))), max(1, int(round(H0 * scale)))
    flow_rs = cv2.resize(flow, (newW, newH), interpolation=cv2.INTER_LINEAR)
    flow_rs *= scale  # both u and v multiplied by the same isotropic scale
    pad_h, pad_w = H - newH, W - newW
    top, left = pad_h // 2, pad_w // 2
    bottom, right = pad_h - top, pad_w - left
    border_mode = cv2.BORDER_CONSTANT  # flow outside image is 0 motion
    return cv2.copyMakeBorder(flow_rs, top, bottom, left, right, border_mode, value=0.0)

# ------------------------------ dataset ------------------------------ #
class UnlabeledClipDataset(Dataset):
    """
    Clip dataset for both training and testing.
    - Train mode (is_test=False): frames only (no GT).
    - Test mode  (is_test=True) : frames + GT flows from test/Flow/... (forward; backward optional).

    Returns:
        {"clip": Tensor[T,3,H,W], "stride": int, "center": int, "seq_id": int}
        + if is_test: "flow_list": List[Tensor[2,H,W]]  # length T-1, aligned with consecutive pairs in the clip
    """
    def __init__(self, root: str,
                 T: int = 5,
                 stride_min: int = 1, stride_max: int = 2,
                 crop_size: Tuple[int,int] = (368,768),
                 color_jitter: Optional[Tuple[float,float,float,float]] = (0.1,0.1,0.1,0.02),
                 do_flip: bool = True,
                 grayscale_p: float = 0.0,
                 resize: bool = True,
                 keep_aspect: bool = False,
                 pad_mode: str = "reflect",
                 load_sam_masks: bool = False,
                 sam_mask_root: Optional[str] = None,
                 is_test: bool = False):
        assert T >= 3 and T % 2 == 1, "Use odd T >= 3 (e.g., 5)"
        self.root = Path(root)
        self.T = T
        self.L = T // 2
        self.is_test = is_test
        self.load_sam_masks = load_sam_masks
        self.sam_mask_root = Path(sam_mask_root) if sam_mask_root else None

        # strides & geometry
        self.smin = 1 if is_test else stride_min
        self.smax = 1 if is_test else stride_max
        self.H, self.W = crop_size
        self.resize_enable = resize
        self.keep_aspect = keep_aspect
        assert pad_mode in ("reflect", "constant")
        self.pad_mode = pad_mode

        # augmentation (train only)
        self.do_flip = do_flip and (not is_test)
        self.color_jitter = color_jitter if (not is_test) else None
        self.gray_p = grayscale_p if (not is_test) else 0.0

        # scan sequences
        if is_test:
            frame_root = self.root / "test" / "Frame_Anime"
            flow_root  = self.root / "test" / "Flow"
            self.test_seqs = _scan_sequences_test(frame_root, flow_root)  # list of dicts
            print(f"[Test] Found {len(self.test_seqs)} scenes under {frame_root}")
        else:
            frame_root = self.root / "train" / "Frame_Anime"
            
            # CRITICAL: When loading SAM masks, use 'original' subdir because:
            # 1. Masks are precomputed for 'original/' only
            # 2. 'original/' has different filenames (Image0186.png vs 0186.png in color_*)
            if self.load_sam_masks and self.sam_mask_root:
                subdir_glob = "original"  # Use original for SAM mask alignment
                print("[Dataset] Using 'original' subdir for SAM mask alignment")
            else:
                subdir_glob = "color_*"
            
            self.train_seqs = _scan_sequences_train(
                frame_root, frame_subdir_glob=subdir_glob, min_frames=3, merge_color_tracks=False
            )
            print(f"[Train] Found {len(self.train_seqs)} sequences under {frame_root}")

        # build index (seq_id, center_t, stride)
        self.index: List[Tuple[int,int,int]] = []
        if is_test:
            for sid, seq in enumerate(self.test_seqs):
                n = len(seq["frames"])
                s = 1
                for t in range(self.L * s, n - self.L * s):
                    # ensure GT exists for all (T-1) pairs within the clip window
                    left = t - self.L * s
                    right = t + self.L * s
                    if right - left >= 1 and right <= n - 1:
                        self.index.append((sid, t, s))
        else:
            for sid, frames in enumerate(self.train_seqs):
                n = len(frames)
                for s in range(self.smin, self.smax + 1):
                    for t in range(self.L * s, n - self.L * s):
                        self.index.append((sid, t, s))

    def __len__(self):
        return len(self.index)

    # ------------ clip-level augment (train only) ------------
    def _augment_clip(self, imgs: List[Image.Image]) -> List[Image.Image]:
        if self.do_flip and random.random() < 0.5:
            imgs = [TF.hflip(x) for x in imgs]
        if self.color_jitter is not None:
            b, c, s, h = self.color_jitter
            db = (random.random() * 2 - 1) * b
            dc = (random.random() * 2 - 1) * c
            ds = (random.random() * 2 - 1) * s
            dh = (random.random() * 2 - 1) * h
            imgs = [TF.adjust_brightness(x, 1 + db) for x in imgs]
            imgs = [TF.adjust_contrast(x,  1 + dc) for x in imgs]
            imgs = [TF.adjust_saturation(x, 1 + ds) for x in imgs]
            imgs = [TF.adjust_hue(x,        dh)    for x in imgs]
        if random.random() < self.gray_p:
            imgs = [TF.rgb_to_grayscale(x, num_output_channels=3) for x in imgs]
        return imgs

    # ------------ resizing ------------
    def _resize_image(self, img_np: np.ndarray) -> np.ndarray:
        if not self.resize_enable:
            return img_np
        if self.keep_aspect:
            return _resize_img_keep_aspect_and_pad(img_np, self.H, self.W, self.pad_mode)
        return _resize_img_direct(img_np, self.H, self.W)

    def _resize_flow(self, flow_np: np.ndarray) -> np.ndarray:
        if not self.resize_enable:
            return flow_np
        if self.keep_aspect:
            return _resize_flow_keep_aspect_and_pad(flow_np, self.H, self.W, self.pad_mode)
        return _resize_flow_direct(flow_np, self.H, self.W)

    def _resize_mask(self, mask_np: np.ndarray) -> np.ndarray:
        """Resize segment mask (S, H, W)."""
        if not self.resize_enable:
            return mask_np
        
        # Transpose to H, W, S for cv2 resize
        mask_hws = mask_np.transpose(1, 2, 0)
        h0, w0 = mask_hws.shape[:2]
        H, W = self.H, self.W
        
        if self.keep_aspect:
             # Similar to keep aspect pad for image, but nearest neighbor for masks?
             # Actually masks are soft (float), so linear is fine. 
             # But if hard masks, nearest.
             # Assuming soft masks from SAM-2 (float).
             scale = min(W / float(w0), H / float(h0))
             newW, newH = max(1, int(round(w0 * scale))), max(1, int(round(h0 * scale)))
             
             mask_rs = cv2.resize(mask_hws, (newW, newH), interpolation=cv2.INTER_LINEAR)
             if mask_rs.ndim == 2: mask_rs = mask_rs[..., None]
             
             pad_h, pad_w = H - newH, W - newW
             top, left = pad_h // 2, pad_w // 2
             bottom, right = pad_h - top, pad_w - left
             
             # Pad with 0
             mask_padded = cv2.copyMakeBorder(mask_rs, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        else:
            mask_padded = cv2.resize(mask_hws, (W, H), interpolation=cv2.INTER_LINEAR)
            
        if mask_padded.ndim == 2:
            mask_padded = mask_padded[..., None]
            
        return mask_padded.transpose(2, 0, 1) # Back to S, H, W

    # ------------------- main -------------------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sid, t, s = self.index[idx]

        # ---- gather frame paths for this clip ----
        if self.is_test:
            frames = self.test_seqs[sid]["frames"]
        else:
            frames = self.train_seqs[sid]

        picks = [t + k * s for k in range(-self.L, self.L + 1)]
        imgs_pil = [_read_image(frames[p]) for p in picks]

        # augment (train only)
        if not self.is_test:
            imgs_pil = self._augment_clip(imgs_pil)

        # to numpy uint8
        imgs_np = [np.asarray(im)[..., :3].astype(np.uint8) for im in imgs_pil]
        # resize images
        imgs_np = [self._resize_image(x) for x in imgs_np]
        # to tensor [0,1]
        imgs_t = [torch.from_numpy(x).permute(2, 0, 1).float() / 255.0 for x in imgs_np]
        clip = torch.stack(imgs_t, dim=0)  # (T,3,H,W) if resized; else original sizes (must be consistent)

        sample: Dict[str, Any] = {"clip": clip, "stride": s, "center": t, "seq_id": sid}

        # ---- load precomputed SAM masks ----
        if self.load_sam_masks and self.sam_mask_root:
            # Structure: sam_mask_root/[train|test]/Frame_Anime/Sequence/frame.pt
            # Need to reconstruct path relative to data root
            # frames[0] is absolute path.
            # Convert to relative path from data root
            
            mask_list = []
            for p in picks:
                frame_path = frames[p]
                # Assuming standard structure: .../Frame_Anime/Sequence/Frame/img.png
                # We need to map this to .../sam_mask_root/.../img.pt
                
                # Robust way: find relative path from 'train/Frame_Anime' or 'test/Frame_Anime'
                try:
                    # Find 'Frame_Anime' in path parts
                    parts = list(frame_path.parts)
                    idx = parts.index('Frame_Anime')
                    # Include the parent (train/test) to match SAM_Masks structure on disk
                    # Frame_Anime is at idx, so train/test is at idx-1
                    rel_parts = parts[max(0, idx-1):]
                    
                    # CRITICAL FIX: Masks are only generated for 'original' subdirectory
                    # But training uses color_1, color_2, etc. Need to remap to 'original'
                    # Structure: .../Scene/color_X/frame.png -> .../Scene/original/frame.pt
                    for i, part in enumerate(rel_parts):
                        if part.startswith('color_'):
                            rel_parts[i] = 'original'
                            break
                    
                    rel_path = Path(*rel_parts)
                    mask_path = self.sam_mask_root / rel_path.parent / (frame_path.stem + ".pt")
                except (ValueError, IndexError):
                    mask_path = Path("nonexistent")
                
                if mask_path.exists():
                    mask = torch.load(mask_path)  # Either (H, W) uint8 or (S, H, W) float
                    if self.resize_enable and mask.ndim in [2, 3]:
                        mask_np = mask.numpy()
                        if mask.ndim == 2:
                            # Integer label map (H, W)
                            from scipy.ndimage import zoom
                            zoom_factors = (self.H / mask_np.shape[0], self.W / mask_np.shape[1])
                            mask_np = zoom(mask_np, zoom_factors, order=0)  # Nearest neighbor for labels
                        else:
                            # Legacy multi-channel (S, H, W)
                            mask_np = self._resize_mask(mask_np)
                        mask = torch.from_numpy(mask_np)
                    mask_list.append(mask)
                else:
                    # Fallback: empty integer label map (background only)
                    mask_list.append(torch.zeros((self.H, self.W), dtype=torch.uint8))
            
            if len(mask_list) == len(picks):
                # Stack: (T, S, H, W) for legacy or (T, H, W) for optimized
                stacked_masks = torch.stack(mask_list, dim=0)
                
                # Detect format and normalize to (T, 1, H, W) uint8 labels
                if stacked_masks.dtype == torch.uint8 and stacked_masks.ndim == 3:
                    # Optimized format: (T, H, W) uint8 integer labels
                    sample["sam_masks"] = stacked_masks.unsqueeze(1)  # (T, 1, H, W)
                elif stacked_masks.dtype in [torch.float16, torch.float32] and stacked_masks.ndim == 4:
                    # Legacy format: (T, S, H, W) float multi-channel masks
                    # Convert to integer labels on-the-fly
                    T, S, H, W = stacked_masks.shape
                    labels = torch.zeros((T, H, W), dtype=torch.uint8)
                    for t in range(T):
                        max_vals, label_t = stacked_masks[t].max(dim=0)  # (H, W)
                        label_t = label_t + 1  # Shift to 1-indexed
                        label_t[max_vals < 0.5] = 0  # Background threshold
                        labels[t] = label_t.to(torch.uint8)
                    sample["sam_masks"] = labels.unsqueeze(1)  # (T, 1, H, W)
                else:
                    # Unexpected format, use as-is
                    sample["sam_masks"] = stacked_masks

        # ---- attach GT flows for test mode ----
        if self.is_test:
            flow_fw_paths = self.test_seqs[sid]["flow_fw"]
            # Map from frame index i to flow file for pair (i -> i+1)
            # Assumption: flow_fw_paths[k] corresponds to frames[k] -> frames[k+1]
            gt_list: List[torch.Tensor] = []
            for i in range(self.T - 1):
                pair_start_idx = picks[i]  # index of frames[] for the left image
                # Safety: clip to available range
                pair_start_idx = min(pair_start_idx, len(flow_fw_paths) - 1)
                flow_np = _read_flow_any(flow_fw_paths[pair_start_idx])  # (H,W,2)
                flow_np = self._resize_flow(flow_np)
                flow_t = torch.from_numpy(flow_np).permute(2, 0, 1).float()  # (2,H,W)
                gt_list.append(flow_t)
            sample["flow_list"] = gt_list  # DataLoader will collate into List[Tensor[B,2,H,W]]

        return sample
