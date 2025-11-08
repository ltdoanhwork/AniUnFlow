from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional
import random
import cv2
from utils import frame_utils
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image


class UnlabeledPairDataset(Dataset):
    """Samples (I_t, I_{t+Δ}) pairs. Supports resize (with optional keep-aspect + pad) and flow scaling."""
    def __init__(self, root: str,
                 stride_min: int = 1, stride_max: int = 2,
                 crop_size: Tuple[int,int] = (368,768),
                 color_jitter: Optional[Tuple[float,float,float,float]] = None,
                 do_flip: bool = True,
                 grayscale_p: float = 0.0,
                 is_test: bool = False,
                 # --- new resize options ---
                 resize: bool = True,
                 keep_aspect: bool = False,
                 pad_mode: str = "reflect"  # "reflect" | "constant"
                 ):
        self.root = Path(root)
        self.flow_list: List[str] = []
        self.image_list: List[Tuple[str, str]] = []
        self.extra_info: List = []

        self.occ_list: List[str] = []
        self.line_list: List[str] = []

        self.smin, self.smax = stride_min, stride_max
        self.H, self.W = crop_size
        self.color_jitter = color_jitter
        self.do_flip = do_flip
        self.gray_p = grayscale_p
        self.is_test = is_test

        # resize options
        self.resize_enable = resizex
        self.keep_aspect = keep_aspect
        assert pad_mode in ("reflect", "constant")
        self.pad_mode = pad_mode

    def __len__(self):
        return len(self.image_list)

    # ---------------------- aug ----------------------
    def _augment_pair(self, i1, i2):
        # Random horizontal flip
        if self.do_flip and random.random() < 0.5:
            i1 = TF.hflip(i1); i2 = TF.hflip(i2)
        # Color jitter (brightness, contrast, saturation, hue)
        if self.color_jitter is not None:
            b,c,s,h = self.color_jitter
            fn = TF.adjust_brightness; i1 = fn(i1, 1 + (random.random()*2-1)*b); i2 = fn(i2, 1 + (random.random()*2-1)*b)
            fn = TF.adjust_contrast;  i1 = fn(i1, 1 + (random.random()*2-1)*c); i2 = fn(i2, 1 + (random.random()*2-1)*c)
            fn = TF.adjust_saturation;i1 = fn(i1, 1 + (random.random()*2-1)*s); i2 = fn(i2, 1 + (random.random()*2-1)*s)
            fn = TF.adjust_hue;       i1 = fn(i1, (random.random()*2-1)*h);    i2 = fn(i2, (random.random()*2-1)*h)
        # Occasionally grayscale
        if random.random() < self.gray_p:
            i1 = TF.rgb_to_grayscale(i1, num_output_channels=3)
            i2 = TF.rgb_to_grayscale(i2, num_output_channels=3)
        return i1, i2

    # ------------------- resize core -------------------
    def _choose_interp(self, h0, w0, H, W):
        # downscale -> AREA, upscale -> LINEAR
        if H < h0 or W < w0:
            return cv2.INTER_AREA
        return cv2.INTER_LINEAR

    def _resize_direct(self, img1, img2, flow, H, W):
        """Resize directly to (H,W); scale flow by (sx, sy)."""
        h0, w0 = img1.shape[:2]
        interp_img = self._choose_interp(h0, w0, H, W)
        img1 = cv2.resize(img1, (W, H), interpolation=interp_img)
        img2 = cv2.resize(img2, (W, H), interpolation=interp_img)

        # flow resize + scale
        sx = W / float(w0)
        sy = H / float(h0)
        flow_rs = cv2.resize(flow, (W, H), interpolation=cv2.INTER_LINEAR)
        flow_rs[..., 0] *= sx
        flow_rs[..., 1] *= sy
        return img1, img2, flow_rs

    def _resize_keep_aspect_and_pad(self, img1, img2, flow, H, W):
        """Keep aspect: resize by same scale, then pad to (H,W)."""
        h0, w0 = img1.shape[:2]
        scale = min(W / float(w0), H / float(h0))
        newW = max(1, int(round(w0 * scale)))
        newH = max(1, int(round(h0 * scale)))

        interp_img = self._choose_interp(h0, w0, newH, newW)
        img1 = cv2.resize(img1, (newW, newH), interpolation=interp_img)
        img2 = cv2.resize(img2, (newW, newH), interpolation=interp_img)

        flow_rs = cv2.resize(flow, (newW, newH), interpolation=cv2.INTER_LINEAR)
        flow_rs[..., 0] *= (newW / float(w0))
        flow_rs[..., 1] *= (newH / float(h0))

        # pad to (H,W)
        pad_h = H - newH
        pad_w = W - newW
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        border_mode = cv2.BORDER_REFLECT_101 if self.pad_mode == "reflect" else cv2.BORDER_CONSTANT
        img1 = cv2.copyMakeBorder(img1, top, bottom, left, right, border_mode, value=0)
        img2 = cv2.copyMakeBorder(img2, top, bottom, left, right, border_mode, value=0)
        flow_rs = cv2.copyMakeBorder(flow_rs, top, bottom, left, right, border_mode, value=0)
        return img1, img2, flow_rs

    def _resize_masks_like(self, mask_np: np.ndarray, H: int, W: int, keep_aspect: bool):
        """Resize mask (H0,W0) -> (H,W). If keep_aspect: same rule as images (center pad)."""
        h0, w0 = mask_np.shape[:2]
        if not keep_aspect:
            m_rs = cv2.resize(mask_np, (W, H), interpolation=cv2.INTER_NEAREST)
            return m_rs

        # keep aspect + pad center
        scale = min(W / float(w0), H / float(h0))
        newW = max(1, int(round(w0 * scale)))
        newH = max(1, int(round(h0 * scale)))
        m_rs = cv2.resize(mask_np, (newW, newH), interpolation=cv2.INTER_NEAREST)

        pad_h = H - newH
        pad_w = W - newW
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        border_mode = cv2.BORDER_REFLECT_101 if self.pad_mode == "reflect" else cv2.BORDER_CONSTANT
        m_rs = cv2.copyMakeBorder(m_rs, top, bottom, left, right, border_mode, value=0)
        return m_rs

    # ------------------- main getitem -------------------
    def __getitem__(self, index):
        index = index % len(self.image_list)

        # disable augment at test
        if self.is_test:
            aug_fn = None
        else:
            aug_fn = self._augment_pair

        flow = frame_utils.read_gen(self.flow_list[index])
        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        # augment (works with PIL or tensor)
        if aug_fn is not None:
            img1, img2 = aug_fn(img1, img2)

        # to numpy
        flow = np.array(flow).astype(np.float32)    # (H,W,2)
        img1 = np.array(img1).astype(np.uint8)      # (H,W,3) or (H,W)
        img2 = np.array(img2).astype(np.uint8)

        # ensure 3-ch
        if img1.ndim == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
        if img2.ndim == 2:
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img2 = img2[..., :3]

        # ----------- resize + scale flow -----------
        if self.resize_enable:
            if self.keep_aspect:
                img1, img2, flow = self._resize_keep_aspect_and_pad(img1, img2, flow, self.H, self.W)
            else:
                img1, img2, flow = self._resize_direct(img1, img2, flow, self.H, self.W)

        # valid mask (if not provided): large magnitude cutoff
        valid = ((np.abs(flow[..., 0]) < 1000.0) & (np.abs(flow[..., 1]) < 1000.0)).astype(np.uint8)

        # ----------- optional test masks -----------
        occ = line = None
        if self.is_test:
            occ = np.load(self.occ_list[index])   # assumed (H0,W0) or (H,W)
            line = np.load(self.line_list[index]) # 0=line, >0=flat
            if self.resize_enable:
                occ  = self._resize_masks_like(occ,  self.H, self.W, self.keep_aspect)
                line = self._resize_masks_like(line, self.H, self.W, self.keep_aspect)

        # to torch tensor
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float().contiguous().clone()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float().contiguous().clone()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float().contiguous().clone()
        valid = torch.from_numpy(valid).bool().contiguous().clone()

        if self.is_test:
            occ_t  = torch.from_numpy(occ).contiguous().clone()  if occ  is not None else None
            line_t = torch.from_numpy(line).contiguous().clone() if line is not None else None
            sample = {"image1": img1, "image2": img2, "flow": flow, "valid": valid}
            if occ_t is not None:  sample["occ"]  = occ_t
            if line_t is not None: sample["line"] = line_t
            return sample
        else:
            return {"image1": img1, "image2": img2}
