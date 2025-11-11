from __future__ import annotations
import os
from glob import glob
from pathlib import Path
from typing import List, Tuple, Optional
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image
from utils import frame_utils
from tqdm import tqdm
# utility to scan sequences: root/seq_xxx/*.png(jpg)
from pathlib import Path
from typing import List
import os

def _scan_sequences_v2(frame_root: Path,
                       frame_subdir_glob: str = "color_*",
                       min_frames: int = 3,
                       merge_color_tracks: bool = False) -> List[List[Path]]:
    """
    frame_root = .../AnimeRun_v2/train/Frame_Anime
    - Nếu scene có các thư mục con color_*:
        + merge_color_tracks=False: mỗi color_* là 1 sequence.
        + merge_color_tracks=True: gộp tất cả ảnh của các color_* thành 1 sequence dài.
    - Nếu scene chứa ảnh trực tiếp: nhận scene làm 1 sequence.
    """
    valid_ext = {".png", ".jpg", ".jpeg"}
    seqs: List[List[Path]] = []

    if not frame_root.exists():
        raise FileNotFoundError(f"Frame root not found: {frame_root}")

    for scene_dir in sorted([p for p in frame_root.iterdir() if p.is_dir()]):
        color_dirs = sorted([d for d in scene_dir.glob(frame_subdir_glob) if d.is_dir()])

        if color_dirs:
            if merge_color_tracks:
                frames = []
                for d in color_dirs:
                    frames += sorted([q for q in d.iterdir() if q.suffix.lower() in valid_ext])
                if len(frames) >= min_frames:
                    seqs.append(frames)
            else:
                for d in color_dirs:
                    frames = sorted([q for q in d.iterdir() if q.suffix.lower() in valid_ext])
                    if len(frames) >= min_frames:
                        seqs.append(frames)
        else:
            # fallback: scene chứa ảnh trực tiếp
            frames = sorted([q for q in scene_dir.iterdir() if q.suffix.lower() in valid_ext])
            if len(frames) >= min_frames:
                seqs.append(frames)

    return seqs


class UnlabeledClipDataset(Dataset):
    """Return clips of length T: (I_{t-L}, ..., I_{t+L}).
    - Random start index and random stride in [smin, smax].
    - Resize to (H,W) either direct or keep-aspect+pad.
    - Optional color/flip/grayscale augmentation applied consistently across the clip.
    
    Output: dict {"clip": Tensor BxTx3xHxW when collated} per item is Tx3xHxW.
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
                 is_test: bool = False):
        assert T >= 3 and T % 2 == 1, "Use odd T >= 3 (e.g., 5)"
        self.root = Path(root)
        self.T = T
        self.L = T // 2
        self.smin, self.smax = stride_min, stride_max
        self.H, self.W = crop_size
        self.color_jitter = color_jitter
        self.do_flip = do_flip
        self.gray_p = grayscale_p
        self.resize_enable = resize
        self.keep_aspect = keep_aspect
        assert pad_mode in ("reflect","constant")
        self.pad_mode = pad_mode
        self.is_test = is_test
        image_root = Path(os.path.join(root, 'train', 'Frame_Anime'))
        self.seqs = _scan_sequences_v2(image_root, frame_subdir_glob="color_*", min_frames=3, merge_color_tracks=False)
        print(f"Found {len(self.seqs)} sequences under {image_root}")
        # for scene in os.listdir(image_root):
        #     for color_pass in os.listdir(os.path.join(image_root, scene)):
        #         self.seqs = sorted(glob(os.path.join(image_root, scene, color_pass, '*.png')))
        # print(f"Found {len(self.seqs)} sequences under {self.root}")

        # build index of valid (seq_id, center_t, stride)
        self.index: List[Tuple[int,int,int]] = []
        for sid, frames in tqdm(enumerate(self.seqs), desc="Building clip dataset index", total=len(self.seqs)):
            n = len(frames)
            for s in range(self.smin, self.smax+1):
                for t in range(self.L * s, n - self.L * s):
                    self.index.append((sid, t, s))

    def __len__(self):
        return len(self.index)

    # ------- augmentation applied consistently over the clip -------
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

    # ------------------- resize helpers -------------------
    def _choose_interp(self, h0, w0, H, W):
        return cv2.INTER_AREA if (H < h0 or W < w0) else cv2.INTER_LINEAR

    def _resize_direct(self, img_np, H, W):
        h0, w0 = img_np.shape[:2]
        return cv2.resize(img_np, (W, H), interpolation=self._choose_interp(h0,w0,H,W))

    def _resize_keep_aspect_and_pad(self, img_np, H, W):
        h0, w0 = img_np.shape[:2]
        scale = min(W/float(w0), H/float(h0))
        newW, newH = max(1,int(round(w0*scale))), max(1,int(round(h0*scale)))
        img_rs = cv2.resize(img_np, (newW,newH), interpolation=self._choose_interp(h0,w0,newH,newW))
        pad_h, pad_w = H - newH, W - newW
        top, left = pad_h//2, pad_w//2
        bottom, right = pad_h-top, pad_w-left
        border_mode = cv2.BORDER_REFLECT_101 if self.pad_mode=="reflect" else cv2.BORDER_CONSTANT
        return cv2.copyMakeBorder(img_rs, top,bottom,left,right, border_mode, value=0)

    # ------------------- main -------------------
    def __getitem__(self, idx):
        sid, t, s = self.index[idx]
        frames = self.seqs[sid]
        picks = [t + k*s for k in range(-self.L, self.L+1)]
        imgs = [frame_utils.read_gen(str(frames[p])) for p in picks]  # PIL
        if not self.is_test:
            imgs = self._augment_clip(imgs)
        # to numpy
        imgs_np = [np.asarray(im)[..., :3].astype(np.uint8) for im in imgs]
        # resize
        if self.resize_enable:
            if self.keep_aspect:
                imgs_np = [self._resize_keep_aspect_and_pad(x, self.H, self.W) for x in imgs_np]
            else:
                imgs_np = [self._resize_direct(x, self.H, self.W) for x in imgs_np]
        # to tensor [0,1]
        imgs_t = [torch.from_numpy(x).permute(2,0,1).float()/255.0 for x in imgs_np]
        clip = torch.stack(imgs_t, dim=0)  # T x 3 x H x W
        return {"clip": clip, "stride": s, "center": t, "seq_id": sid}