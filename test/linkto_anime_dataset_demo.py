# file: linkto_anime_dataset_demo.py
from __future__ import annotations
import argparse, random, cv2, numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
from dataio.linkto_anime_clip_dataset import LinkToAnimeClipDataset


# ============ Simple test runner ============
def demo_shapes(args):
    ds = LinkToAnimeClipDataset(
        root=args.root,
        split=args.split,
        T=args.T,
        crop_size=(args.H, args.W),
        resize=not args.no_resize,
        keep_aspect=args.keep_aspect,
        pad_mode="reflect",
        limit=args.limit
    )
    print("\n[Sample check]")
    s = ds[0]
    print(" keys:", list(s.keys()))
    print(" clip:", tuple(s["clip"].shape))           # (T,3,H,W)
    print(" stride / center / seq_id:", s["stride"], s["center"], s["seq_id"])
    if "flow_list" in s:
        print(" flow_list len:", len(s["flow_list"]))
        print(" flow_list[0]:", tuple(s["flow_list"][0].shape))  # (2,H,W)

    print("\n[Batch check]")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    b = next(iter(loader))
    print(" batch clip:", tuple(b["clip"].shape))  # (B,T,3,H,W)
    if "flow_list" in b:
        print(" batch flow_list len:", len(b["flow_list"]))      # T-1
        print(" batch flow_list[0]:", tuple(b["flow_list"][0].shape))  # (B,2,H,W)
    print("\n[Done]")

def parse_args():
    p = argparse.ArgumentParser(description="LinkTo-Anime dataset shape demo")
    p.add_argument("--root", type=str, default="data", help="Folder that contains train/val/test")
    p.add_argument("--split", type=str, default="val", choices=["train","val","test"])
    p.add_argument("--T", type=int, default=5)
    p.add_argument("--H", type=int, default=368)
    p.add_argument("--W", type=int, default=768)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--no_resize", action="store_true")
    p.add_argument("--keep_aspect", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="Limit number of sequences/clips for quick test")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    demo_shapes(args)

"""
python -m test.linkto_anime_dataset_demo \
  --root /home/serverai/ltdoanh/AniUnFlow/data/LinkTo-Anime \
  --split val \
  --T 5 --H 368 --W 768 \
  --batch_size 2 --num_workers 2

"""