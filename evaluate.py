import argparse
from utils.config import load_config, override_by_cli
# tools/eval_clip.py
from __future__ import annotations
import argparse, json, yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from engine import UnsupervisedClipTrainer, UnsupervisedFlowTrainer
from dataio import *

def build_val_loader(cfg, type, override_root=None):
    dcfg = cfg["data"]
    root = override_root or dcfg.get("val_root") or dcfg["train_root"]
    if type == "clip":
        ds = UnlabeledClipDataset(
            root=root,
            T=int(dcfg.get("T", 5)),
            stride_min=int(dcfg.get("stride_min", 1)),
            stride_max=int(dcfg.get("stride_max", 2)),
            crop_size=tuple(dcfg.get("crop_size", [368, 768])),
            color_jitter=tuple(dcfg.get("color_jitter", [0.0,0.0,0.0,0.0])),
            do_flip=bool(dcfg.get("do_flip", False)),
            grayscale_p=float(dcfg.get("grayscale_p", 0.0)),
            resize=bool(dcfg.get("resize", True)),
            keep_aspect=bool(dcfg.get("keep_aspect", True)),
            pad_mode=str(dcfg.get("pad_mode", "reflect")),
            is_test=True,  # turn off heavy augs for eval
        )
    else:
        ds = Animerun(root=cfg["data"]["val_root"], stride_min=1, stride_max=1,
                                    crop_size=tuple(cfg["data"]["crop_size"]),
                                    color_jitter=None, do_flip=False, grayscale_p=0.0,
                                    img_exts=cfg["data"].get("img_exts"),
                                    is_test=True)
    dl = DataLoader(
        ds,
        batch_size=int(dcfg.get("batch_size", 4)),
        shuffle=False,
        num_workers=int(dcfg.get("num_workers", 4)),
        drop_last=False,
        pin_memory=True,
    )
    return dl

def load_ckpt(trainer: UnsupervisedClipTrainer, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    trainer.model.load_state_dict(ckpt["state_dict"], strict=True)
    print(f"Loaded checkpoint: {ckpt_path}")

def clip_main(args):

    cfg = yaml.safe_load(open(args.cfg, "r"))
    if args.batch is not None:
        cfg["data"]["batch_size"] = args.batch

    ws = Path(cfg.get("workspace", "outputs/eval_only"))
    ws.mkdir(parents=True, exist_ok=True)

    # Build trainer (no need to pass a real train loader)
    trainer = UnsupervisedClipTrainer(args=None, cfg=cfg, workspace=ws)
    load_ckpt(trainer, args.ckpt)

    # Build val loader
    val_loader = build_val_loader(cfg,type="clip" override_root=args.val_root)

    # Run validate
    metrics = trainer.validate(val_loader, epoch=0, full_pass=args.full_pass)
    print("==== EVAL METRICS ====")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(metrics, f, indent=2)

def main(args):
    
    cfg = load_config(args.config)
    if args.batch is not None:
        cfg["data"]["batch_size"] = args.batch

    ws = Path(cfg.get("workspace", "outputs/eval_only"))
    ws.mkdir(parents=True, exist_ok=True)

    # Build trainer (no need to pass a real train loader)
    trainer = UnsupervisedFlowTrainer(args=None, cfg=cfg, workspace=ws)
    load_ckpt(trainer, args.ckpt)

    # Build val loader
    val_loader = build_val_loader(cfg,type="clip", override_root=args.val_root)

    # Run validate
    metrics = trainer.validate(val_loader, epoch=0, full_pass=args.full_pass)
    print("==== EVAL METRICS ====")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(metrics, f, indent=2)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('overrides', nargs='*', help='key=val pairs to override config')
    ap.add_argument("--clip", action="store_true", help="Use clip trainer")
    args = ap.parse_args()
    if args.clip:
        clip_main(args)
    else:
        main(args)