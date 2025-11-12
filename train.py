# file: train.py
import os, json, random
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader


from dataio import *
from engine import UnsupervisedFlowTrainer, UnsupervisedClipTrainer

import yaml




def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)




def main(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)


    set_seed(cfg.get("seed", 1337))
    cudnn.benchmark = True


    ws = Path(cfg["workspace"])
    ws.mkdir(parents=True, exist_ok=True)
    (ws/"cfg.json").write_text(json.dumps(cfg, indent=2))

    if cfg["data"]["name"] == "animerun":
        tr_ds = Animerun(root=cfg["data"]["train_root"],
                                    stride_min=cfg["data"]["stride_min"],
                                    stride_max=cfg["data"]["stride_max"],
                                    crop_size=tuple(cfg["data"]["crop_size"]),
                                    color_jitter=tuple(cfg["data"]["color_jitter"]),
                                    do_flip=cfg["data"]["do_flip"],
                                    grayscale_p=cfg["data"]["grayscale_p"],
                                    img_exts=cfg["data"].get("img_exts"),
                                    is_test=False)


        va_ds = Animerun(root=cfg["data"]["val_root"], stride_min=1, stride_max=1,
                                    crop_size=tuple(cfg["data"]["crop_size"]),
                                    color_jitter=None, do_flip=False, grayscale_p=0.0,
                                    img_exts=cfg["data"].get("img_exts"),
                                    is_test=True)
    

    tr_loader = DataLoader(tr_ds, batch_size=cfg["data"]["batch_size"], shuffle=True,
                            num_workers=cfg["data"]["num_workers"], pin_memory=True, drop_last=True)
    
    va_loader = DataLoader(va_ds, batch_size=max(1, cfg["data"]["batch_size"]//2), shuffle=False,
                            num_workers=max(2, cfg["data"]["num_workers"]//2), pin_memory=True)


    trainer = UnsupervisedFlowTrainer(args, cfg, ws)
    trainer.fit(tr_loader, va_loader)


def clip_main(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)


    set_seed(cfg.get("seed", 1337))
    cudnn.benchmark = True


    ws = Path(cfg["workspace"])
    ws.mkdir(parents=True, exist_ok=True)
    (ws/"cfg.json").write_text(json.dumps(cfg, indent=2))
    if cfg["data"]["name"] == "animerun_clip":
        train_ds = UnlabeledClipDataset(
            root=cfg["data"]["train_root"],
            T=cfg["data"]["T"],
            stride_min=cfg["data"]["stride_min"],
            stride_max=cfg["data"]["stride_max"],
            crop_size=tuple(cfg["data"]["crop_size"]),
            color_jitter=tuple(cfg["data"]["color_jitter"]),
            do_flip=cfg["data"]["do_flip"],
            grayscale_p=cfg["data"]["grayscale_p"],
            resize=cfg["data"].get("resize", True),
            keep_aspect=cfg["data"].get("keep_aspect", False),
            pad_mode=cfg["data"].get("pad_mode", "reflect"),
            is_test=False
        )
        val_ds = UnlabeledClipDataset(
            root=cfg["data"]["val_root"],
            T=cfg["data"]["T"],    
            stride_min=1,
            stride_max=1,
            crop_size=tuple(cfg["data"]["crop_size"]),
            color_jitter=None,
            do_flip=False,
            grayscale_p=0.0,
            resize=cfg["data"].get("resize", True),
            keep_aspect=cfg["data"].get("keep_aspect", False),
            pad_mode=cfg["data"].get("pad_mode", "reflect"),
            is_test=True
        )
    elif cfg["data"]["name"] == "linkto_anime_clip":
        train_ds = LinkToAnimeClipDataset(
            root=cfg["data"]["train_root"],
            split = 'train',
            T=cfg["data"]["T"],
            stride_min=cfg["data"]["stride_min"],
            stride_max=cfg["data"]["stride_max"],
            crop_size=tuple(cfg["data"]["crop_size"]),
            color_jitter=tuple(cfg["data"]["color_jitter"]),
            do_flip=cfg["data"]["do_flip"],
            grayscale_p=cfg["data"]["grayscale_p"],
            resize=cfg["data"].get("resize", True),
            keep_aspect=cfg["data"].get("keep_aspect", False),
            pad_mode=cfg["data"].get("pad_mode", "reflect"),
        )
        val_ds = LinkToAnimeClipDataset(
            root=cfg["data"]["val_root"],
            T=cfg["data"]["T"],  
            split='val',  
            stride_min=1,
            stride_max=1,
            crop_size=tuple(cfg["data"]["crop_size"]),
            color_jitter=None,
            do_flip=False,
            grayscale_p=0.0,
            resize=cfg["data"].get("resize", True),
            keep_aspect=cfg["data"].get("keep_aspect", False),
            pad_mode=cfg["data"].get("pad_mode", "reflect"),
        )
    tr_loader = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True,
                            num_workers=cfg["data"]["num_workers"], pin_memory=True, drop_last=True)

    va_loader = DataLoader(val_ds, batch_size=max(1, cfg["data"]["batch_size"]//2), shuffle=False,
                            num_workers=max(2, cfg["data"]["num_workers"]//2), pin_memory=True)

    trainer = UnsupervisedClipTrainer(args=None, cfg=cfg, workspace=args.workspace)
    trainer.fit(tr_loader, va_loader)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/unsup_clip_animotionflow.yaml")
    ap.add_argument("--clip", action="store_true", help="Use clip trainer")
    ap.add_argument("--workspace", default="workspace")
    args = ap.parse_args()
    if args.clip:
        clip_main(args)
    else:
        main(args)