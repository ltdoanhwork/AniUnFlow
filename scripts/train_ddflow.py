import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from engine.trainer_ddflow import DDFlowTrainer
from dataio import UnlabeledClipDataset
from torch.utils.data import DataLoader
import yaml
import argparse
import torch

def build_dataset(cfg, is_test=False):
    return UnlabeledClipDataset(
        root=cfg["root"],
        T=cfg.get("clip_len", 2),
        stride_min=1,
        stride_max=1,
        crop_size=tuple(cfg.get("crop_size", [384, 768])),
        color_jitter=cfg.get("color_jitter", None),
        do_flip=cfg.get("do_flip", False),
        grayscale_p=cfg.get("grayscale_p", 0.0),
        resize=cfg.get("resize", True),
        keep_aspect=cfg.get("keep_aspect", True),
        pad_mode=cfg.get("pad_mode", "reflect"),
        is_test=is_test,
        load_sam_masks=False,
        sam_mask_root=None,
    )

def build_loader(ds, cfg):
    return DataLoader(
        ds,
        batch_size=cfg.get("batch_size", 4),
        shuffle=cfg.get("shuffle", True),
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=cfg.get("drop_last", True),
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
        
    trainer = DDFlowTrainer(cfg, workspace=Path(cfg["workspace"]))
    
    if args.dry_run:
        print("Dry run: Checking dataset...")
        t_ds = build_dataset(cfg["data"]["train"], is_test=False)
        v_ds = build_dataset(cfg["data"]["val"], is_test=True)
        print(f"[Train] Found {len(t_ds)} sequences")
        print(f"[Test] Found {len(v_ds)} scenes")
        
        t_loader = build_loader(t_ds, cfg["data"]["train"])
        
        batch = next(iter(t_loader))
        clip = batch["clip"]
        print(f"Batch shape: {clip.shape}")
        
        # Test Model
        # Input: T=3 usually, but DDFlow uses pairs.
        # We need to manually slice.
        img1 = clip[:, 0].to(trainer.device)
        img2 = clip[:, 1].to(trainer.device)
        if img1.dtype == torch.uint8:
            img1 = img1.float() / 255.0
            img2 = img2.float() / 255.0
        imgs = torch.cat([img1, img2], dim=1)
        out = trainer.model(imgs)
        print("Model forward success")
        return

    # Train
    train_ds = build_dataset(cfg["data"]["train"], is_test=False)
    val_ds = build_dataset(cfg["data"]["val"], is_test=True)
    
    train_loader = build_loader(train_ds, cfg["data"]["train"])
    val_loader = build_loader(val_ds, cfg["data"]["val"])
    
    trainer.fit(train_loader, val_loader)

if __name__ == "__main__":
    main()
