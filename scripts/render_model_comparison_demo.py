#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataio.clip_dataset_unsup import UnlabeledClipDataset
from models.aniunflow_v5 import AniFlowFormerTV5, V5Config
from utils.flow_viz import compute_flow_magnitude_radmax, flow_to_image


UPFLOW_MEAN = torch.tensor([104.920005, 110.1753, 114.785955], dtype=torch.float32).view(1, 3, 1, 1)
UPFLOW_SCALE = 0.0039216


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render qualitative comparisons between Ours and weaker baselines.")
    parser.add_argument("--scene", default="cami_02_05_A")
    parser.add_argument("--centers", nargs="*", type=int, default=[31, 6])
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "demo" / "v5_1_object_memory_dense_parallel_v4_compare"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def patch_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if "sam" in cfg:
        cfg["sam"]["encoder_config"] = "configs/sam2.1/sam2.1_hiera_t.yaml"
    return cfg


def load_v5_model(config_path: Path, checkpoint_path: Path, device: torch.device) -> AniFlowFormerTV5:
    cfg = patch_cfg(load_yaml(config_path))
    model = AniFlowFormerTV5(V5Config.from_dict(cfg)).to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    current = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in current and getattr(current[k], "shape", None) == getattr(v, "shape", None)}
    model.load_state_dict(filtered, strict=False)
    return model


def load_upflow_model(device: torch.device):
    upflow_root = ROOT / "models" / "UPFlow_pytorch"
    sys.modules.pop("utils", None)
    sys.modules.pop("utils.tools", None)
    if str(upflow_root) not in sys.path:
        sys.path.insert(0, str(upflow_root))
    from model.upflow import UPFlow_net

    net_conf = UPFlow_net.config()
    net_conf.update({"if_use_cor_pytorch": True})
    model = net_conf().to(device).eval()
    ckpt = torch.load(ROOT / "workspaces" / "upflow_animerun_full" / "best_upflow_animerun.pth", map_location=device)
    state = ckpt
    if isinstance(ckpt, dict):
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    return model


def normalize_upflow(images: torch.Tensor) -> torch.Tensor:
    mean = UPFLOW_MEAN.to(images.device, images.dtype)
    return (images * 255.0 - mean) * UPFLOW_SCALE


def resize_flow_tensor(flow: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    if tuple(flow.shape[-2:]) == tuple(size):
        return flow
    h0, w0 = flow.shape[-2:]
    h1, w1 = size
    out = F.interpolate(flow.unsqueeze(0), size=size, mode="bilinear", align_corners=True).squeeze(0)
    out[0] *= float(w1) / max(float(w0), 1.0)
    out[1] *= float(h1) / max(float(h0), 1.0)
    return out


def tensor_to_uint8_rgb(image: torch.Tensor) -> np.ndarray:
    return np.uint8(np.round(image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0))


def flow_to_rgb(flow: torch.Tensor, rad_max: float) -> np.ndarray:
    return flow_to_image(flow.detach().cpu().permute(1, 2, 0).numpy().astype(np.float32), rad_max=rad_max)


def epe_to_rgb(pred: torch.Tensor, gt: torch.Tensor, hi: float) -> np.ndarray:
    epe = torch.norm(pred - gt, dim=0).detach().cpu().numpy()
    if hi <= 0:
        hi = 1.0
    norm = np.clip(epe / hi, 0.0, 1.0)
    heat = np.uint8(np.round(norm * 255.0))
    return cv2.cvtColor(cv2.applyColorMap(heat, getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)), cv2.COLOR_BGR2RGB)


def draw_tile(image_np: np.ndarray, label: str) -> Image.Image:
    image = Image.fromarray(image_np).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle((18, 16, 220, 54), radius=10, fill=(17, 24, 39, 210))
    draw.text((30, 24), label, fill=(255, 255, 255, 255), font=ImageFont.load_default())
    return image


def build_panel(tiles: Sequence[Tuple[str, np.ndarray]], title: str, subtitle: str, output_path: Path, cols: int = 3) -> None:
    images = [draw_tile(image_np, label) for label, image_np in tiles]
    tile_w = max(image.width for image in images)
    tile_h = max(image.height for image in images)
    rows = int(math.ceil(len(images) / float(cols)))
    pad = 22
    gap = 18
    title_h = 86
    canvas = Image.new(
        "RGB",
        (pad * 2 + cols * tile_w + gap * (cols - 1), title_h + pad * 2 + rows * tile_h + gap * (rows - 1)),
        (248, 246, 242),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((pad, 18), title, fill=(16, 22, 32), font=font)
    draw.text((pad, 42), subtitle, fill=(86, 92, 104), font=font)

    for idx, image in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = pad + col * (tile_w + gap)
        y = title_h + pad + row * (tile_h + gap)
        canvas.paste(image, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def build_contact_sheet(image_paths: Sequence[Path], output_path: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    if not images:
        return
    pad = 20
    gap = 20
    width = max(image.width for image in images) + pad * 2
    height = sum(image.height for image in images) + gap * (len(images) - 1) + pad * 2
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    y = pad
    for image in images:
        x = (width - image.width) // 2
        canvas.paste(image, (x, y))
        y += image.height + gap
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def scene_name(seq: Dict[str, Any]) -> str:
    first = seq["frames"][0]
    return first.parents[1].name if first.parent.name == "original" else first.parent.name


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = UnlabeledClipDataset(
        root="data/AnimeRun_v2",
        T=5,
        crop_size=(256, 512),
        resize=True,
        keep_aspect=False,
        load_sam_masks=True,
        sam_mask_root="data/AnimeRun_v2/SAM_Masks_v2",
        sam_mask_cache_size=64,
        is_test=True,
    )
    scene_id = None
    seq = None
    for sid, item in enumerate(dataset.test_seqs):
        if scene_name(item) == args.scene:
            scene_id = sid
            seq = item
            break
    if scene_id is None or seq is None:
        raise ValueError(f"Scene '{args.scene}' not found.")
    index_map = {tuple(item): idx for idx, item in enumerate(dataset.index)}

    model_ours = load_v5_model(
        ROOT / "workspaces" / "v5_1_object_memory_dense_parallel_v4" / "config.yaml",
        ROOT / "workspaces" / "v5_1_object_memory_dense_parallel_v4" / "best.pth",
        device,
    )
    model_object = load_v5_model(
        ROOT / "workspaces" / "v5_object_memory_sam_parallel" / "config.yaml",
        ROOT / "workspaces" / "v5_object_memory_sam_parallel" / "best.pth",
        device,
    )
    model_upflow = load_upflow_model(device)

    records: List[Dict[str, Any]] = []
    panel_paths: List[Path] = []

    for center in args.centers:
        key = (scene_id, center, 1)
        if key not in index_map:
            continue
        sample = dataset[index_map[key]]
        frame_name = seq["frames"][center].stem
        clip = sample["clip"].unsqueeze(0).to(device)
        sam_masks = sample["sam_masks"].unsqueeze(0).to(device)
        gt = sample["flow_list"][2].to(device)
        img1 = clip[:, 2]
        img2 = clip[:, 3]

        with torch.inference_mode():
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                pred_ours = resize_flow_tensor(model_ours(clip, sam_masks=sam_masks)["flows_fw"][2][0], tuple(gt.shape[-2:]))
                pred_object = resize_flow_tensor(model_object(clip, sam_masks=sam_masks)["flows_fw"][2][0], tuple(gt.shape[-2:]))
                start = torch.zeros((1, 2, 1, 1), device=device, dtype=img1.dtype)
                up_out = model_upflow(
                    {
                        "im1": normalize_upflow(img1),
                        "im2": normalize_upflow(img2),
                        "im1_raw": normalize_upflow(img1),
                        "im2_raw": normalize_upflow(img2),
                        "im1_sp": normalize_upflow(img1),
                        "im2_sp": normalize_upflow(img2),
                        "start": start,
                        "if_loss": False,
                    }
                )
                pred_upflow = resize_flow_tensor(up_out["flow_f_out"][0], tuple(gt.shape[-2:]))

        gt_flow = gt.detach().cpu()
        pred_ours_cpu = pred_ours.detach().cpu()
        pred_object_cpu = pred_object.detach().cpu()
        pred_upflow_cpu = pred_upflow.detach().cpu()
        rad_max = compute_flow_magnitude_radmax(
            [
                gt_flow.permute(1, 2, 0).numpy(),
                pred_ours_cpu.permute(1, 2, 0).numpy(),
                pred_object_cpu.permute(1, 2, 0).numpy(),
                pred_upflow_cpu.permute(1, 2, 0).numpy(),
            ],
            robust_percentile=95,
        )

        epe_ours = float(torch.norm(pred_ours_cpu - gt_flow, dim=0).mean().item())
        epe_object = float(torch.norm(pred_object_cpu - gt_flow, dim=0).mean().item())
        epe_upflow = float(torch.norm(pred_upflow_cpu - gt_flow, dim=0).mean().item())
        epe_stack = torch.stack(
            [
                torch.norm(pred_object_cpu - gt_flow, dim=0),
                torch.norm(pred_upflow_cpu - gt_flow, dim=0),
                torch.norm(pred_ours_cpu - gt_flow, dim=0),
            ],
            dim=0,
        ).numpy()
        shared_epe_hi = float(np.percentile(epe_stack, 95))

        tiles = [
            ("Frame t", tensor_to_uint8_rgb(img1[0].cpu())),
            ("Frame t+1", tensor_to_uint8_rgb(img2[0].cpu())),
            ("GT flow", flow_to_rgb(gt_flow, rad_max)),
            (f"Object baseline ({epe_object:.2f})", flow_to_rgb(pred_object_cpu, rad_max)),
            (f"UPFlow ({epe_upflow:.2f})", flow_to_rgb(pred_upflow_cpu, rad_max)),
            (f"Ours ({epe_ours:.2f})", flow_to_rgb(pred_ours_cpu, rad_max)),
            ("Object EPE", epe_to_rgb(pred_object_cpu, gt_flow, shared_epe_hi)),
            ("UPFlow EPE", epe_to_rgb(pred_upflow_cpu, gt_flow, shared_epe_hi)),
            ("Ours EPE", epe_to_rgb(pred_ours_cpu, gt_flow, shared_epe_hi)),
        ]

        title = f"{args.scene} | frame={frame_name} | center={center}"
        subtitle = (
            f"Ours={epe_ours:.2f} px | Object baseline={epe_object:.2f} px | "
            f"UPFlow={epe_upflow:.2f} px"
        )
        panel_path = output_dir / "panels" / f"{args.scene}_compare_{center:04d}_{frame_name}.png"
        build_panel(tiles, title, subtitle, panel_path)
        panel_paths.append(panel_path)
        records.append(
            {
                "scene": args.scene,
                "center": center,
                "frame": frame_name,
                "panel_path": str(panel_path.relative_to(ROOT)),
                "ours_epe": epe_ours,
                "object_epe": epe_object,
                "upflow_epe": epe_upflow,
            }
        )
        print(f"[Compare] Saved {panel_path.relative_to(ROOT)}")

    if panel_paths:
        summary_path = output_dir / f"{args.scene}_comparison_sheet.png"
        build_contact_sheet(panel_paths, summary_path)
        print(f"[Compare] Saved {summary_path.relative_to(ROOT)}")

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps({"scene": args.scene, "records": records}, indent=2))
    print(f"[Compare] Saved {metadata_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
