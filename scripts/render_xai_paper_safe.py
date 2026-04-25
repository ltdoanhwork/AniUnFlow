#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataio.clip_dataset_unsup import UnlabeledClipDataset
from models.aniunflow_v5 import AniFlowFormerTV5, V5Config


DEFAULT_SAM_CONFIG_FALLBACKS = [
    ROOT / "models" / "sam2" / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_t.yaml",
    ROOT / "models" / "sam2" / "sam2" / "sam2_hiera_t.yaml",
]

PALETTE = np.array(
    [
        [0, 0, 0],
        [233, 98, 72],
        [244, 194, 76],
        [101, 184, 130],
        [83, 164, 229],
        [134, 112, 229],
        [220, 120, 193],
        [72, 198, 202],
        [247, 143, 95],
        [160, 201, 104],
        [209, 122, 87],
        [120, 137, 210],
        [196, 168, 90],
        [99, 151, 122],
        [226, 133, 144],
        [142, 207, 201],
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render paper-safe XAI figures for AniUnFlow.")
    parser.add_argument("--config", default=str(ROOT / "workspaces" / "v5_4_sam_propagation_memory" / "config.yaml"))
    parser.add_argument("--checkpoint", default=str(ROOT / "workspaces" / "v5_4_sam_propagation_memory" / "best.pth"))
    parser.add_argument("--output-dir", default=str(ROOT / "reports" / "img" / "aniunflow_xai_safe"))
    parser.add_argument("--scene", default="cami_02_05_A")
    parser.add_argument("--centers", nargs="*", type=int, default=[31, 43])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fp32", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_repo_path(path_str: str) -> Optional[Path]:
    if not path_str:
        return None
    raw = Path(path_str)
    if raw.is_absolute() and raw.exists():
        return raw
    repo_path = (ROOT / raw).resolve()
    if repo_path.exists():
        return repo_path
    return None


def patch_sam_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    sam_cfg = cfg.setdefault("sam", {})
    current_raw = str(sam_cfg.get("encoder_config", ""))
    if "configs/" in current_raw:
        sam_cfg["encoder_config"] = current_raw[current_raw.index("configs/") :]
        return cfg
    current = resolve_repo_path(current_raw)
    if current is not None:
        current_str = current.as_posix()
        if "configs/" in current_str:
            sam_cfg["encoder_config"] = current_str[current_str.index("configs/") :]
            return cfg
    for candidate in DEFAULT_SAM_CONFIG_FALLBACKS:
        if candidate.exists():
            candidate_str = candidate.as_posix()
            sam_cfg["encoder_config"] = candidate_str[candidate_str.index("configs/") :]
            return cfg
    return cfg


def build_dataset(cfg: Dict[str, Any], data_root: Optional[str]) -> UnlabeledClipDataset:
    data_cfg = cfg["data"]
    root = data_root or data_cfg.get("val_root") or data_cfg["train_root"]
    load_sam_masks = bool(data_cfg.get("val_load_sam_masks", data_cfg.get("load_sam_masks", False)))
    sam_mask_root = data_cfg.get("val_sam_mask_dir") or data_cfg.get("sam_mask_dir")
    return UnlabeledClipDataset(
        root=root,
        T=int(data_cfg.get("T", 5)),
        crop_size=tuple(data_cfg.get("crop_size", [256, 512])),
        resize=bool(data_cfg.get("resize", True)),
        keep_aspect=bool(data_cfg.get("keep_aspect", False)),
        load_sam_masks=load_sam_masks,
        sam_mask_root=sam_mask_root,
        sam_mask_cache_size=int(data_cfg.get("val_sam_mask_cache_size", data_cfg.get("sam_mask_cache_size", 0))),
        is_test=True,
    )


def apply_checkpoint_state(module: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    current_state = module.state_dict()
    filtered_state = {}
    for key, value in state_dict.items():
        if key not in current_state:
            continue
        target_value = current_state[key]
        if hasattr(target_value, "shape") and hasattr(value, "shape") and target_value.shape != value.shape:
            continue
        filtered_state[key] = value
    module.load_state_dict(filtered_state, strict=False)


def build_model(cfg: Dict[str, Any], checkpoint_path: Path, device: torch.device) -> AniFlowFormerTV5:
    model = AniFlowFormerTV5(V5Config.from_dict(cfg)).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    apply_checkpoint_state(model, state_dict)
    model.eval()
    return model


def scene_name_from_seq(seq: Dict[str, Any]) -> str:
    first_frame = seq["frames"][0]
    return first_frame.parents[1].name if first_frame.parent.name == "original" else first_frame.parent.name


def build_index_map(dataset: UnlabeledClipDataset) -> Dict[Tuple[int, int, int], int]:
    return {tuple(item): idx for idx, item in enumerate(dataset.index)}


def tensor_to_uint8_rgb(image: torch.Tensor) -> np.ndarray:
    image_np = image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return np.uint8(np.round(image_np * 255.0))


def float_map_to_rgb(value_map: np.ndarray, cmap: int, lo: Optional[float] = None, hi: Optional[float] = None) -> np.ndarray:
    value_map = np.asarray(value_map, dtype=np.float32)
    if lo is None:
        lo = float(np.min(value_map))
    if hi is None:
        hi = float(np.max(value_map))
    if not math.isfinite(lo):
        lo = 0.0
    if not math.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    norm = np.clip((value_map - lo) / (hi - lo), 0.0, 1.0)
    heat = np.uint8(np.round(norm * 255.0))
    return cv2.cvtColor(cv2.applyColorMap(heat, cmap), cv2.COLOR_BGR2RGB)


def resize_flow_tensor(flow: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    if tuple(flow.shape[-2:]) == tuple(size):
        return flow
    h0, w0 = flow.shape[-2:]
    h1, w1 = size
    resized = torch.nn.functional.interpolate(flow.unsqueeze(0), size=size, mode="bilinear", align_corners=True).squeeze(0)
    resized[0] *= float(w1) / max(float(w0), 1.0)
    resized[1] *= float(h1) / max(float(h0), 1.0)
    return resized


def resize_map_tensor(value: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    if tuple(value.shape[-2:]) == tuple(size):
        return value
    if value.ndim == 2:
        value = value.unsqueeze(0).unsqueeze(0)
        return torch.nn.functional.interpolate(value, size=size, mode="bilinear", align_corners=True).squeeze(0).squeeze(0)
    if value.ndim == 3:
        return torch.nn.functional.interpolate(value.unsqueeze(0), size=size, mode="bilinear", align_corners=True).squeeze(0)
    raise ValueError(f"Unsupported map shape for resize: {tuple(value.shape)}")


def label_mask_to_rgb(mask: torch.Tensor) -> np.ndarray:
    mask_np = mask.detach().cpu().numpy().astype(np.int64)
    rgb = np.zeros(mask_np.shape + (3,), dtype=np.uint8)
    positive = mask_np > 0
    if positive.any():
        rgb[positive] = PALETTE[1 + (mask_np[positive] - 1) % (len(PALETTE) - 1)]
    return rgb


def label_boundaries(mask: torch.Tensor) -> np.ndarray:
    labels = mask.detach().cpu().numpy().astype(np.int32)
    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[1:, :] |= labels[1:, :] != labels[:-1, :]
    boundary[:-1, :] |= labels[1:, :] != labels[:-1, :]
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[:, :-1] |= labels[:, 1:] != labels[:, :-1]
    return boundary


def boundary_overlay(frame_rgb: np.ndarray, boundary: np.ndarray, edge_rgb: Tuple[int, int, int] = (255, 208, 72)) -> np.ndarray:
    overlay = frame_rgb.copy()
    overlay[boundary] = edge_rgb
    return overlay


def draw_labeled_tile(image_np: np.ndarray, label: str, tile_size: Tuple[int, int]) -> Image.Image:
    image = Image.fromarray(image_np).convert("RGB").resize(tile_size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (tile_size[0], tile_size[1] + 38), (250, 248, 244))
    canvas.paste(image, (0, 38))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rounded_rectangle((10, 8, min(tile_size[0] - 10, 180), 30), radius=8, fill=(17, 24, 39, 210))
    draw.text((18, 14), label, fill=(255, 255, 255, 255), font=ImageFont.load_default())
    return canvas


def build_panel(
    tiles: Sequence[Tuple[str, np.ndarray]],
    title: str,
    subtitle: str,
    output_path: Path,
    cols: int,
    tile_size: Tuple[int, int],
    gap: int = 14,
    pad: int = 20,
) -> None:
    images = [draw_labeled_tile(image_np, label, tile_size) for label, image_np in tiles]
    tile_w = max(image.width for image in images)
    tile_h = max(image.height for image in images)
    rows = int(math.ceil(len(images) / float(cols)))
    title_h = 78
    width = pad * 2 + cols * tile_w + gap * (cols - 1)
    height = title_h + pad * 2 + rows * tile_h + gap * (rows - 1)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((pad, 16), title, fill=(16, 22, 32), font=font)
    draw.text((pad, 38), subtitle, fill=(88, 94, 104), font=font)
    for idx, image in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = pad + col * (tile_w + gap)
        y = title_h + pad + row * (tile_h + gap)
        canvas.paste(image, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def build_contact_sheet(image_paths: Sequence[Path], output_path: Path, gap: int = 20, pad: int = 20) -> None:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    if not images:
        return
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


def render_case_panels(
    model: AniFlowFormerTV5,
    sample: Dict[str, Any],
    scene_name: str,
    center: int,
    output_dir: Path,
    device: torch.device,
    use_autocast: bool,
) -> Dict[str, Any]:
    clip = sample["clip"].unsqueeze(0).to(device)
    sam_masks = sample["sam_masks"].unsqueeze(0).to(device)
    flow_list = [flow.unsqueeze(0).to(device) for flow in sample["flow_list"]]
    pair_idx = min(clip.shape[1] // 2, clip.shape[1] - 2)
    with torch.inference_mode():
        with torch.amp.autocast(device_type=device.type, enabled=use_autocast):
            output = model(clip, sam_masks=sam_masks)

    frame_a = tensor_to_uint8_rgb(clip[0, pair_idx].cpu())
    frame_b = tensor_to_uint8_rgb(clip[0, pair_idx + 1].cpu())
    labels = sam_masks[0, pair_idx, 0].cpu()
    mask_rgb = label_mask_to_rgb(labels)
    boundary = label_boundaries(labels)
    boundary_rgb = boundary_overlay(frame_a, boundary)

    gt_flow = flow_list[pair_idx][0].cpu()
    pred_flow = resize_flow_tensor(output["flows_fw"][pair_idx][0].detach().cpu(), tuple(gt_flow.shape[-2:]))
    dense_flow = resize_flow_tensor(output["dense_prior_flow_fw"][pair_idx][0].detach().cpu(), tuple(gt_flow.shape[-2:]))
    sam_memory = resize_map_tensor(output["sam_memory_agreement_fw"][pair_idx][0].detach().cpu(), tuple(gt_flow.shape[-2:]))
    residual_mag = torch.norm(pred_flow - dense_flow, dim=0).numpy()
    epe_map = torch.norm(pred_flow - gt_flow, dim=0).numpy()
    disagreement_map = 1.0 - np.clip(sam_memory[0].numpy(), 0.0, 1.0)

    structure_tiles = [
        ("Frame t", frame_a),
        ("Frame t+1", frame_b),
        ("SAM labels", mask_rgb),
        ("Boundary overlay", boundary_rgb),
        ("SAM disagreement", float_map_to_rgb(disagreement_map, getattr(cv2, "COLORMAP_INFERNO", cv2.COLORMAP_JET), 0.0, 1.0)),
    ]
    diagnostic_tiles = [
        ("Frame t", frame_a),
        ("Boundary overlay", boundary_rgb),
        ("Residual magnitude", float_map_to_rgb(residual_mag, getattr(cv2, "COLORMAP_MAGMA", cv2.COLORMAP_JET), 0.0, float(np.percentile(residual_mag, 95)))),
        ("EPE heatmap", float_map_to_rgb(epe_map, getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET), 0.0, float(np.percentile(epe_map, 95)))),
    ]

    frame_tag = sample["frame_name"]
    structure_path = output_dir / "panels" / f"{scene_name}_{frame_tag}_xai_structure.png"
    diagnostic_path = output_dir / "panels" / f"{scene_name}_{frame_tag}_xai_diagnostics.png"
    build_panel(
        structure_tiles,
        title=f"{scene_name} | frame {frame_tag} | structure-oriented XAI",
        subtitle="Only the most stable explanatory cues are kept for paper presentation.",
        output_path=structure_path,
        cols=5,
        tile_size=(220, 124),
    )
    build_panel(
        diagnostic_tiles,
        title=f"{scene_name} | frame {frame_tag} | error and repair signals",
        subtitle="Residual activity and EPE are shown together with the original frame and boundary map.",
        output_path=diagnostic_path,
        cols=4,
        tile_size=(250, 142),
    )
    return {
        "frame": frame_tag,
        "center": center,
        "structure_panel": str(structure_path.relative_to(ROOT)),
        "diagnostic_panel": str(diagnostic_path.relative_to(ROOT)),
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = patch_sam_config(load_yaml(config_path))
    device = torch.device(args.device)
    use_autocast = (device.type == "cuda") and (not args.fp32)
    dataset = build_dataset(cfg, args.data_root)
    index_map = build_index_map(dataset)
    scene_id = None
    seq = None
    for sid, item in enumerate(dataset.test_seqs):
        if scene_name_from_seq(item) == args.scene:
            scene_id = sid
            seq = item
            break
    if scene_id is None or seq is None:
        raise ValueError(f"Scene '{args.scene}' not found.")
    model = build_model(cfg, checkpoint_path, device)

    structure_paths: List[Path] = []
    diagnostic_paths: List[Path] = []
    records: List[Dict[str, Any]] = []
    for center in args.centers:
        key = (scene_id, center, 1)
        if key not in index_map:
            continue
        sample = dataset[index_map[key]]
        sample["frame_name"] = seq["frames"][center].stem
        record = render_case_panels(model, sample, args.scene, center, output_dir, device, use_autocast)
        records.append(record)
        structure_paths.append(ROOT / record["structure_panel"])
        diagnostic_paths.append(ROOT / record["diagnostic_panel"])
        print(f"[XAI Safe] Saved {record['structure_panel']}")
        print(f"[XAI Safe] Saved {record['diagnostic_panel']}")

    structure_sheet = output_dir / f"{args.scene}_xai_structure_sheet.png"
    diagnostic_sheet = output_dir / f"{args.scene}_xai_diagnostics_sheet.png"
    build_contact_sheet(structure_paths, structure_sheet)
    build_contact_sheet(diagnostic_paths, diagnostic_sheet)
    print(f"[XAI Safe] Saved {structure_sheet.relative_to(ROOT)}")
    print(f"[XAI Safe] Saved {diagnostic_sheet.relative_to(ROOT)}")
    metadata = {
        "scene": args.scene,
        "centers": args.centers,
        "records": records,
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[XAI Safe] Saved {metadata_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
