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

from dataio.clip_dataset_unsup import UnlabeledClipDataset, _read_flow_any
from models.aniunflow_v5 import AniFlowFormerTV5, V5Config
from utils.flow_viz import compute_flow_magnitude_radmax, flow_to_image


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
    parser = argparse.ArgumentParser(description="Render V5.1 object-memory dense paper demo panels.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "workspaces" / "v5_1_object_memory_dense_parallel_v4" / "config.yaml"),
        help="Path to workspace config YAML.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "workspaces" / "v5_1_object_memory_dense_parallel_v4" / "best.pth"),
        help="Path to trained checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "demo" / "v5_1_object_memory_dense_parallel_v4_paper"),
        help="Directory to store rendered demo artifacts.",
    )
    parser.add_argument(
        "--scene",
        default=None,
        help="Scene name under data/AnimeRun_v2/test/Frame_Anime. Defaults to the highest-motion scene.",
    )
    parser.add_argument(
        "--centers",
        nargs="*",
        type=int,
        default=None,
        help="Optional center frame indices to render. Overrides automatic selection.",
    )
    parser.add_argument("--topk", type=int, default=3, help="Number of panels to render when auto-selecting.")
    parser.add_argument(
        "--min-gap",
        type=int,
        default=8,
        help="Minimum gap between selected centers when auto-selecting.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Optional override for AnimeRun_v2 root.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Disable autocast and run full precision inference.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as handle:
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
    missing_keys, unexpected_keys = module.load_state_dict(filtered_state, strict=False)
    if missing_keys:
        preview = ", ".join(missing_keys[:5])
        print(f"[Demo] Missing model keys after load: {preview}")
    if unexpected_keys:
        preview = ", ".join(unexpected_keys[:5])
        print(f"[Demo] Unexpected model keys after load: {preview}")


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


def frame_name_from_center(seq: Dict[str, Any], center: int) -> str:
    return seq["frames"][center].stem


def mean_motion_for_flow(flow_path: Path) -> float:
    flow = _read_flow_any(flow_path)
    mag = np.linalg.norm(flow, axis=2)
    return float(np.mean(mag))


def percentile_motion_for_scene(seq: Dict[str, Any], percentile: float = 90.0) -> float:
    mags = [mean_motion_for_flow(path) for path in seq["flow_fw"]]
    if not mags:
        return 0.0
    return float(np.percentile(np.asarray(mags, dtype=np.float32), percentile))


def choose_scene(dataset: UnlabeledClipDataset, requested_scene: Optional[str]) -> Tuple[int, Dict[str, Any]]:
    if requested_scene:
        for sid, seq in enumerate(dataset.test_seqs):
            if scene_name_from_seq(seq) == requested_scene:
                return sid, seq
        available = ", ".join(scene_name_from_seq(seq) for seq in dataset.test_seqs[:12])
        raise ValueError(f"Scene '{requested_scene}' not found. Examples: {available}")

    scored = []
    for sid, seq in enumerate(dataset.test_seqs):
        scored.append((percentile_motion_for_scene(seq), sid, seq))
    scored.sort(reverse=True, key=lambda item: item[0])
    if not scored:
        raise RuntimeError("No test scenes found in AnimeRun_v2.")
    score, sid, seq = scored[0]
    print(f"[Demo] Auto-selected scene '{scene_name_from_seq(seq)}' with p90 motion {score:.2f}.")
    return sid, seq


def choose_centers(
    seq: Dict[str, Any],
    temporal_radius: int,
    topk: int,
    min_gap: int,
    requested_centers: Optional[Sequence[int]],
) -> List[int]:
    n_frames = len(seq["frames"])
    valid_centers = list(range(temporal_radius, n_frames - temporal_radius))
    if requested_centers:
        centers = [c for c in requested_centers if c in valid_centers]
        if not centers:
            raise ValueError(f"Requested centers {requested_centers} are invalid for scene with {n_frames} frames.")
        return centers

    scored = []
    for center in valid_centers:
        motion = mean_motion_for_flow(seq["flow_fw"][center])
        scored.append((motion, center))
    scored.sort(reverse=True)

    selected: List[int] = []
    for _, center in scored:
        if all(abs(center - existing) >= min_gap for existing in selected):
            selected.append(center)
        if len(selected) >= topk:
            break
    return sorted(selected)


def build_index_map(dataset: UnlabeledClipDataset) -> Dict[Tuple[int, int, int], int]:
    return {tuple(item): idx for idx, item in enumerate(dataset.index)}


def tensor_to_uint8_rgb(image: torch.Tensor) -> np.ndarray:
    image_np = image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return np.uint8(np.round(image_np * 255.0))


def label_mask_to_rgb(mask: torch.Tensor) -> np.ndarray:
    mask_np = mask.detach().cpu().numpy().astype(np.int64)
    rgb = np.zeros(mask_np.shape + (3,), dtype=np.uint8)
    positive = mask_np > 0
    if positive.any():
        rgb[positive] = PALETTE[1 + (mask_np[positive] - 1) % (len(PALETTE) - 1)]
    return rgb


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
    rgb = cv2.cvtColor(cv2.applyColorMap(heat, cmap), cv2.COLOR_BGR2RGB)
    return rgb


def flow_to_rgb(flow: torch.Tensor, rad_max: float) -> np.ndarray:
    flow_np = flow.detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
    return flow_to_image(flow_np, rad_max=rad_max)


def resize_flow_tensor(flow: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    if tuple(flow.shape[-2:]) == tuple(size):
        return flow
    h0, w0 = flow.shape[-2:]
    h1, w1 = size
    resized = torch.nn.functional.interpolate(
        flow.unsqueeze(0),
        size=size,
        mode="bilinear",
        align_corners=True,
    ).squeeze(0)
    resized[0] *= float(w1) / max(float(w0), 1.0)
    resized[1] *= float(h1) / max(float(h0), 1.0)
    return resized


def resize_map_tensor(value: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    if tuple(value.shape[-2:]) == tuple(size):
        return value
    if value.ndim == 2:
        value = value.unsqueeze(0).unsqueeze(0)
        return torch.nn.functional.interpolate(
            value,
            size=size,
            mode="bilinear",
            align_corners=True,
        ).squeeze(0).squeeze(0)
    if value.ndim == 3:
        return torch.nn.functional.interpolate(
            value.unsqueeze(0),
            size=size,
            mode="bilinear",
            align_corners=True,
        ).squeeze(0)
    raise ValueError(f"Unsupported map shape for resize: {tuple(value.shape)}")


def draw_labeled_tile(image_np: np.ndarray, label: str) -> Image.Image:
    image = Image.fromarray(image_np).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle((18, 16, 188, 54), radius=10, fill=(17, 24, 39, 210))
    draw.text((30, 24), label, fill=(255, 255, 255, 255), font=ImageFont.load_default())
    return image


def build_panel(
    tiles: Sequence[Tuple[str, np.ndarray]],
    title: str,
    subtitle: str,
    output_path: Path,
    cols: int = 3,
    gap: int = 18,
    pad: int = 22,
) -> None:
    images = [draw_labeled_tile(image_np, label) for label, image_np in tiles]
    tile_w = max(image.width for image in images)
    tile_h = max(image.height for image in images)
    rows = int(math.ceil(len(images) / float(cols)))
    title_h = 84
    width = pad * 2 + cols * tile_w + gap * (cols - 1)
    height = title_h + pad * 2 + rows * tile_h + gap * (rows - 1)
    canvas = Image.new("RGB", (width, height), (250, 248, 244))
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


def build_contact_sheet(image_paths: Sequence[Path], output_path: Path, gap: int = 18, pad: int = 20) -> None:
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


def to_device_sample(sample: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    batch: Dict[str, Any] = {}
    batch["clip"] = sample["clip"].unsqueeze(0).to(device)
    if "sam_masks" in sample:
        batch["sam_masks"] = sample["sam_masks"].unsqueeze(0).to(device)
    batch["flow_list"] = [flow.unsqueeze(0).to(device) for flow in sample.get("flow_list", [])]
    return batch


def render_sample_panel(
    model: AniFlowFormerTV5,
    sample: Dict[str, Any],
    scene_name: str,
    center: int,
    output_dir: Path,
    device: torch.device,
    use_autocast: bool,
) -> Dict[str, Any]:
    batch = to_device_sample(sample, device)
    clip = batch["clip"]
    sam_masks = batch.get("sam_masks")
    t = clip.shape[1]
    temporal_radius = t // 2
    pair_idx = min(temporal_radius, t - 2)

    with torch.inference_mode():
        with torch.amp.autocast(device_type=device.type, enabled=use_autocast):
            output = model(clip, sam_masks=sam_masks)

    frame_a = tensor_to_uint8_rgb(clip[0, pair_idx].cpu())
    frame_b = tensor_to_uint8_rgb(clip[0, pair_idx + 1].cpu())

    gt_flow = batch["flow_list"][pair_idx][0].cpu()
    pred_flow = output["flows_fw"][pair_idx][0].detach().cpu()
    slot_flow = output["slot_flow_fw"][pair_idx][0].detach().cpu()
    dense_flow = output["dense_prior_flow_fw"][pair_idx][0].detach().cpu() if output.get("dense_prior_flow_fw") else None
    corr_conf = output["corr_confidence_fw"][pair_idx][0].detach().cpu() if output.get("corr_confidence_fw") else None
    target_size = tuple(gt_flow.shape[-2:])

    pred_flow = resize_flow_tensor(pred_flow, target_size)
    slot_flow = resize_flow_tensor(slot_flow, target_size)
    if dense_flow is not None:
        dense_flow = resize_flow_tensor(dense_flow, target_size)
    if corr_conf is not None:
        corr_conf = resize_map_tensor(corr_conf, target_size)

    rad_inputs = [gt_flow.permute(1, 2, 0).numpy(), pred_flow.permute(1, 2, 0).numpy(), slot_flow.permute(1, 2, 0).numpy()]
    if dense_flow is not None:
        rad_inputs.append(dense_flow.permute(1, 2, 0).numpy())
    rad_max = compute_flow_magnitude_radmax(rad_inputs, robust_percentile=95)

    epe_map = torch.norm(pred_flow - gt_flow, dim=0).numpy()
    gt_mag = torch.norm(gt_flow, dim=0).numpy()
    epe_mean = float(epe_map.mean())
    gt_motion_mean = float(gt_mag.mean())

    epe_hi = float(np.percentile(epe_map, 95))
    conf_map = corr_conf[0].numpy() if corr_conf is not None else np.zeros_like(epe_map)
    mask_rgb = label_mask_to_rgb(sam_masks[0, pair_idx, 0].cpu()) if sam_masks is not None else np.zeros_like(frame_a)

    tiles = [
        ("Frame t", frame_a),
        ("Frame t+1", frame_b),
        ("Object labels", mask_rgb),
        ("Slot prior", flow_to_rgb(slot_flow, rad_max)),
        ("Dense prior", flow_to_rgb(dense_flow, rad_max) if dense_flow is not None else np.full_like(frame_a, 242)),
        ("Final flow", flow_to_rgb(pred_flow, rad_max)),
        ("GT flow", flow_to_rgb(gt_flow, rad_max)),
        ("EPE heatmap", float_map_to_rgb(epe_map, getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET), 0.0, epe_hi)),
        ("Corr conf", float_map_to_rgb(conf_map, getattr(cv2, "COLORMAP_VIRIDIS", cv2.COLORMAP_JET), 0.0, 1.0)),
    ]

    frame_tag = sample["frame_name"]
    title = f"{scene_name} | center={center} | frame={frame_tag}"
    subtitle = f"mean EPE={epe_mean:.2f} px | mean GT motion={gt_motion_mean:.2f} px | rad_max={rad_max:.2f}"
    output_path = output_dir / f"{scene_name}_center_{center:04d}_{frame_tag}.png"
    build_panel(tiles, title, subtitle, output_path)

    return {
        "scene": scene_name,
        "center": center,
        "frame": frame_tag,
        "panel_path": str(output_path.relative_to(ROOT)),
        "mean_epe": epe_mean,
        "mean_gt_motion": gt_motion_mean,
        "rad_max": float(rad_max),
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
    temporal_radius = int(cfg["data"].get("T", 5)) // 2
    scene_id, seq = choose_scene(dataset, args.scene)
    scene_name = scene_name_from_seq(seq)
    centers = choose_centers(seq, temporal_radius, args.topk, args.min_gap, args.centers)
    index_map = build_index_map(dataset)

    model = build_model(cfg, checkpoint_path, device)

    panels_dir = output_dir / "panels"
    records = []
    panel_paths = []

    for center in centers:
        key = (scene_id, center, 1)
        if key not in index_map:
            raise KeyError(f"Could not find dataset sample for scene_id={scene_id}, center={center}, stride=1.")
        sample = dataset[index_map[key]]
        sample["frame_name"] = frame_name_from_center(seq, center)
        record = render_sample_panel(model, sample, scene_name, center, panels_dir, device, use_autocast)
        records.append(record)
        panel_paths.append(ROOT / record["panel_path"])
        print(f"[Demo] Saved {record['panel_path']}")

    if panel_paths:
        summary_path = output_dir / f"{scene_name}_summary_sheet.png"
        build_contact_sheet(panel_paths, summary_path)
        print(f"[Demo] Saved {summary_path.relative_to(ROOT)}")

    metadata = {
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "scene": scene_name,
        "centers": centers,
        "records": records,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"[Demo] Saved {metadata_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
