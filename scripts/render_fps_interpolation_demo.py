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
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataio.clip_dataset_unsup import UnlabeledClipDataset
from models.aniunflow.utils import warp
from models.aniunflow_v5 import AniFlowFormerTV5, V5Config
from models.aniunflow_v6 import AniFlowFormerTV6, V6Config


UPFLOW_MEAN = torch.tensor([104.920005, 110.1753, 114.785955], dtype=torch.float32).view(1, 3, 1, 1)
UPFLOW_SCALE = 0.0039216
DEFAULT_SCENE = "cami_02_05_A"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render x2 FPS interpolation demo and comparison panels.")
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--start", type=int, default=None, help="Start pair index. Auto-picked from high-motion region when omitted.")
    parser.add_argument("--pairs", type=int, default=6, help="Number of consecutive pairs to render.")
    parser.add_argument("--base-fps", type=float, default=12.0, help="Nominal source FPS before interpolation.")
    parser.add_argument("--factor", type=int, default=2, help="Interpolation factor. The current demo is tuned for x2.")
    parser.add_argument("--ours-config", default=str(ROOT / "workspaces" / "v5_4_sam_propagation_memory" / "config.yaml"))
    parser.add_argument("--ours-checkpoint", default=str(ROOT / "workspaces" / "v5_4_sam_propagation_memory" / "best.pth"))
    parser.add_argument("--ours-branch", choices=["auto", "main", "large_motion"], default="auto")
    parser.add_argument("--ours-label", default="AniUnFlow")
    parser.add_argument("--reference-config", default=str(ROOT / "workspaces" / "v5_object_memory_sam_parallel" / "config.yaml"))
    parser.add_argument("--reference-checkpoint", default=str(ROOT / "workspaces" / "v5_object_memory_sam_parallel" / "best.pth"))
    parser.add_argument("--reference-branch", choices=["auto", "main", "large_motion"], default="auto")
    parser.add_argument("--reference-label", default="Object-memory baseline")
    parser.add_argument("--upflow-label", default="UPFlow")
    parser.add_argument("--data-root", default=str(ROOT / "data" / "AnimeRun_v2"))
    parser.add_argument("--sam-mask-root", default=str(ROOT / "data" / "AnimeRun_v2" / "SAM_Masks_v2"))
    parser.add_argument("--output-dir", default=str(ROOT / "demo" / "fps_demo"))
    parser.add_argument("--report-dir", default=str(ROOT / "reports" / "img" / "fps_demo"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fp32", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def patch_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if "sam" in cfg:
        cfg["sam"]["encoder_config"] = "configs/sam2.1/sam2.1_hiera_t.yaml"
    return cfg


def infer_branch(cfg: Dict[str, Any], branch_override: str) -> str:
    if branch_override != "auto":
        return branch_override
    backbone = str(cfg.get("model", {}).get("backbone", "")).lower()
    return "large_motion" if backbone.startswith("v6_") or "global_slot_search" in backbone else "main"


def load_model(config_path: Path, checkpoint_path: Path, device: torch.device, branch_override: str) -> torch.nn.Module:
    cfg = patch_cfg(load_yaml(config_path))
    branch = infer_branch(cfg, branch_override)
    if branch == "large_motion":
        model = AniFlowFormerTV6(V6Config.from_dict(cfg)).to(device).eval()
    else:
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


def tensor_to_rgb_uint8(image: torch.Tensor) -> np.ndarray:
    return np.uint8(np.round(image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0))


def scene_name(seq: Dict[str, Any]) -> str:
    first = seq["frames"][0]
    return first.parents[1].name if first.parent.name == "original" else first.parent.name


def mean_motion(flow_tensor: torch.Tensor) -> float:
    return float(torch.norm(flow_tensor, dim=0).mean().item())


def select_demo_start(dataset: UnlabeledClipDataset, scene_id: int, seq: Dict[str, Any], pair_count: int) -> int:
    temporal_radius = dataset.T // 2
    index_map = {tuple(item): idx for idx, item in enumerate(dataset.index)}
    valid_centers = list(range(temporal_radius, len(seq["frames"]) - temporal_radius))
    scored: List[Tuple[float, int]] = []
    for center in valid_centers:
        key = (scene_id, center, 1)
        if key not in index_map:
            continue
        sample = dataset[index_map[key]]
        score = mean_motion(sample["flow_list"][dataset.T // 2])
        scored.append((score, center))
    if not scored:
        return temporal_radius

    scored.sort(reverse=True)
    best_center = scored[0][1]
    start_min = temporal_radius
    start_max = max(start_min, len(seq["frames"]) - temporal_radius - pair_count)
    return max(start_min, min(best_center - max(pair_count // 2, 1), start_max))


def sample_pair(dataset: UnlabeledClipDataset, scene_id: int, center: int) -> Dict[str, Any]:
    index_map = {tuple(item): idx for idx, item in enumerate(dataset.index)}
    key = (scene_id, center, 1)
    if key not in index_map:
        raise KeyError(f"Missing sample for scene_id={scene_id}, center={center}")
    sample = dataset[index_map[key]]
    sample["frame_name"] = dataset.test_seqs[scene_id]["frames"][center].stem
    return sample


def run_pair_model(
    model: torch.nn.Module,
    clip: torch.Tensor,
    sam_masks: torch.Tensor,
    center_pair_idx: int,
    reverse_pair_idx: int,
    device: torch.device,
    use_autocast: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    clip_b = clip.unsqueeze(0).to(device)
    masks_b = sam_masks.unsqueeze(0).to(device)
    rev_clip_b = torch.flip(clip_b, dims=[1])
    rev_masks_b = torch.flip(masks_b, dims=[1])
    with torch.inference_mode():
        with torch.amp.autocast(device_type=device.type, enabled=use_autocast):
            out_fw = model(clip_b, sam_masks=masks_b)
            out_bw = model(rev_clip_b, sam_masks=rev_masks_b)
    flow_fw = out_fw["flows_fw"][center_pair_idx][0].detach()
    flow_bw = out_bw["flows_fw"][reverse_pair_idx][0].detach()
    return flow_fw, flow_bw


def run_pair_upflow(
    model: torch.nn.Module,
    img0: torch.Tensor,
    img1: torch.Tensor,
    device: torch.device,
    use_autocast: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    start = torch.zeros((1, 2, 1, 1), device=device, dtype=img0.dtype)

    def infer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            with torch.amp.autocast(device_type=device.type, enabled=use_autocast):
                out = model(
                    {
                        "im1": normalize_upflow(a.unsqueeze(0).to(device)),
                        "im2": normalize_upflow(b.unsqueeze(0).to(device)),
                        "im1_raw": normalize_upflow(a.unsqueeze(0).to(device)),
                        "im2_raw": normalize_upflow(b.unsqueeze(0).to(device)),
                        "im1_sp": normalize_upflow(a.unsqueeze(0).to(device)),
                        "im2_sp": normalize_upflow(b.unsqueeze(0).to(device)),
                        "start": start,
                        "if_loss": False,
                    }
                )
        return out["flow_f_out"][0].detach()

    return infer(img0, img1), infer(img1, img0)


def synthesize_middle_frame(img0: torch.Tensor, img1: torch.Tensor, flow_01: torch.Tensor, flow_10: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    img0_b = img0.unsqueeze(0)
    img1_b = img1.unsqueeze(0)
    flow_01_b = flow_01.unsqueeze(0).to(img0.device)
    flow_10_b = flow_10.unsqueeze(0).to(img0.device)

    pred_from_0 = warp(img0_b, -alpha * flow_01_b)
    pred_from_1 = warp(img1_b, (1.0 - alpha) * flow_10_b)

    fb_consistency_0 = torch.norm(flow_01_b + warp(flow_10_b, flow_01_b), dim=1, keepdim=True)
    fb_consistency_1 = torch.norm(flow_10_b + warp(flow_01_b, flow_10_b), dim=1, keepdim=True)
    weight_0 = torch.exp(-fb_consistency_0 / 2.0)
    weight_1 = torch.exp(-fb_consistency_1 / 2.0)
    denom = (weight_0 + weight_1).clamp_min(1e-6)
    blended = (weight_0 * pred_from_0 + weight_1 * pred_from_1) / denom
    return blended[0].clamp(0.0, 1.0)


def draw_label(image_np: np.ndarray, text: str) -> Image.Image:
    image = Image.fromarray(image_np).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle((14, 14, 220, 48), radius=10, fill=(17, 24, 39, 210))
    draw.text((24, 22), text, fill=(255, 255, 255, 255), font=ImageFont.load_default())
    return image


def build_row_panel(rows: Sequence[Sequence[Tuple[str, np.ndarray]]], row_titles: Sequence[str], title: str, subtitle: str, output_path: Path) -> None:
    if not rows:
        return
    pad = 22
    gap = 16
    title_h = 82
    row_gap = 22
    label_w = 148

    rendered_rows = [[draw_label(image_np, label) for label, image_np in row] for row in rows]
    tile_w = max(img.width for row in rendered_rows for img in row)
    tile_h = max(img.height for row in rendered_rows for img in row)
    cols = max(len(row) for row in rendered_rows)
    total_w = pad * 2 + label_w + cols * tile_w + max(0, cols - 1) * gap
    total_h = title_h + pad * 2 + len(rendered_rows) * tile_h + max(0, len(rendered_rows) - 1) * row_gap

    canvas = Image.new("RGB", (total_w, total_h), (248, 246, 242))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((pad, 18), title, fill=(16, 22, 32), font=font)
    draw.text((pad, 42), subtitle, fill=(86, 92, 104), font=font)

    for row_idx, row in enumerate(rendered_rows):
        y = title_h + pad + row_idx * (tile_h + row_gap)
        draw.text((pad, y + 16), row_titles[row_idx], fill=(16, 22, 32), font=font)
        x = pad + label_w
        for image in row:
            canvas.paste(image, (x, y))
            x += tile_w + gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def make_video_frame(columns: Sequence[Tuple[str, np.ndarray]], title: str, footer: str) -> np.ndarray:
    images = [draw_label(image_np, label) for label, image_np in columns]
    tile_w = max(image.width for image in images)
    tile_h = max(image.height for image in images)
    pad = 20
    gap = 16
    title_h = 66
    footer_h = 32
    width = pad * 2 + len(images) * tile_w + max(0, len(images) - 1) * gap
    height = title_h + tile_h + footer_h + pad * 2
    canvas = Image.new("RGB", (width, height), (248, 246, 242))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((pad, 16), title, fill=(16, 22, 32), font=font)
    y = title_h
    x = pad
    for image in images:
        canvas.paste(image, (x, y))
        x += tile_w + gap
    draw.text((pad, title_h + tile_h + 10), footer, fill=(86, 92, 104), font=font)
    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)


def save_video(frames: Sequence[np.ndarray], fps: float, output_path: Path) -> None:
    if not frames:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        writer.write(frame)
    writer.release()


def main() -> None:
    args = parse_args()
    if args.factor != 2:
        raise ValueError("The current interpolation demo is implemented for x2 FPS only.")

    device = torch.device(args.device)
    use_autocast = (device.type == "cuda") and (not args.fp32)

    output_dir = Path(args.output_dir).resolve()
    report_dir = Path(args.report_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    dataset = UnlabeledClipDataset(
        root=args.data_root,
        T=5,
        crop_size=(256, 512),
        resize=True,
        keep_aspect=False,
        load_sam_masks=True,
        sam_mask_root=args.sam_mask_root,
        sam_mask_cache_size=64,
        is_test=True,
    )

    seq = None
    scene_id = None
    for sid, item in enumerate(dataset.test_seqs):
        if scene_name(item) == args.scene:
            scene_id = sid
            seq = item
            break
    if seq is None or scene_id is None:
        raise ValueError(f"Scene '{args.scene}' not found.")

    start = args.start if args.start is not None else select_demo_start(dataset, scene_id, seq, args.pairs)
    temporal_radius = dataset.T // 2
    max_start = len(seq["frames"]) - temporal_radius - args.pairs
    start = max(temporal_radius, min(start, max_start))
    centers = list(range(start, start + args.pairs))

    print(f"[FPS Demo] Scene={args.scene} start={start} pairs={len(centers)}")

    model_ours = load_model(Path(args.ours_config), Path(args.ours_checkpoint), device, args.ours_branch)
    model_reference = load_model(Path(args.reference_config), Path(args.reference_checkpoint), device, args.reference_branch)
    model_upflow = load_upflow_model(device)

    pair_idx = dataset.T // 2
    reverse_pair_idx = max(pair_idx - 1, 0)

    per_model_frames: Dict[str, List[np.ndarray]] = {
        args.ours_label: [],
        args.reference_label: [],
        args.upflow_label: [],
    }
    comparison_frames: List[np.ndarray] = []
    paper_rows: List[List[Tuple[str, np.ndarray]]] = []
    paper_row_titles: List[str] = []
    metadata_records: List[Dict[str, Any]] = []

    paper_centers = [centers[1] if len(centers) > 1 else centers[0], centers[-2] if len(centers) > 2 else centers[-1]]
    seen_paper_centers = set()

    for frame_id, center in enumerate(centers):
        sample = sample_pair(dataset, scene_id, center)
        clip = sample["clip"]
        sam_masks = sample["sam_masks"]
        img0 = clip[pair_idx]
        img1 = clip[pair_idx + 1]
        size = tuple(img0.shape[-2:])
        frame_name = seq["frames"][center].stem
        next_name = seq["frames"][center + 1].stem

        flow_ours_01, flow_ours_10 = run_pair_model(model_ours, clip, sam_masks, pair_idx, reverse_pair_idx, device, use_autocast)
        flow_ref_01, flow_ref_10 = run_pair_model(model_reference, clip, sam_masks, pair_idx, reverse_pair_idx, device, use_autocast)
        flow_up_01, flow_up_10 = run_pair_upflow(model_upflow, img0, img1, device, use_autocast)

        flow_ours_01 = resize_flow_tensor(flow_ours_01, size)
        flow_ours_10 = resize_flow_tensor(flow_ours_10, size)
        flow_ref_01 = resize_flow_tensor(flow_ref_01, size)
        flow_ref_10 = resize_flow_tensor(flow_ref_10, size)
        flow_up_01 = resize_flow_tensor(flow_up_01, size)
        flow_up_10 = resize_flow_tensor(flow_up_10, size)

        mid_ours = synthesize_middle_frame(img0, img1, flow_ours_01, flow_ours_10)
        mid_ref = synthesize_middle_frame(img0, img1, flow_ref_01, flow_ref_10)
        mid_up = synthesize_middle_frame(img0, img1, flow_up_01, flow_up_10)

        rgb0 = tensor_to_rgb_uint8(img0)
        rgb1 = tensor_to_rgb_uint8(img1)
        rgb_ours = tensor_to_rgb_uint8(mid_ours)
        rgb_ref = tensor_to_rgb_uint8(mid_ref)
        rgb_up = tensor_to_rgb_uint8(mid_up)

        for label in per_model_frames:
            if not per_model_frames[label]:
                per_model_frames[label].append(cv2.cvtColor(rgb0, cv2.COLOR_RGB2BGR))
        per_model_frames[args.ours_label].append(cv2.cvtColor(rgb_ours, cv2.COLOR_RGB2BGR))
        per_model_frames[args.ours_label].append(cv2.cvtColor(rgb1, cv2.COLOR_RGB2BGR))
        per_model_frames[args.reference_label].append(cv2.cvtColor(rgb_ref, cv2.COLOR_RGB2BGR))
        per_model_frames[args.reference_label].append(cv2.cvtColor(rgb1, cv2.COLOR_RGB2BGR))
        per_model_frames[args.upflow_label].append(cv2.cvtColor(rgb_up, cv2.COLOR_RGB2BGR))
        per_model_frames[args.upflow_label].append(cv2.cvtColor(rgb1, cv2.COLOR_RGB2BGR))

        title_real = f"{args.scene} | source frame {frame_name}"
        title_mid = f"{args.scene} | x2 in-between {frame_name}->{next_name}"
        footer = f"Output FPS = {args.base_fps * args.factor:.1f} from {args.base_fps:.1f} fps source"
        real_columns = [
            (args.upflow_label, rgb0),
            (args.reference_label, rgb0),
            (args.ours_label, rgb0),
        ]
        mid_columns = [
            (args.upflow_label, rgb_up),
            (args.reference_label, rgb_ref),
            (args.ours_label, rgb_ours),
        ]
        comparison_frames.append(make_video_frame(real_columns, title_real, footer))
        comparison_frames.append(make_video_frame(mid_columns, title_mid, footer))
        if frame_id == len(centers) - 1:
            comparison_frames.append(make_video_frame(real_columns, f"{args.scene} | source frame {next_name}", footer))

        pair_dir = output_dir / "frames" / f"{center:04d}_{frame_name}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb0).save(pair_dir / "frame_t.png")
        Image.fromarray(rgb1).save(pair_dir / "frame_t1.png")
        Image.fromarray(rgb_up).save(pair_dir / "upflow_mid.png")
        Image.fromarray(rgb_ref).save(pair_dir / "reference_mid.png")
        Image.fromarray(rgb_ours).save(pair_dir / "ours_mid.png")

        if center in paper_centers and center not in seen_paper_centers:
            paper_rows.append(
                [
                    ("Frame t", rgb0),
                    (args.upflow_label, rgb_up),
                    (args.reference_label, rgb_ref),
                    (args.ours_label, rgb_ours),
                    ("Frame t+1", rgb1),
                ]
            )
            paper_row_titles.append(f"{frame_name} -> {next_name}")
            seen_paper_centers.add(center)

        metadata_records.append(
            {
                "center": center,
                "frame": frame_name,
                "next_frame": next_name,
                "saved_dir": str(pair_dir.relative_to(ROOT)),
            }
        )
        print(f"[FPS Demo] Rendered pair {center}: {frame_name}->{next_name}")

    target_fps = args.base_fps * args.factor
    for label, frames in per_model_frames.items():
        stem = label.lower().replace(" ", "_").replace("-", "_")
        save_video(frames, target_fps, output_dir / f"{stem}_x2.mp4")
        print(f"[FPS Demo] Saved {stem}_x2.mp4")
    save_video(comparison_frames, target_fps, output_dir / "comparison_x2.mp4")
    print("[FPS Demo] Saved comparison_x2.mp4")

    panel_path = output_dir / f"{args.scene}_fps_interpolation_sheet.png"
    panel_title = f"{args.scene} x2 FPS interpolation demo"
    panel_subtitle = f"Columns show source frame, two baseline in-betweens, AniUnFlow in-between, and next frame."
    build_row_panel(paper_rows, paper_row_titles, panel_title, panel_subtitle, panel_path)
    report_panel_path = report_dir / f"{args.scene}_fps_interpolation_sheet.png"
    report_panel_path.write_bytes(panel_path.read_bytes())
    print(f"[FPS Demo] Saved {panel_path.relative_to(ROOT)}")
    print(f"[FPS Demo] Saved {report_panel_path.relative_to(ROOT)}")

    metadata = {
        "scene": args.scene,
        "start": start,
        "pairs": len(centers),
        "base_fps": args.base_fps,
        "output_fps": target_fps,
        "models": [args.upflow_label, args.reference_label, args.ours_label],
        "records": metadata_records,
        "report_panel": str(report_panel_path.relative_to(ROOT)),
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[FPS Demo] Saved {metadata_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
