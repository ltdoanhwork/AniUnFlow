#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataio.clip_dataset_unsup import UnlabeledClipDataset
from models.aniunflow.utils import warp
from models.aniunflow_v5 import AniFlowFormerTV5, V5Config
from models.aniunflow_v6 import AniFlowFormerTV6, V6Config


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
    parser = argparse.ArgumentParser(description="Render downstream propagation task demos from AniUnFlow and a baseline.")
    parser.add_argument("--scene", default="cami_02_05_A")
    parser.add_argument("--center", type=int, default=31, help="Center index in the T=5 test clip.")
    parser.add_argument("--ours-config", default=str(ROOT / "workspaces" / "v5_4_sam_propagation_memory" / "config.yaml"))
    parser.add_argument("--ours-checkpoint", default=str(ROOT / "workspaces" / "v5_4_sam_propagation_memory" / "best.pth"))
    parser.add_argument("--ours-branch", choices=["auto", "main", "large_motion"], default="auto")
    parser.add_argument("--ours-label", default="AniUnFlow")
    parser.add_argument("--reference-config", default=str(ROOT / "workspaces" / "v5_object_memory_sam_parallel" / "config.yaml"))
    parser.add_argument("--reference-checkpoint", default=str(ROOT / "workspaces" / "v5_object_memory_sam_parallel" / "best.pth"))
    parser.add_argument("--reference-branch", choices=["auto", "main", "large_motion"], default="auto")
    parser.add_argument("--reference-label", default="Object-memory baseline")
    parser.add_argument("--data-root", default=str(ROOT / "data" / "AnimeRun_v2"))
    parser.add_argument("--sam-mask-root", default=str(ROOT / "data" / "AnimeRun_v2" / "SAM_Masks_v2"))
    parser.add_argument("--output-dir", default=str(ROOT / "demo" / "downstream_tasks"))
    parser.add_argument("--report-dir", default=str(ROOT / "reports" / "img" / "downstream_tasks"))
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


def run_pair_model(
    model: torch.nn.Module,
    clip: torch.Tensor,
    sam_masks: torch.Tensor,
    pair_idx: int,
    reverse_pair_idx: int,
    device: torch.device,
    use_autocast: bool,
) -> torch.Tensor:
    clip_b = clip.unsqueeze(0).to(device)
    masks_b = sam_masks.unsqueeze(0).to(device)
    rev_clip_b = torch.flip(clip_b, dims=[1])
    rev_masks_b = torch.flip(masks_b, dims=[1])
    with torch.inference_mode():
        with torch.amp.autocast(device_type=device.type, enabled=use_autocast):
            out_fw = model(clip_b, sam_masks=masks_b)
            out_bw = model(rev_clip_b, sam_masks=rev_masks_b)
    flow_01 = out_fw["flows_fw"][pair_idx][0].detach()
    flow_10 = out_bw["flows_fw"][reverse_pair_idx][0].detach()
    return flow_01, flow_10


def load_contour(scene: str, frame_name: str, target_size: Tuple[int, int]) -> torch.Tensor:
    contour_path = ROOT / "data" / "AnimeRun_v2" / "test" / "contour" / scene / f"{frame_name}.png"
    contour = Image.open(contour_path).convert("L")
    h, w = target_size
    contour = contour.resize((w, h), Image.Resampling.BILINEAR)
    contour_np = np.asarray(contour, dtype=np.float32) / 255.0
    return torch.from_numpy(contour_np).unsqueeze(0)


def label_map_to_rgb(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros(mask.shape + (3,), dtype=np.uint8)
    pos = mask > 0
    if pos.any():
        rgb[pos] = PALETTE[1 + (mask[pos] - 1) % (len(PALETTE) - 1)]
    return rgb


def overlay_segmentation(image_rgb: np.ndarray, labels: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color = label_map_to_rgb(labels)
    out = image_rgb.copy().astype(np.float32)
    pos = labels > 0
    out[pos] = (1.0 - alpha) * out[pos] + alpha * color[pos].astype(np.float32)

    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[1:, :] |= labels[1:, :] != labels[:-1, :]
    boundary[:-1, :] |= labels[1:, :] != labels[:-1, :]
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[:, :-1] |= labels[:, 1:] != labels[:, :-1]
    out[boundary] = np.array([255, 214, 74], dtype=np.float32)
    return np.uint8(np.clip(out, 0, 255))


def warp_dense_tensor(src: torch.Tensor, flow_01: torch.Tensor) -> torch.Tensor:
    src_b = src.unsqueeze(0) if src.ndim == 3 else src.unsqueeze(0).unsqueeze(0)
    warped = warp(src_b, -flow_01.unsqueeze(0).to(src.device))
    return warped[0]


def warp_label_map(src_labels: torch.Tensor, flow_01: torch.Tensor) -> torch.Tensor:
    max_label = int(src_labels.max().item())
    if max_label <= 0:
        return torch.zeros_like(src_labels)
    one_hot = F.one_hot(src_labels.long(), num_classes=max_label + 1).permute(2, 0, 1).float()
    warped = warp(one_hot.unsqueeze(0), -flow_01.unsqueeze(0).to(one_hot.device))[0]
    return warped.argmax(dim=0).to(torch.uint8)


def stylize_frame(frame_rgb: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
    image = Image.fromarray(frame_rgb).convert("RGB")
    poster = ImageOps.posterize(image, 3)
    arr = np.asarray(poster, dtype=np.float32)
    arr = 0.78 * arr + 0.22 * np.array([72, 160, 210], dtype=np.float32)
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    gray = image.convert("L").filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(3))
    edge = np.asarray(gray, dtype=np.uint8)
    edge_mask = edge > 28
    arr[edge_mask] = np.array([255, 238, 180], dtype=np.uint8)

    if labels is not None:
        focus = labels > 0
        arr[focus] = np.clip(0.8 * arr[focus].astype(np.float32) + 0.2 * np.array([255, 170, 120], dtype=np.float32), 0, 255).astype(np.uint8)
    return arr


def overlay_contour(frame_rgb: np.ndarray, contour_gray: np.ndarray, color: Tuple[int, int, int] = (255, 86, 86)) -> np.ndarray:
    out = frame_rgb.copy()
    edge = contour_gray < 0.9
    if edge.any():
        edge = cv2.dilate(edge.astype(np.uint8), np.ones((2, 2), dtype=np.uint8), iterations=1).astype(bool)
    out[edge] = np.array(color, dtype=np.uint8)
    return out


def draw_tile(image_np: np.ndarray, label: str) -> Image.Image:
    image = Image.fromarray(image_np).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle((14, 14, 200, 48), radius=10, fill=(17, 24, 39, 210))
    draw.text((24, 22), label, fill=(255, 255, 255, 255))
    return image


def build_task_panel(
    rows: Sequence[Tuple[str, Sequence[Tuple[str, np.ndarray]]]],
    title: str,
    subtitle: str,
    output_path: Path,
) -> None:
    pad = 22
    col_gap = 16
    row_gap = 18
    label_w = 170
    title_h = 84
    rendered_rows: List[Tuple[str, List[Image.Image]]] = []
    for row_title, cols in rows:
        rendered_rows.append((row_title, [draw_tile(img, label) for label, img in cols]))

    tile_w = max(img.width for _, row in rendered_rows for img in row)
    tile_h = max(img.height for _, row in rendered_rows for img in row)
    n_cols = max(len(row) for _, row in rendered_rows)
    width = pad * 2 + label_w + n_cols * tile_w + (n_cols - 1) * col_gap
    height = title_h + pad * 2 + len(rendered_rows) * tile_h + (len(rendered_rows) - 1) * row_gap
    canvas = Image.new("RGB", (width, height), (248, 246, 242))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 18), title, fill=(16, 22, 32))
    draw.text((pad, 44), subtitle, fill=(86, 92, 104))

    for ridx, (row_title, row_imgs) in enumerate(rendered_rows):
        y = title_h + pad + ridx * (tile_h + row_gap)
        draw.text((pad, y + 18), row_title, fill=(16, 22, 32))
        x = pad + label_w
        for img in row_imgs:
            canvas.paste(img, (x, y))
            x += tile_w + col_gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def build_single_row_panel(
    row_title: str,
    cols: Sequence[Tuple[str, np.ndarray]],
    title: str,
    subtitle: str,
    output_path: Path,
) -> None:
    build_task_panel([(row_title, cols)], title=title, subtitle=subtitle, output_path=output_path)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    use_autocast = (device.type == "cuda") and (not args.fp32)

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

    index_map = {tuple(item): idx for idx, item in enumerate(dataset.index)}
    key = (scene_id, args.center, 1)
    if key not in index_map:
        raise ValueError(f"Center {args.center} unavailable for scene {args.scene}.")
    sample = dataset[index_map[key]]
    pair_idx = dataset.T // 2
    reverse_pair_idx = max(pair_idx - 1, 0)

    clip = sample["clip"]
    sam_masks = sample["sam_masks"]
    img0 = clip[pair_idx]
    img1 = clip[pair_idx + 1]
    size = tuple(img0.shape[-2:])
    frame_name = seq["frames"][args.center].stem
    next_name = seq["frames"][args.center + 1].stem

    model_ours = load_model(Path(args.ours_config), Path(args.ours_checkpoint), device, args.ours_branch)
    model_ref = load_model(Path(args.reference_config), Path(args.reference_checkpoint), device, args.reference_branch)

    flow_ours_01, flow_ours_10 = run_pair_model(model_ours, clip, sam_masks, pair_idx, reverse_pair_idx, device, use_autocast)
    flow_ref_01, flow_ref_10 = run_pair_model(model_ref, clip, sam_masks, pair_idx, reverse_pair_idx, device, use_autocast)
    flow_ours_01 = resize_flow_tensor(flow_ours_01, size).cpu()
    flow_ref_01 = resize_flow_tensor(flow_ref_01, size).cpu()

    frame0_rgb = tensor_to_rgb_uint8(img0)
    frame1_rgb = tensor_to_rgb_uint8(img1)

    src_labels = sam_masks[pair_idx, 0].cpu().to(torch.uint8)
    tgt_labels = sam_masks[pair_idx + 1, 0].cpu().to(torch.uint8)
    prop_ref_labels = warp_label_map(src_labels, flow_ref_01)
    prop_ours_labels = warp_label_map(src_labels, flow_ours_01)

    contour0 = load_contour(args.scene, frame_name, size).cpu()
    contour1 = load_contour(args.scene, next_name, size).cpu()
    prop_ref_contour = warp_dense_tensor(contour0, flow_ref_01).clamp(0.0, 1.0)
    prop_ours_contour = warp_dense_tensor(contour0, flow_ours_01).clamp(0.0, 1.0)

    prop_ref_rgb = warp_dense_tensor(img0.cpu(), flow_ref_01).clamp(0.0, 1.0)
    prop_ours_rgb = warp_dense_tensor(img0.cpu(), flow_ours_01).clamp(0.0, 1.0)

    styled0 = stylize_frame(frame0_rgb, src_labels.numpy())
    styled1 = stylize_frame(frame1_rgb, tgt_labels.numpy())
    styled0_tensor = torch.from_numpy(styled0).permute(2, 0, 1).float() / 255.0
    prop_ref_style = tensor_to_rgb_uint8(warp_dense_tensor(styled0_tensor, flow_ref_01).clamp(0.0, 1.0))
    prop_ours_style = tensor_to_rgb_uint8(warp_dense_tensor(styled0_tensor, flow_ours_01).clamp(0.0, 1.0))

    rows = [
        (
            "Mask propagation",
            [
                ("Source mask", overlay_segmentation(frame0_rgb, src_labels.numpy())),
                (args.reference_label, overlay_segmentation(frame1_rgb, prop_ref_labels.numpy())),
                (args.ours_label, overlay_segmentation(frame1_rgb, prop_ours_labels.numpy())),
                ("Target mask", overlay_segmentation(frame1_rgb, tgt_labels.numpy())),
            ],
        ),
        (
            "Color propagation",
            [
                ("Source color", frame0_rgb),
                (args.reference_label, tensor_to_rgb_uint8(prop_ref_rgb)),
                (args.ours_label, tensor_to_rgb_uint8(prop_ours_rgb)),
                ("Target frame", frame1_rgb),
            ],
        ),
        (
            "Contour propagation",
            [
                ("Source contour", overlay_contour(frame0_rgb, contour0[0].numpy())),
                (args.reference_label, overlay_contour(frame1_rgb, prop_ref_contour[0].numpy())),
                (args.ours_label, overlay_contour(frame1_rgb, prop_ours_contour[0].numpy())),
                ("Target contour", overlay_contour(frame1_rgb, contour1[0].numpy())),
            ],
        ),
        (
            "Stylization consistency",
            [
                ("Styled source", styled0),
                (args.reference_label, prop_ref_style),
                (args.ours_label, prop_ours_style),
                ("Styled target", styled1),
            ],
        ),
    ]

    output_dir = Path(args.output_dir).resolve()
    report_dir = Path(args.report_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    panel_path = output_dir / f"{args.scene}_{frame_name}_downstream_tasks.png"
    build_task_panel(
        rows,
        title=f"{args.scene} downstream propagation demos",
        subtitle=f"Pair {frame_name}->{next_name}: baseline vs AniUnFlow across four flow-driven tasks.",
        output_path=panel_path,
    )
    report_path = report_dir / f"{args.scene}_{frame_name}_downstream_tasks.png"
    report_path.write_bytes(panel_path.read_bytes())

    task_stems = {
        "Mask propagation": "mask_propagation",
        "Color propagation": "color_propagation",
        "Contour propagation": "contour_propagation",
        "Stylization consistency": "stylization_consistency",
    }
    for row_title, cols in rows:
        stem = task_stems[row_title]
        row_path = output_dir / f"{args.scene}_{frame_name}_{stem}.png"
        build_single_row_panel(
            row_title=row_title,
            cols=cols,
            title=f"{args.scene} {row_title.lower()}",
            subtitle=f"Pair {frame_name}->{next_name}: source, baseline propagation, AniUnFlow propagation, and target view.",
            output_path=row_path,
        )
        row_report_path = report_dir / f"{args.scene}_{frame_name}_{stem}.png"
        row_report_path.write_bytes(row_path.read_bytes())
        print(row_path.relative_to(ROOT))
        print(row_report_path.relative_to(ROOT))

    print(panel_path.relative_to(ROOT))
    print(report_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
