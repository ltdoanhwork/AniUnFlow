#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a SAM structural-prior figure for the thesis.")
    parser.add_argument(
        "--image",
        type=Path,
        default=ROOT / "data/AnimeRun_v2/test/Frame_Anime/cami_02_05_A/original/0306.png",
    )
    parser.add_argument(
        "--sam-mask",
        type=Path,
        default=ROOT / "data/AnimeRun_v2/SAM_Masks_v2/test/Frame_Anime/cami_02_05_A/original/0306.pt",
    )
    parser.add_argument(
        "--contour",
        type=Path,
        default=ROOT / "data/AnimeRun_v2/test/contour/cami_02_05_A/0306.png",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/img/sam_structural_prior_overview.png",
    )
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def load_label_map(path: Path) -> np.ndarray:
    labels = torch.load(path, map_location="cpu")
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    return np.asarray(labels, dtype=np.int32)


def palette_color(index: int) -> tuple[int, int, int]:
    if index == 0:
        return (245, 244, 240)
    hue = (index * 0.61803398875) % 1.0
    sat = 0.55 + 0.25 * ((index * 37) % 7) / 6.0
    val = 0.82 + 0.10 * ((index * 19) % 5) / 4.0
    return hsv_to_rgb(hue, sat, val)


def hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i %= 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return (int(r * 255), int(g * 255), int(b * 255))


def labels_to_rgb(labels: np.ndarray) -> np.ndarray:
    unique = np.unique(labels)
    color_map = {int(label): palette_color(int(label)) for label in unique}
    rgb = np.zeros(labels.shape + (3,), dtype=np.uint8)
    for label, color in color_map.items():
        rgb[labels == label] = color
    return rgb


def label_boundaries(labels: np.ndarray) -> np.ndarray:
    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[1:, :] |= labels[1:, :] != labels[:-1, :]
    boundary[:-1, :] |= labels[1:, :] != labels[:-1, :]
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[:, :-1] |= labels[:, 1:] != labels[:, :-1]
    return boundary


def overlay_boundaries(image: np.ndarray, boundaries: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    output = image.copy()
    output[boundaries] = color
    return output


def contour_to_rgb(contour: np.ndarray) -> np.ndarray:
    if contour.ndim == 2:
        contour_gray = contour
    else:
        contour_gray = contour.mean(axis=2).astype(np.uint8)
    inv = 255 - contour_gray
    rgb = np.stack([inv, inv, inv], axis=-1)
    return rgb.astype(np.uint8)


def sam_overlay(image: np.ndarray, labels_rgb: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    blend = (0.62 * image.astype(np.float32) + 0.38 * labels_rgb.astype(np.float32)).clip(0, 255).astype(np.uint8)
    blend = overlay_boundaries(blend, boundaries, (255, 215, 0))
    return blend


def draw_tile(image_np: np.ndarray, title: str) -> Image.Image:
    image = Image.fromarray(image_np.astype(np.uint8), mode="RGB")
    image = image.resize((430, 242), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (450, 300), (250, 248, 244))
    canvas.paste(image, (10, 48))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.rounded_rectangle((12, 12, 210, 36), radius=8, fill=(18, 24, 33))
    draw.text((24, 18), title, fill=(255, 255, 255), font=font)
    draw.rounded_rectangle((8, 44, 442, 292), radius=12, outline=(215, 208, 198), width=2)
    return canvas


def build_panel(tiles: list[tuple[str, np.ndarray]], output_path: Path) -> None:
    rendered = [draw_tile(image_np, title) for title, image_np in tiles]
    cols = 2
    rows = 2
    gap = 18
    pad = 24
    title_h = 70
    tile_w, tile_h = rendered[0].size
    width = pad * 2 + cols * tile_w + gap
    height = title_h + pad * 2 + rows * tile_h + gap
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    title = "SAM as a structural prior for AniUnFlow"
    subtitle = "RGB appearance alone is ambiguous in anime-style regions; SAM exposes coherent parts and boundaries that stabilize motion reasoning."
    draw.text((pad, 18), title, fill=(18, 24, 33), font=font)
    draw.text((pad, 40), subtitle, fill=(94, 101, 112), font=font)

    for idx, tile in enumerate(rendered):
        row = idx // cols
        col = idx % cols
        x = pad + col * (tile_w + gap)
        y = title_h + pad + row * (tile_h + gap)
        canvas.paste(tile, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    image = load_rgb(args.image)
    contour = load_rgb(args.contour)
    labels = load_label_map(args.sam_mask)

    labels_rgb = labels_to_rgb(labels)
    boundaries = label_boundaries(labels)
    contour_rgb = contour_to_rgb(contour)
    overlay_rgb = sam_overlay(image, labels_rgb, boundaries)

    tiles = [
        ("Animation frame", image),
        ("Contour structure", contour_rgb),
        ("SAM region partition", labels_rgb),
        ("Boundary-aware overlay", overlay_rgb),
    ]
    build_panel(tiles, args.output)
    print(f"[SAMFigure] Saved {args.output}")


if __name__ == "__main__":
    main()
