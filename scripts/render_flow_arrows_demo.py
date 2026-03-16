#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.frame_utils import readFlow


def draw_arrow(draw: ImageDraw.ImageDraw, x0: float, y0: float, x1: float, y1: float, color, width: int = 2) -> None:
    draw.line((x0, y0, x1, y1), fill=color, width=width)
    angle = math.atan2(y1 - y0, x1 - x0)
    head_len = max(6, 3 * width)
    head_angle = math.pi / 7
    xa = x1 - head_len * math.cos(angle - head_angle)
    ya = y1 - head_len * math.sin(angle - head_angle)
    xb = x1 - head_len * math.cos(angle + head_angle)
    yb = y1 - head_len * math.sin(angle + head_angle)
    draw.polygon([(x1, y1), (xa, ya), (xb, yb)], fill=color)


def render_arrow_overlay(
    image_path: Path,
    flow_path: Path,
    output_path: Path,
    label: str,
    step: int = 56,
    min_mag: float = 6.0,
    scale: float = 0.22,
    max_len: float = 34.0,
) -> None:
    image = Image.open(image_path).convert("RGBA")
    flow = readFlow(str(flow_path))
    draw = ImageDraw.Draw(image, "RGBA")

    h, w = flow.shape[:2]
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            u, v = float(flow[y, x, 0]), float(flow[y, x, 1])
            mag = math.sqrt(u * u + v * v)
            if mag < min_mag:
                continue

            dx = max(-max_len, min(max_len, u * scale))
            dy = max(-max_len, min(max_len, v * scale))
            # magnitude-coded warm color
            t = min(1.0, mag / 120.0)
            color = (
                int(255),
                int(210 - 90 * t),
                int(40 + 80 * t),
                235,
            )
            draw_arrow(draw, x, y, x + dx, y + dy, color=color, width=2)
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(20, 20, 20, 220))

    # small label tag
    draw.rounded_rectangle((18, 16, 128, 52), radius=10, fill=(17, 24, 39, 210))
    draw.text((30, 24), label, fill=(255, 255, 255, 255))
    draw.rounded_rectangle((18, 60, 220, 92), radius=8, fill=(255, 255, 255, 170))
    draw.text((28, 68), "GT flow arrows", fill=(20, 20, 20, 255))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path)


def build_triptych(image_paths: list[Path], output_path: Path, gap: int = 18, pad: int = 20) -> None:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    width = sum(image.width for image in images) + gap * (len(images) - 1) + pad * 2
    height = max(image.height for image in images) + pad * 2
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    x = pad
    for image in images:
        canvas.paste(image, (x, pad))
        x += image.width + gap
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render optical flow arrows overlay demo.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--pair",
        nargs=3,
        action="append",
        metavar=("LABEL", "IMAGE", "FLOW"),
        required=True,
    )
    parser.add_argument("--triptych-name", default="flow_arrows_triptych.png")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    rendered = []
    for label, image_arg, flow_arg in args.pair:
        out_path = output_dir / f"{label}_flow_arrows.png"
        render_arrow_overlay(Path(image_arg), Path(flow_arg), out_path, label=label)
        rendered.append(out_path)
        print(out_path)

    build_triptych(rendered, output_dir / args.triptych_name)
    print(output_dir / args.triptych_name)


if __name__ == "__main__":
    main()
