#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageChops, ImageDraw, ImageFilter


def load_binary_mask(mask_path: Path) -> Image.Image:
    mask = torch.load(mask_path, map_location="cpu")
    if isinstance(mask, torch.Tensor):
        arr = mask.detach().cpu().numpy()
    else:
        arr = np.asarray(mask)
    arr = (arr > 0).astype(np.uint8) * 255
    return Image.fromarray(arr, mode="L")


def render_overlay(
    image_path: Path,
    mask_path: Path,
    output_path: Path,
    label: str,
    fill_rgb: tuple[int, int, int] = (46, 196, 182),
    edge_rgb: tuple[int, int, int] = (255, 190, 11),
    fill_alpha: int = 108,
    edge_alpha: int = 245,
) -> None:
    image = Image.open(image_path).convert("RGBA")
    mask = load_binary_mask(mask_path)

    fill = Image.new("RGBA", image.size, fill_rgb + (0,))
    fill.putalpha(mask.point(lambda p: fill_alpha if p > 0 else 0))
    over = Image.alpha_composite(image, fill)

    dilated = mask.filter(ImageFilter.MaxFilter(5))
    eroded = mask.filter(ImageFilter.MinFilter(5))
    edge = ImageChops.subtract(dilated, eroded).filter(ImageFilter.GaussianBlur(0.5))
    edge_layer = Image.new("RGBA", image.size, edge_rgb + (0,))
    edge_layer.putalpha(edge.point(lambda p: edge_alpha if p > 0 else 0))
    over = Image.alpha_composite(over, edge_layer)

    draw = ImageDraw.Draw(over, "RGBA")
    draw.rounded_rectangle((18, 16, 118, 52), radius=10, fill=(17, 24, 39, 210))
    draw.text((34, 24), label, fill=(255, 255, 255, 255))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    over.save(output_path)


def build_triptych(image_paths: list[Path], output_path: Path, gap: int = 18, pad: int = 20) -> None:
    images = [Image.open(path).convert("RGBA") for path in image_paths]
    width = sum(image.width for image in images) + gap * (len(images) - 1) + pad * 2
    height = max(image.height for image in images) + pad * 2
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))

    x = pad
    for image in images:
        canvas.alpha_composite(image, (x, pad))
        x += image.width + gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render SAM mask overlay demo images.")
    parser.add_argument("--output-dir", required=True, help="Directory for individual overlays.")
    parser.add_argument(
        "--pair",
        nargs=3,
        action="append",
        metavar=("LABEL", "IMAGE", "MASK"),
        required=True,
        help="One overlay item as: LABEL IMAGE_PATH MASK_PATH",
    )
    parser.add_argument("--triptych-name", default="sam_overlay_triptych.png")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    rendered_paths: list[Path] = []
    for label, image_arg, mask_arg in args.pair:
        image_path = Path(image_arg)
        mask_path = Path(mask_arg)
        output_path = output_dir / f"{label}_sam_overlay.png"
        render_overlay(image_path, mask_path, output_path, label=label)
        rendered_paths.append(output_path)
        print(output_path)

    build_triptych(rendered_paths, output_dir / args.triptych_name)
    print(output_dir / args.triptych_name)


if __name__ == "__main__":
    main()
