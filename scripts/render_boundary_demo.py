#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


def compute_boundary_map(labels: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    if labels.dim() == 2:
        labels = labels.unsqueeze(0)
    if labels.dim() == 3:
        labels = labels.float().unsqueeze(1)

    padding = kernel_size // 2
    dilated = F.max_pool2d(labels, kernel_size, stride=1, padding=padding)
    eroded = -F.max_pool2d(-labels, kernel_size, stride=1, padding=padding)
    return (dilated != eroded).float()


def load_label_map(mask_path: Path) -> torch.Tensor:
    mask = torch.load(mask_path, map_location="cpu")
    if isinstance(mask, torch.Tensor):
        return mask.to(torch.float32)
    return torch.from_numpy(np.asarray(mask)).to(torch.float32)


def render_boundary_overlay(
    image_path: Path,
    mask_path: Path,
    output_path: Path,
    label: str,
    edge_rgb: tuple[int, int, int] = (255, 64, 64),
    edge_alpha: int = 255,
    line_width: int = 2,
) -> None:
    image = Image.open(image_path).convert("RGBA")
    labels = load_label_map(mask_path)
    boundary = compute_boundary_map(labels)[0, 0]
    boundary_img = Image.fromarray((boundary.numpy() * 255).astype(np.uint8), mode="L")

    edge_layer = Image.new("RGBA", image.size, edge_rgb + (0,))
    edge_layer.putalpha(boundary_img.point(lambda p: edge_alpha if p > 0 else 0))
    over = Image.alpha_composite(image, edge_layer)

    # Slightly thicken the displayed contour for paper readability.
    if line_width > 1:
        thick = boundary_img
        for _ in range(line_width - 1):
            thick = thick.filter(ImageFilter.MaxFilter(3))
        edge_layer = Image.new("RGBA", image.size, edge_rgb + (0,))
        edge_layer.putalpha(thick.point(lambda p: int(edge_alpha * 0.8) if p > 0 else 0))
        over = Image.alpha_composite(image, edge_layer)

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
    parser = argparse.ArgumentParser(description="Render boundary demo images from label maps.")
    parser.add_argument("--output-dir", required=True, help="Directory for boundary overlays.")
    parser.add_argument(
        "--pair",
        nargs=3,
        action="append",
        metavar=("LABEL", "IMAGE", "MASK"),
        required=True,
        help="One boundary item as: LABEL IMAGE_PATH MASK_PATH",
    )
    parser.add_argument("--triptych-name", default="boundary_triptych.png")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    rendered_paths: list[Path] = []
    for label, image_arg, mask_arg in args.pair:
        image_path = Path(image_arg)
        mask_path = Path(mask_arg)
        output_path = output_dir / f"{label}_boundary_overlay.png"
        render_boundary_overlay(image_path, mask_path, output_path, label=label)
        rendered_paths.append(output_path)
        print(output_path)

    build_triptych(rendered_paths, output_dir / args.triptych_name)
    print(output_dir / args.triptych_name)


if __name__ == "__main__":
    from PIL import ImageFilter  # local import to keep dependency list simple

    main()
