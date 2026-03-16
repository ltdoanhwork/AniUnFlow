#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageEnhance

from utils.flow_viz import compute_flow_magnitude_radmax, flow_to_image
from utils.frame_utils import readFlow


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


def soften_flow_image(
    image: Image.Image,
    *,
    white_mix: float = 0.55,
    saturation: float = 0.72,
    contrast: float = 0.92,
    brightness: float = 1.0,
    sharpness: float = 1.0,
) -> Image.Image:
    white_mix = max(0.0, min(1.0, white_mix))
    white_bg = Image.new("RGB", image.size, (248, 246, 242))
    image = Image.blend(image, white_bg, white_mix)
    image = ImageEnhance.Color(image).enhance(saturation)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Sharpness(image).enhance(sharpness)
    return image


def render_flow(
    flow_path: Path,
    output_path: Path,
    label: str,
    rad_max: float,
    *,
    soft: bool = False,
    white_mix: float = 0.55,
    saturation: float = 0.72,
    contrast: float = 0.92,
    brightness: float = 1.0,
    sharpness: float = 1.0,
) -> None:
    flow = readFlow(str(flow_path))
    flow_rgb = flow_to_image(flow, rad_max=rad_max)
    image = Image.fromarray(flow_rgb).convert("RGB")
    if soft:
        image = soften_flow_image(
            image,
            white_mix=white_mix,
            saturation=saturation,
            contrast=contrast,
            brightness=brightness,
            sharpness=sharpness,
        )

    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle((18, 16, 128, 52), radius=10, fill=(17, 24, 39, 210))
    draw.text((32, 24), label, fill=(255, 255, 255, 255))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render GT optical flow demo images.")
    parser.add_argument("--output-dir", required=True, help="Directory for rendered flow images.")
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("LABEL", "FLOW"),
        required=True,
        help="One flow item as: LABEL FLOW_PATH",
    )
    parser.add_argument("--triptych-name", default="gt_flow_triptych.png")
    parser.add_argument(
        "--independent-scale",
        action="store_true",
        help="Normalize each flow independently for prettier qualitative visualization.",
    )
    parser.add_argument(
        "--soft",
        action="store_true",
        help="Use a softer pastel-like color wheel for paper figures.",
    )
    parser.add_argument("--white-mix", type=float, default=0.55, help="Blend ratio with an off-white background.")
    parser.add_argument("--saturation", type=float, default=0.72, help="Color saturation multiplier.")
    parser.add_argument("--contrast", type=float, default=0.92, help="Contrast multiplier.")
    parser.add_argument("--brightness", type=float, default=1.0, help="Brightness multiplier.")
    parser.add_argument("--sharpness", type=float, default=1.0, help="Sharpness multiplier.")
    parser.add_argument(
        "--percentile",
        type=float,
        default=95.0,
        help="Robust percentile used to normalize flow magnitude.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    flows = [readFlow(flow_path) for _, flow_path in args.pair]
    shared_rad_max = compute_flow_magnitude_radmax(flows, robust_percentile=args.percentile)

    rendered_paths: list[Path] = []
    for idx, (label, flow_arg) in enumerate(args.pair):
        flow_path = Path(flow_arg)
        output_path = output_dir / f"{label}_gt_flow.png"
        rad_max = (
            compute_flow_magnitude_radmax([flows[idx]], robust_percentile=args.percentile)
            if args.independent_scale
            else shared_rad_max
        )
        render_flow(
            flow_path,
            output_path,
            label=label,
            rad_max=rad_max,
            soft=args.soft,
            white_mix=args.white_mix,
            saturation=args.saturation,
            contrast=args.contrast,
            brightness=args.brightness,
            sharpness=args.sharpness,
        )
        rendered_paths.append(output_path)
        print(output_path)

    build_triptych(rendered_paths, output_dir / args.triptych_name)
    print(output_dir / args.triptych_name)


if __name__ == "__main__":
    main()
