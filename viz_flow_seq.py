import os
import glob
import argparse
import numpy as np
import cv2
import sys

# Add project root to sys.path (adjust if needed)
sys.path.append("/home/serverai/ltdoanh/AniUnFlow")

from utils.flow_viz import flow_to_image, compute_flow_magnitude_radmax


def read_flo_file(filename):
    """Read .flo optical flow file in Middlebury format."""
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)[0]
        if magic != 202021.25:
            raise Exception(f"[ERROR] {filename}: Invalid .flo file (wrong magic number)")
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2 * w * h)
        flow = np.resize(data, (h, w, 2))
        return flow


def load_all_flows(flo_files):
    """Load all .flo files into memory as a list of (filename, flow_array)."""
    flows = []
    for fp in flo_files:
        flow = read_flo_file(fp)
        flows.append(flow)
    return flows


def visualize_flows(
    flo_dir: str,
    out_dir: str,
    robust_percentile: float = 95.0,
    clip_flow: float | None = None,
    filename_suffix: str = "_flow",
):
    """
    Convert all .flo files in flo_dir to color images and save to out_dir.

    - flo_dir: directory containing *.flo files.
    - out_dir: output directory for PNG visualizations.
    - robust_percentile: percentile used to compute shared rad_max across the sequence.
    - clip_flow: optional hard clip on flow magnitude (in pixels) before color mapping.
    - filename_suffix: suffix appended to each output PNG filename.
    """
    os.makedirs(out_dir, exist_ok=True)

    flo_files = sorted(glob.glob(os.path.join(flo_dir, "*.flo")))
    print(f"[INFO] Found {len(flo_files)} .flo files in {flo_dir}")
    if not flo_files:
        raise SystemExit("[ERROR] No .flo files found, please check the path.")

    # -------- Pass 1: load all flows & compute shared rad_max --------
    print("[INFO] Loading flows to compute shared rad_max...")
    flows = load_all_flows(flo_files)
    rad_max = compute_flow_magnitude_radmax(flows, robust_percentile=robust_percentile)
    print(f"[INFO] Shared rad_max (p{robust_percentile:.1f}) = {rad_max:.4f}")

    # -------- Pass 2: visualize each flow with fixed rad_max --------
    print("[INFO] Converting flows to color images...")
    for flo_path, flow in zip(flo_files, flows):
        img_color = flow_to_image(
            flow,
            clip_flow=clip_flow,          # e.g., 50.0 if you want to hard-clip, else None
            robust_percentile=None,       # we already computed rad_max across the sequence
            convert_to_bgr=True,          # OpenCV expects BGR
            rad_max=rad_max
        )

        base = os.path.splitext(os.path.basename(flo_path))[0]  # e.g. Image0186
        out_png = os.path.join(out_dir, base + f"{filename_suffix}.png")
        cv2.imwrite(out_png, img_color)
        print("[OK] Saved:", out_png)

    print("[DONE] All flows have been visualized.")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize .flo optical flow files as color images.")
    parser.add_argument(
        "--flo_dir",
        type=str,
        required=True,
        help="Directory containing .flo files."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory to save PNG visualizations."
    )
    parser.add_argument(
        "--robust_percentile",
        type=float,
        default=95.0,
        help="Percentile to compute shared rad_max across the sequence (default: 95)."
    )
    parser.add_argument(
        "--clip_flow",
        type=float,
        default=None,
        help="Optional hard clip on flow magnitude before color mapping (e.g., 50.0). "
             "If None, no hard clipping is applied."
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_flow",
        help="Suffix added to output PNG filenames (default: '_flow')."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    visualize_flows(
        flo_dir=args.flo_dir,
        out_dir=args.out_dir,
        robust_percentile=args.robust_percentile,
        clip_flow=args.clip_flow,
        filename_suffix=args.suffix,
    )
