import os
import glob
import argparse
import numpy as np
import cv2
from PIL import Image
import sys

# Add AniUnFlow root
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


def load_flow_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".flo":
        return read_flo_file(path)
    elif ext == ".npy":
        return np.load(path).astype(np.float32)
    elif ext == ".exr":
        import OpenEXR, Imath

        exr = OpenEXR.InputFile(path)
        dw = exr.header()["dataWindow"]
        h = dw.max.y - dw.min.y + 1
        w = dw.max.x - dw.min.x + 1

        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        u = np.frombuffer(exr.channel("R", FLOAT), dtype=np.float32).reshape(h, w)
        v = np.frombuffer(exr.channel("G", FLOAT), dtype=np.float32).reshape(h, w)
        flow = np.stack([u, v], axis=-1)
        return flow
    else:
        raise ValueError(f"Unsupported flow format: {ext}")


def build_triplet_frame(img_rgb, flow_gt, flow_pred,
                        rad_max, clip_flow=None):
    """
    img_rgb: HxWx3 uint8 (RGB)
    flow_gt, flow_pred: HxWx2 float32
    rad_max: float (shared)
    return: frame_bgr: (3H)xW x 3 uint8 (BGR)  [Image; GT; Pred]
    """
    # Image BGR
    img_bgr = img_rgb[:, :, ::-1]

    # GT flow -> BGR
    gt_bgr = flow_to_image(
        flow_gt,
        clip_flow=clip_flow,
        robust_percentile=None,   # we already fix rad_max
        convert_to_bgr=True,
        rad_max=rad_max
    )

    # Pred flow -> BGR
    pred_bgr = flow_to_image(
        flow_pred,
        clip_flow=clip_flow,
        robust_percentile=None,
        convert_to_bgr=True,
        rad_max=rad_max
    )

    # Stack vertically: [image; GT; Pred]
    frame_bgr = np.concatenate([img_bgr, gt_bgr, pred_bgr], axis=0).astype(np.uint8)
    return frame_bgr


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Load images
    img_files = sorted(
        glob.glob(os.path.join(args.img_dir, "*.png")) +
        glob.glob(os.path.join(args.img_dir, "*.jpg"))
    )
    if len(img_files) < 2:
        raise SystemExit(f"[ERROR] Need at least 2 images in {args.img_dir}, found {len(img_files)}")

    print(f"[INFO] Found {len(img_files)} images.")

    # 2) Load GT & Pred flow list
    gt_files = sorted(glob.glob(args.flow_gt_glob))
    pred_files = sorted(glob.glob(args.flow_pred_glob))

    if len(gt_files) == 0:
        raise SystemExit(f"[ERROR] No GT flow files matched: {args.flow_gt_glob}")
    if len(pred_files) == 0:
        raise SystemExit(f"[ERROR] No Pred flow files matched: {args.flow_pred_glob}")

    # Expect length = num_images - 1
    if not (len(gt_files) == len(pred_files) == len(img_files) - 1):
        raise SystemExit(
            f"[ERROR] Mismatch: num_images={len(img_files)}, "
            f"num_gt_flows={len(gt_files)}, num_pred_flows={len(pred_files)}. "
            f"Expected num_flows = num_images - 1."
        )

    print(f"[INFO] Found {len(gt_files)} GT flows and {len(pred_files)} Pred flows.")

    # 3) First pass: load all flows to compute shared rad_max over GT+Pred
    all_flows = []
    print("[INFO] Loading all flows to compute shared rad_max over GT + Pred...")
    for fp in gt_files:
        all_flows.append(load_flow_any(fp))
    for fp in pred_files:
        all_flows.append(load_flow_any(fp))

    rad_max = compute_flow_magnitude_radmax(
        all_flows,
        robust_percentile=args.shared_radmax_percentile
    )
    print(f"[INFO] Shared rad_max (p{args.shared_radmax_percentile}) = {rad_max:.4f}")

    # 4) Init VideoWriter after building first frame
    out_video = os.path.join(args.out_dir, args.out_video)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Build first triplet frame (index 0 → image[0] + flow[0])
    img0 = np.array(Image.open(img_files[0]).convert("RGB"), dtype=np.uint8)
    flow_gt0 = load_flow_any(gt_files[0])
    flow_pred0 = load_flow_any(pred_files[0])

    # Safety: resize flows to image size if needed
    H_img, W_img = img0.shape[:2]
    H_gt, W_gt = flow_gt0.shape[:2]
    H_pr, W_pr = flow_pred0.shape[:2]
    if (H_gt, W_gt) != (H_img, W_img):
        flow_gt0 = cv2.resize(flow_gt0, (W_img, H_img), interpolation=cv2.INTER_NEAREST)
    if (H_pr, W_pr) != (H_img, W_img):
        flow_pred0 = cv2.resize(flow_pred0, (W_img, H_img), interpolation=cv2.INTER_NEAREST)

    first_frame = build_triplet_frame(
        img0, flow_gt0, flow_pred0,
        rad_max=rad_max,
        clip_flow=args.clip_flow
    )
    H_vis, W_vis = first_frame.shape[:2]

    writer = cv2.VideoWriter(out_video, fourcc, args.fps, (W_vis, H_vis))
    writer.write(first_frame)

    if args.save_frames:
        os.makedirs(args.save_frames, exist_ok=True)
        cv2.imwrite(os.path.join(args.save_frames, f"frame_000.png"), first_frame)

    # 5) Remaining frames
    print("[INFO] Writing remaining frames...")
    for idx in range(1, len(gt_files)):
        img = np.array(Image.open(img_files[idx]).convert("RGB"), dtype=np.uint8)
        flow_gt = load_flow_any(gt_files[idx])
        flow_pred = load_flow_any(pred_files[idx])

        H_img, W_img = img.shape[:2]
        H_gt, W_gt = flow_gt.shape[:2]
        H_pr, W_pr = flow_pred.shape[:2]
        if (H_gt, W_gt) != (H_img, W_img):
            flow_gt = cv2.resize(flow_gt, (W_img, H_img), interpolation=cv2.INTER_NEAREST)
        if (H_pr, W_pr) != (H_img, W_img):
            flow_pred = cv2.resize(flow_pred, (W_img, H_img), interpolation=cv2.INTER_NEAREST)

        frame = build_triplet_frame(
            img, flow_gt, flow_pred,
            rad_max=rad_max,
            clip_flow=args.clip_flow
        )

        if frame.shape[0] != H_vis or frame.shape[1] != W_vis:
            frame = cv2.resize(frame, (W_vis, H_vis), interpolation=cv2.INTER_AREA)

        writer.write(frame)
        if args.save_frames:
            cv2.imwrite(os.path.join(args.save_frames, f"frame_{idx:03d}.png"), frame)

    writer.release()
    print(f"[OK] Saved pair visualization video to: {out_video}")
    if args.save_frames:
        print(f"[OK] Saved per-frame PNG to: {args.save_frames}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pair visualizer: [Image; GT flow; Pred flow] with shared rad_max over GT+Pred."
    )
    parser.add_argument(
        "--img_dir", type=str, required=True,
        help="Directory containing frames (PNG/JPG)."
    )
    parser.add_argument(
        "--flow_gt_glob", type=str, required=True,
        help="Glob pattern for GT flow files, e.g. '/path/to/gt/*.flo'."
    )
    parser.add_argument(
        "--flow_pred_glob", type=str, required=True,
        help="Glob pattern for Pred flow files, e.g. '/path/to/pred/*.flo'."
    )
    parser.add_argument(
        "--out_dir", type=str, required=True,
        help="Output directory for video/frames."
    )
    parser.add_argument(
        "--out_video", type=str, default="pair_flow_vis.mp4",
        help="Output video filename (inside out_dir)."
    )
    parser.add_argument(
        "--fps", type=float, default=24.0,
        help="FPS for output video."
    )
    parser.add_argument(
        "--save_frames", type=str, default=None,
        help="If set, save each visualization frame as PNG into this folder."
    )
    parser.add_argument(
        "--shared_radmax_percentile", type=float, default=95.0,
        help="Percentile used to compute shared rad_max over all GT+Pred flows."
    )
    parser.add_argument(
        "--clip_flow", type=float, default=None,
        help="Optional hard clip on flow magnitude before color mapping (e.g., 50.0)."
    )

    args = parser.parse_args()
    main(args)

"""
python3 pair_flow_viz.py \
  --img_dir "/home/serverai/ltdoanh/AniUnFlow/data/AnimeRun_v2/train/Frame_Anime/cami_06_03_A(renew_background)/color_1" \
  --flow_gt_glob "/home/serverai/ltdoanh/AniUnFlow/data/AnimeRun_v2/train/Flow/cami_06_03_A(renew_background)/forward/*.flo" \
  --flow_pred_glob "/home/serverai/ltdoanh/AniUnFlow/outputs/aft_eval/pred_flow/*.flo" \
  --out_dir "/home/serverai/ltdoanh/AniUnFlow/outputs/aft_eval/pair_vis" \
  --out_video "pair_gt_pred.mp4" \
  --fps 24 \
  --save_frames "/home/serverai/ltdoanh/AniUnFlow/outputs/aft_eval/pair_vis/frames" \
  --shared_radmax_percentile 95

"""