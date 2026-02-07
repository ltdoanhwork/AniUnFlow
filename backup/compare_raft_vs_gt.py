import os
import glob
import argparse
import numpy as np
import cv2
from PIL import Image
import sys
import torch

# RAFT core
sys.path.append("models/RAFT/core")
from raft import RAFT
from utils.utils import InputPadder

# AniUnFlow for flow_viz
sys.path.append("/home/serverai/ltdoanh/AniUnFlow")
from utils.flow_viz import flow_to_image, compute_flow_magnitude_radmax

DEVICE = "cuda"


def read_flo_file(filename):
    """Read .flo optical flow file in Middlebury format."""
    with open(filename, "rb") as f:
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


def load_image_tensor(path):
    img = Image.open(path).convert("RGB")
    img_np = np.array(img).astype(np.uint8)
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).float()[None].to(DEVICE)
    return img_np, img_t


def build_triplet_frame(img_rgb, flow_gt, flow_pred, rad_max, clip_flow=None):
    """
    img_rgb : HxWx3 uint8 (RGB)
    flow_gt, flow_pred : HxWx2 float32
    rad_max : shared rad_max over GT+Pred
    return : frame_bgr (3H x W x 3 uint8)  [Image; GT; Pred]
    """
    img_bgr = img_rgb[:, :, ::-1]

    gt_bgr = flow_to_image(
        flow_gt,
        clip_flow=clip_flow,
        robust_percentile=None,
        convert_to_bgr=True,
        rad_max=rad_max,
    )

    pred_bgr = flow_to_image(
        flow_pred,
        clip_flow=clip_flow,
        robust_percentile=None,
        convert_to_bgr=True,
        rad_max=rad_max,
    )

    frame_bgr = np.concatenate([img_bgr, gt_bgr, pred_bgr], axis=0).astype(np.uint8)
    return frame_bgr


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    if args.save_frames:
        os.makedirs(args.save_frames, exist_ok=True)

    # 1) Load image list
    img_files = sorted(
        glob.glob(os.path.join(args.img_dir, "*.png"))
        + glob.glob(os.path.join(args.img_dir, "*.jpg"))
    )
    if len(img_files) < 2:
        raise SystemExit(f"[ERROR] Need at least 2 images in {args.img_dir}, found {len(img_files)}")

    print(f"[INFO] Found {len(img_files)} images.")

    # 2) Load GT list
    gt_files = sorted(glob.glob(args.flow_gt_glob))
    if len(gt_files) == 0:
        raise SystemExit(f"[ERROR] No GT flow files matched: {args.flow_gt_glob}")

    if len(gt_files) != len(img_files) - 1:
        raise SystemExit(
            f"[ERROR] Mismatch: num_images={len(img_files)}, num_gt_flows={len(gt_files)}. "
            f"Expected num_gt_flows = num_images - 1."
        )

    print(f"[INFO] Found {len(gt_files)} GT flows.")

    # 3) Build RAFT model
    raft_args = argparse.Namespace(
        small=args.small,
        mixed_precision=args.mixed_precision,
        alternate_corr=args.alternate_corr,
    )
    model = torch.nn.DataParallel(RAFT(raft_args))
    model.load_state_dict(torch.load(args.model))
    model = model.module
    model.to(DEVICE)
    model.eval()

    # 4) Pass 1: run RAFT on all pairs, collect GT+Pred for rad_max
    print("[INFO] Running RAFT on all pairs and collecting GT+Pred flows...")
    all_flows_for_rad = []
    imgs_rgb = []
    flows_gt = []
    flows_pred = []

    with torch.no_grad():
        for i in range(len(img_files) - 1):
            img1_path = img_files[i]
            img2_path = img_files[i + 1]
            gt_path = gt_files[i]

            img1_rgb, img1_t = load_image_tensor(img1_path)
            img2_rgb, img2_t = load_image_tensor(img2_path)

            padder = InputPadder(img1_t.shape)
            img1_p, img2_p = padder.pad(img1_t, img2_t)

            flow_low, flow_up = model(img1_p, img2_p, iters=20, test_mode=True)
            flow_pred = flow_up[0].permute(1, 2, 0).detach().cpu().numpy()

            flow_gt = load_flow_any(gt_path)

            # Resize GT to match image if needed
            H_img, W_img = img1_rgb.shape[:2]
            H_gt, W_gt = flow_gt.shape[:2]
            if (H_gt, W_gt) != (H_img, W_img):
                flow_gt = cv2.resize(flow_gt, (W_img, H_img), interpolation=cv2.INTER_NEAREST)

            imgs_rgb.append(img1_rgb)
            flows_gt.append(flow_gt)
            flows_pred.append(flow_pred)

            all_flows_for_rad.append(flow_gt)
            all_flows_for_rad.append(flow_pred)

    # 5) Compute shared rad_max over GT+Pred
    print("[INFO] Computing shared rad_max over GT + RAFT Pred...")
    rad_max = compute_flow_magnitude_radmax(
        all_flows_for_rad,
        robust_percentile=args.shared_radmax_percentile,
    )
    print(f"[INFO] Shared rad_max (p{args.shared_radmax_percentile}) = {rad_max:.4f}")

    # 6) Write video + frames
    out_video = os.path.join(args.out_dir, args.out_video)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Build first frame to get size
    first_frame = build_triplet_frame(
        imgs_rgb[0], flows_gt[0], flows_pred[0],
        rad_max=rad_max,
        clip_flow=args.clip_flow,
    )
    H_vis, W_vis = first_frame.shape[:2]
    writer = cv2.VideoWriter(out_video, fourcc, args.fps, (W_vis, H_vis))

    writer.write(first_frame)
    if args.save_frames:
        cv2.imwrite(os.path.join(args.save_frames, "frame_000.png"), first_frame)

    print("[INFO] Writing remaining frames...")
    for idx in range(1, len(flows_gt)):
        frame = build_triplet_frame(
            imgs_rgb[idx], flows_gt[idx], flows_pred[idx],
            rad_max=rad_max,
            clip_flow=args.clip_flow,
        )
        if frame.shape[0] != H_vis or frame.shape[1] != W_vis:
            frame = cv2.resize(frame, (W_vis, H_vis), interpolation=cv2.INTER_AREA)

        writer.write(frame)
        if args.save_frames:
            cv2.imwrite(os.path.join(args.save_frames, f"frame_{idx:03d}.png"), frame)

    writer.release()
    print(f"[OK] Saved RAFT vs GT comparison video to: {out_video}")
    if args.save_frames:
        print(f"[OK] Saved per-frame PNG to: {args.save_frames}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare RAFT prediction vs GT flow with shared rad_max over GT+Pred. "
                    "Each frame = [Image; GT; Pred]."
    )
    parser.add_argument("--img_dir", type=str, required=True, help="Directory containing frames (PNG/JPG).")
    parser.add_argument("--flow_gt_glob", type=str, required=True,
                        help="Glob pattern for GT flow files, e.g. '/path/to/gt/*.flo'.")
    parser.add_argument("--model", type=str, required=True, help="RAFT checkpoint path.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for video/frames.")
    parser.add_argument("--out_video", type=str, default="raft_vs_gt.mp4",
                        help="Output video filename (inside out_dir).")
    parser.add_argument("--fps", type=float, default=24.0, help="FPS for output video.")
    parser.add_argument("--save_frames", type=str, default=None,
                        help="If set, save each visualization frame as PNG into this folder.")

    parser.add_argument("--shared_radmax_percentile", type=float, default=95.0,
                        help="Percentile for shared rad_max over GT+Pred (default: 95).")
    parser.add_argument("--clip_flow", type=float, default=None,
                        help="Optional hard clip on flow magnitude (e.g., 50.0).")

    # RAFT model options
    parser.add_argument("--small", action="store_true", help="use small RAFT model")
    parser.add_argument("--mixed_precision", action="store_true", help="use mixed precision")
    parser.add_argument("--alternate_corr", action="store_true", help="use efficient correlation implementation")

    args = parser.parse_args()
    main(args)
