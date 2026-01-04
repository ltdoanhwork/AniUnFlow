import sys
sys.path.append('models/RAFT/core')  # for RAFT
# Add AniUnFlow root for shared flow_viz
sys.path.append('/home/serverai/ltdoanh/AniUnFlow')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils.utils import InputPadder

# Import flow_viz from AniUnFlow
from utils.flow_viz import flow_to_image as aft_flow_to_image, compute_flow_magnitude_radmax

DEVICE = 'cuda'


def load_image(imfile):
    """Load image and convert to torch tensor [1,3,H,W] on GPU."""
    img = Image.open(imfile).convert('RGB')
    img = np.array(img).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def colorize_flow(flow_np, clip_flow, robust_percentile, rad_max):
    """
    Colorize flow using AniUnFlow flow_to_image.
    - flow_np: H×W×2 float32
    - If rad_max is not None -> use global rad_max (shared across sequence).
    - Else -> use per-frame robust_percentile.
    Returns: H×W×3 uint8 BGR
    """
    if rad_max is not None:
        img_bgr = aft_flow_to_image(
            flow_np,
            clip_flow=clip_flow,
            robust_percentile=None,     # disable internal percentile
            convert_to_bgr=True,
            rad_max=rad_max
        )
    else:
        img_bgr = aft_flow_to_image(
            flow_np,
            clip_flow=clip_flow,
            robust_percentile=robust_percentile,
            convert_to_bgr=True,
            rad_max=None
        )
    return img_bgr


def make_vis_frame_from_numpy(img_np, flow_np, robust_percentile, clip_flow, rad_max):
    """
    - img_np: H×W×3 uint8, RGB
    - flow_np: H×W×2 float32 (u,v)
    return: H_vis×W×3 uint8, BGR  (for OpenCV VideoWriter)
    """
    # Convert image: RGB -> BGR
    img_bgr = img_np[:, :, ::-1]

    # Colorize flow
    flow_bgr = colorize_flow(
        flow_np,
        clip_flow=clip_flow,
        robust_percentile=robust_percentile,
        rad_max=rad_max
    )

    # Stack vertically: [image; flow]
    vis_bgr = np.concatenate([img_bgr, flow_bgr], axis=0).astype(np.uint8)
    return vis_bgr


def make_vis_frame_from_torch(img, flo, robust_percentile, clip_flow, rad_max):
    """
    - img: torch [1,3,H,W], float
    - flo: torch [1,2,H,W], float
    return: H_vis×W×3 uint8, BGR
    """
    img_np = img[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    flow_np = flo[0].permute(1, 2, 0).detach().cpu().numpy()
    return make_vis_frame_from_numpy(img_np, flow_np, robust_percentile, clip_flow, rad_max)


# ---------- load flow from different formats ----------
def load_flow_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.flo':
        return load_flo(path)
    elif ext == '.npy':
        return np.load(path).astype(np.float32)
    elif ext == '.exr':
        return load_exr_flow(path)
    else:
        raise ValueError(f"Unsupported flow format: {ext}")


def load_flo(path):
    with open(path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)[0]
        if magic != 202021.25:
            raise RuntimeError(f"Invalid .flo file magic number in {path}")
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2 * w * h)
    flow = data.reshape(h, w, 2)
    return flow


def load_exr_flow(path):
    import OpenEXR, Imath

    exr = OpenEXR.InputFile(path)
    dw = exr.header()['dataWindow']
    h = dw.max.y - dw.min.y + 1
    w = dw.max.x - dw.min.x + 1

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    u = np.frombuffer(exr.channel('R', FLOAT), dtype=np.float32).reshape(h, w)
    v = np.frombuffer(exr.channel('G', FLOAT), dtype=np.float32).reshape(h, w)
    flow = np.stack([u, v], axis=-1)
    return flow


def demo(args):
    os.makedirs('outputs', exist_ok=True)

    use_precomputed_flow = args.flow_glob is not None

    if not use_precomputed_flow:
        # RAFT inference mode
        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(args.model))
        model = model.module
        model.to(DEVICE)
        model.eval()
    else:
        model = None
        if args.model is None:
            print("[INFO] Using precomputed flows, RAFT model will not be loaded.")

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        images = sorted(images)

        if len(images) < 1:
            raise RuntimeError(f"Found {len(images)} image(s) in {args.path}. Need at least 1 frame.")

        # ------------------ PRECOMPUTED FLOW MODE ------------------
        if use_precomputed_flow:
            flow_files = sorted(glob.glob(args.flow_glob))
            if len(flow_files) == 0:
                raise RuntimeError(f"No flow files found with pattern: {args.flow_glob}")

            if len(flow_files) != len(images) - 1:
                raise RuntimeError(
                    f"Mismatch: found {len(images)} images but {len(flow_files)} flow files. "
                    f"Expected len(flow) = len(images) - 1."
                )

            # Load all flows if we need shared rad_max
            flows = []
            if args.shared_radmax:
                print("[INFO] Loading all flows to compute shared rad_max...")
                for fp in flow_files:
                    flows.append(load_flow_any(fp))
                rad_max = compute_flow_magnitude_radmax(
                    flows,
                    robust_percentile=args.shared_radmax_percentile
                )
                print(f"[INFO] Shared rad_max (p{args.shared_radmax_percentile}) = {rad_max:.4f}")
            else:
                rad_max = None   # per-frame percentile

            # First frame: img[0] + flow[0]
            img0 = Image.open(images[0]).convert('RGB')
            img0_np = np.array(img0).astype(np.uint8)
            flow0 = flows[0] if args.shared_radmax else load_flow_any(flow_files[0])

            first_frame_bgr = make_vis_frame_from_numpy(
                img0_np, flow0,
                robust_percentile=args.robust_percentile,
                clip_flow=args.clip_flow,
                rad_max=rad_max
            )
            H_vis, W_vis = first_frame_bgr.shape[:2]

            out_path = args.out if args.out else os.path.join('outputs', 'flow_vis.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(out_path, fourcc, args.fps, (W_vis, H_vis))

            writer.write(first_frame_bgr)
            if args.save_frames:
                os.makedirs(args.save_frames, exist_ok=True)
                cv2.imwrite(os.path.join(args.save_frames, f"frame_000.png"), first_frame_bgr)

            # Remaining frames
            for idx in range(1, len(flow_files)):
                img = Image.open(images[idx]).convert('RGB')
                img_np = np.array(img).astype(np.uint8)
                flow_np = flows[idx] if args.shared_radmax else load_flow_any(flow_files[idx])

                frame_bgr = make_vis_frame_from_numpy(
                    img_np, flow_np,
                    robust_percentile=args.robust_percentile,
                    clip_flow=args.clip_flow,
                    rad_max=rad_max
                )
                if frame_bgr.shape[0] != H_vis or frame_bgr.shape[1] != W_vis:
                    frame_bgr = cv2.resize(frame_bgr, (W_vis, H_vis), interpolation=cv2.INTER_AREA)
                writer.write(frame_bgr)

                if args.save_frames:
                    cv2.imwrite(os.path.join(args.save_frames, f"frame_{idx:03d}.png"), frame_bgr)

            writer.release()
            print(f"[OK] Saved optical-flow video (precomputed) to: {out_path}")
            if args.save_frames:
                print(f"[OK] Saved per-frame visualizations to: {args.save_frames}")
            return

        # ------------------ RAFT INFERENCE MODE ------------------
        if len(images) < 2:
            raise RuntimeError(f"Found {len(images)} image(s) in {args.path}. Need at least 2 consecutive frames.")

        out_path = args.out if args.out else os.path.join('outputs', 'flow_vis.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        if args.shared_radmax:
            # -------- Pass 1: run RAFT over all pairs and collect flows --------
            print("[INFO] Running RAFT over all pairs to collect flows for shared rad_max...")
            flows = []
            imgs_np = []

            for imfile1, imfile2 in zip(images[:-1], images[1:]):
                image1 = load_image(imfile1)
                image2 = load_image(imfile2)

                padder = InputPadder(image1.shape)
                image1_p, image2_p = padder.pad(image1, image2)

                flow_low, flow_up = model(image1_p, image2_p, iters=20, test_mode=True)
                flow_np = flow_up[0].permute(1, 2, 0).detach().cpu().numpy()

                img_np = image1_p[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)

                flows.append(flow_np)
                imgs_np.append(img_np)

            # Compute shared rad_max
            print("[INFO] Computing shared rad_max from predicted flows...")
            rad_max = compute_flow_magnitude_radmax(
                flows,
                robust_percentile=args.shared_radmax_percentile
            )
            print(f"[INFO] Shared rad_max (p{args.shared_radmax_percentile}) = {rad_max:.4f}")

            # -------- Pass 2: write video --------
            first_frame_bgr = make_vis_frame_from_numpy(
                imgs_np[0], flows[0],
                robust_percentile=args.robust_percentile,
                clip_flow=args.clip_flow,
                rad_max=rad_max
            )
            H_vis, W_vis = first_frame_bgr.shape[:2]
            writer = cv2.VideoWriter(out_path, fourcc, args.fps, (W_vis, H_vis))

            writer.write(first_frame_bgr)
            if args.save_frames:
                os.makedirs(args.save_frames, exist_ok=True)
                cv2.imwrite(os.path.join(args.save_frames, f"frame_000.png"), first_frame_bgr)

            for idx in range(1, len(flows)):
                frame_bgr = make_vis_frame_from_numpy(
                    imgs_np[idx], flows[idx],
                    robust_percentile=args.robust_percentile,
                    clip_flow=args.clip_flow,
                    rad_max=rad_max
                )
                writer.write(frame_bgr)
                if args.save_frames:
                    cv2.imwrite(os.path.join(args.save_frames, f"frame_{idx:03d}.png"), frame_bgr)

            writer.release()
            print(f"[OK] Saved optical-flow video (RAFT, shared rad_max) to: {out_path}")
            if args.save_frames:
                print(f"[OK] Saved per-frame visualizations to: {args.save_frames}")
        else:
            # -------- Original streaming mode: per-frame percentile --------
            print("[INFO] Using per-frame robust_percentile rad_max (no shared_radmax).")

            # First pair
            image1_first = load_image(images[0])
            image2_first = load_image(images[1])
            padder_first = InputPadder(image1_first.shape)
            image1_p, image2_p = padder_first.pad(image1_first, image2_first)

            flow_low, flow_up = model(image1_p, image2_p, iters=20, test_mode=True)
            first_frame_bgr = make_vis_frame_from_torch(
                image1_p, flow_up,
                robust_percentile=args.robust_percentile,
                clip_flow=args.clip_flow,
                rad_max=None
            )
            H_vis, W_vis = first_frame_bgr.shape[:2]
            writer = cv2.VideoWriter(out_path, fourcc, args.fps, (W_vis, H_vis))

            writer.write(first_frame_bgr)
            if args.save_frames:
                os.makedirs(args.save_frames, exist_ok=True)
                cv2.imwrite(os.path.join(args.save_frames, f"frame_000.png"), first_frame_bgr)

            # Remaining pairs
            for idx, (imfile1, imfile2) in enumerate(zip(images[1:-1], images[2:]), start=1):
                image1 = load_image(imfile1)
                image2 = load_image(imfile2)

                padder = InputPadder(image1.shape)
                image1_p, image2_p = padder.pad(image1, image2)

                flow_low, flow_up = model(image1_p, image2_p, iters=20, test_mode=True)
                frame_bgr = make_vis_frame_from_torch(
                    image1_p, flow_up,
                    robust_percentile=args.robust_percentile,
                    clip_flow=args.clip_flow,
                    rad_max=None
                )
                if frame_bgr.shape[0] != H_vis or frame_bgr.shape[1] != W_vis:
                    frame_bgr = cv2.resize(frame_bgr, (W_vis, H_vis), interpolation=cv2.INTER_AREA)
                writer.write(frame_bgr)

                if args.save_frames:
                    cv2.imwrite(os.path.join(args.save_frames, f"frame_{idx:03d}.png"), frame_bgr)

            writer.release()
            print(f"[OK] Saved optical-flow video (RAFT, per-frame rad_max) to: {out_path}")
            if args.save_frames:
                print(f"[OK] Saved per-frame visualizations to: {args.save_frames}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=False, help="restore checkpoint (required if not using precomputed flow)")
    parser.add_argument('--path', required=True, help="dataset folder containing frames")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficient correlation implementation')

    # Video export
    parser.add_argument('--out', type=str, default=None, help='output video path, e.g., outputs/flow_vis.mp4')
    parser.add_argument('--fps', type=float, default=24.0, help='video FPS')

    # Options for precomputed flows
    parser.add_argument(
        '--flow_glob',
        type=str,
        default=None,
        help='Glob pattern for precomputed flows (e.g., "/path/to/flows/*.flo"). '
             'If set, code will visualize these flows instead of running RAFT.'
    )
    parser.add_argument(
        '--save_frames',
        type=str,
        default=None,
        help='Optional folder to save each visualization frame as PNG.'
    )

    # Visualization hyper-params (per-frame mode)
    parser.add_argument(
        '--robust_percentile',
        type=float,
        default=95.0,
        help='Percentile for per-frame rad_max in flow_to_image (default: 95).'
    )
    parser.add_argument(
        '--clip_flow',
        type=float,
        default=None,
        help='Optional hard clip for flow magnitude (e.g., 50.0).'
    )

    # Shared rad_max across sequence
    parser.add_argument(
        '--shared_radmax',
        action='store_true',
        help='Use a single rad_max (computed across the whole sequence) for all frames.'
    )
    parser.add_argument(
        '--shared_radmax_percentile',
        type=float,
        default=95.0,
        help='Percentile used to compute shared rad_max over the sequence (default: 95).'
    )

    args = parser.parse_args()

    if args.flow_glob is None and args.model is None:
        raise RuntimeError("You must provide --model when not using --flow_glob (precomputed flows).")

    demo(args)


"""
python3 demo.py \
  --path /home/serverai/ltdoanh/AniUnFlow/demo \
  --flow_glob "/home/serverai/ltdoanh/AniUnFlow/demo/*.flo" \
  --out runs/outputs_infer/flow_vis_gt.mp4 \
  --fps 24 \
  --save_frames runs/outputs_infer/frames_gt \
  --robust_percentile 95

  
  python3 demo.py \
  --path /home/serverai/ltdoanh/AniUnFlow/demo \
  --flow_glob "/home/serverai/ltdoanh/AniUnFlow/demo/*.flo" \
  --out runs/outputs_infer/flow_vis_gt_shared.mp4 \
  --fps 10 \
  --save_frames runs/outputs_infer/frames_gt_shared \
  --shared_radmax \
  --shared_radmax_percentile 65

  python3 demo.py \
  --model /home/serverai/ltdoanh/AniUnFlow/data/AnimeRun/flow/checkpoints/5000_raft-ft-sintel.pth \
  --path /home/serverai/ltdoanh/AniUnFlow/demo \
  --out runs/outputs_infer/flow_vis_shared.mp4 \
  --fps 24 \
  --save_frames runs/outputs_infer/frames_raft_shared \
  --shared_radmax \
  --shared_radmax_percentile 95

"""