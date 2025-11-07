import sys
sys.path.append('models/RAFT/core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = 'cuda'

def load_image(imfile):
    # Load image and convert to RGB (ignore alpha if exists)
    img = Image.open(imfile).convert('RGB')
    img = np.array(img).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def make_vis_frame(img, flo):
    """
    Create a visualization frame as a numpy BGR image to be written to a video.
    - img: torch tensor [1,3,H,W], float
    - flo: torch tensor [1,2,H,W], float
    Returns: np.ndarray (H_vis, W, 3) in BGR (uint8)
    """
    # Convert tensors to numpy
    img_np = img[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    flow_np = flo[0].permute(1, 2, 0).detach().cpu().numpy()

    # Map flow to RGB (uint8)
    flow_color = flow_viz.flow_to_image(flow_np)  # returns RGB uint8

    # Stack vertically: [image; flow]
    vis_rgb = np.concatenate([img_np, flow_color], axis=0).astype(np.uint8)

    # Convert RGB -> BGR for OpenCV VideoWriter
    vis_bgr = vis_rgb[:, :, ::-1]
    return vis_bgr

def demo(args):
    os.makedirs('outputs', exist_ok=True)

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        images = sorted(images)

        # Sanity check
        if len(images) < 2:
            raise RuntimeError(f"Found {len(images)} image(s) in {args.path}. Need at least 2 consecutive frames.")

        # Prepare first pair to get frame size for the video writer
        image1_first = load_image(images[0])
        image2_first = load_image(images[1])
        padder_first = InputPadder(image1_first.shape)
        image1_p, image2_p = padder_first.pad(image1_first, image2_first)

        # Run one forward pass to know output shape for visualization frame
        flow_low, flow_up = model(image1_p, image2_p, iters=20, test_mode=True)
        first_frame_bgr = make_vis_frame(image1_p, flow_up)
        H_vis, W_vis = first_frame_bgr.shape[:2]

        # Init video writer
        out_path = args.out if args.out else os.path.join('outputs', 'flow_vis.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can switch to 'XVID' for .avi
        writer = cv2.VideoWriter(out_path, fourcc, args.fps, (W_vis, H_vis))

        # Write first frame
        writer.write(first_frame_bgr)

        # Process remaining pairs
        for imfile1, imfile2 in zip(images[1:-1], images[2:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1_p, image2_p = padder.pad(image1, image2)

            flow_low, flow_up = model(image1_p, image2_p, iters=20, test_mode=True)
            frame_bgr = make_vis_frame(image1_p, flow_up)
            # Ensure size consistency (should match, but guard just in case)
            if frame_bgr.shape[0] != H_vis or frame_bgr.shape[1] != W_vis:
                frame_bgr = cv2.resize(frame_bgr, (W_vis, H_vis), interpolation=cv2.INTER_AREA)
            writer.write(frame_bgr)

        writer.release()
        print(f"[OK] Saved optical-flow video to: {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help="restore checkpoint")
    parser.add_argument('--path', required=True, help="dataset folder containing frames")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficient correlation implementation')

    # New args for video export
    parser.add_argument('--out', type=str, default=None, help='output video path, e.g., outputs/flow_vis.mp4')
    parser.add_argument('--fps', type=float, default=24.0, help='video FPS')

    args = parser.parse_args()
    demo(args)
"""

python3 demo.py \
  --model /home/serverai/ltdoanh/optical_flow/data/AnimeRun/flow/checkpoints/15000_raft-ft-sintel.pth \
  --path /home/serverai/ltdoanh/optical_flow/data/AnimeRun_v2/train/contour/agent_indoor4_dodges \
  --out runs/outputs_infer/flow_vis.mp4 \
  --fps 24

"""