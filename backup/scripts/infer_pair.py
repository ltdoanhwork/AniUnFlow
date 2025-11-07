import argparse, os
import torch
import cv2
import numpy as np
from utils.config import load_config, override_by_cli
from models.registry import build_model

# --- small utils ---

def write_flo(path, flow):
    # flow: (H,W,2) float32
    with open(path, 'wb') as f:
        np.array([202021.25], dtype=np.float32).tofile(f)
        np.array([flow.shape[1]], dtype=np.int32).tofile(f)
        np.array([flow.shape[0]], dtype=np.int32).tofile(f)
        flow.astype(np.float32).tofile(f)

def flow_to_color(flow):
    # flow: (H,W,2) in pixels
    u, v = flow[...,0], flow[...,1]
    rad = np.sqrt(u*u + v*v)
    ang = np.arctan2(-v, -u) / np.pi
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.float32)
    hsv[...,0] = (ang + 1.0) / 2.0  # [0,1]
    hsv[...,1] = 1.0
    hsv[...,2] = np.clip(rad / (np.percentile(rad, 99.0) + 1e-6), 0, 1)  # normalize by 99th perc
    rgb = cv2.cvtColor((hsv*255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    return rgb

def read_rgb(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='YAML config with model section')
    ap.add_argument('--ckpt', default=None, help='optional: override checkpoint (trainer.ckpt_path or model.ckpt)')
    ap.add_argument('--img1', required=True)
    ap.add_argument('--img2', required=True)
    ap.add_argument('--out_dir', default='runs/outputs_infer')
    ap.add_argument('--prefix', default='pred')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cfg = load_config(args.config)
    # allow quick override of ckpt via CLI
    if args.ckpt:
        # Prefer model.ckpt if model is a wrapper like RAFT; else trainer.ckpt_path
        if 'model' in cfg and 'ckpt' in cfg['model']:
            cfg['model']['ckpt'] = args.ckpt
        else:
            cfg.setdefault('trainer', {})['ckpt_path'] = args.ckpt

    device = torch.device(args.device if torch.cuda.is_available() and args.device=='cuda' else 'cpu')

    model = build_model(cfg).to(device)
    # Load ckpt for trainable baselines
    ckpt_path = cfg.get('trainer', {}).get('ckpt_path')
    if ckpt_path:
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state['state_dict'])
    model.eval()

    im1 = read_rgb(args.img1)
    im2 = read_rgb(args.img2)

    # to tensor [1,3,H,W] in [0,1]
    t1 = torch.from_numpy(im1.transpose(2,0,1)).float().unsqueeze(0)/255.0
    t2 = torch.from_numpy(im2.transpose(2,0,1)).float().unsqueeze(0)/255.0
    t1 = t1.to(device)
    t2 = t2.to(device)

    with torch.no_grad():
        out = model(t1, t2)
        flow = out['flow'][0].detach().cpu().numpy().transpose(1,2,0)  # HWC, 2

    # save .flo and color viz
    flo_path = os.path.join(args.out_dir, f"{args.prefix}.flo")
    png_path = os.path.join(args.out_dir, f"{args.prefix}.png")
    write_flo(flo_path, flow)
    cv2.imwrite(png_path, cv2.cvtColor(flow_to_color(flow), cv2.COLOR_RGB2BGR))
    print({'flo': flo_path, 'viz': png_path})

if __name__ == '__main__':
    main()

"""
python3 -m scripts.infer_pair \
  --config configs/animerun_baseline.yaml \
  --img1 ./data/AnimeRun_v2/train/Frame_Anime/agent_basement2_weapon_approach/color_1/0186.png \
  --img2 ./data/AnimeRun_v2/train/Frame_Anime/agent_basement2_weapon_approach/color_1/0187.png \
  --out_dir outputs_infer --prefix demo
"""