import argparse, os, csv
import torch
import cv2
import numpy as np
from utils.config import load_config
from models.registry import build_model

from scripts.infer_pair import write_flo, flow_to_color, read_rgb  # reuse helpers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--csv', required=True, help='CSV with columns: img1,img2[,flow]')
    ap.add_argument('--root', default=None, help='optional: root to prepend to CSV paths')
    ap.add_argument('--out_dir', default='outputs_infer_csv')
    ap.add_argument('--max_items', type=int, default=50)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() and args.device=='cuda' else 'cpu')
    model = build_model(cfg).to(device).eval()
    ckpt_path = cfg.get('trainer', {}).get('ckpt_path')
    if ckpt_path:
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state['state_dict'])

    def jp(p):
        return os.path.join(args.root, p) if (args.root and not os.path.isabs(p)) else p

    with open(args.csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            if i >= args.max_items:
                break
            p1, p2 = jp(r['img1']), jp(r['img2'])
            im1, im2 = read_rgb(p1), read_rgb(p2)
            t1 = torch.from_numpy(im1.transpose(2,0,1)).float().unsqueeze(0)/255.0
            t2 = torch.from_numpy(im2.transpose(2,0,1)).float().unsqueeze(0)/255.0
            t1, t2 = t1.to(device), t2.to(device)
            with torch.no_grad():
                out = model(t1, t2)
                flow = out['flow'][0].detach().cpu().numpy().transpose(1,2,0)
            base = os.path.splitext(os.path.basename(p1))[0]
            write_flo(os.path.join(args.out_dir, f'{base}.flo'), flow)
            viz = flow_to_color(flow)
            cv2.imwrite(os.path.join(args.out_dir, f'{base}.png'), cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
            print(f"[OK] {p1} -> {base}.flo/.png")

if __name__ == '__main__':
    main()