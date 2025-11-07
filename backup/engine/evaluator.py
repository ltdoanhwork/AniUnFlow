import torch
import os, json, time
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from dataio.registry import get_dataset_builder
from models.registry import build_model
from metrics.flow_metrics import *
from utils.utils import InputPadder
from tqdm import tqdm

def resize_mask(mask, target_hw, device):
    if mask is None:
        return None
    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask)
    mask = mask.to(device)
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:,0]
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    Ht, Wt = target_hw
    if mask.shape[-2:] != (Ht, Wt):
        mask = F.interpolate(mask.unsqueeze(1).float(), size=(Ht, Wt), mode="nearest").squeeze(1)
    return mask.bool()

class Evaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        build_ds = get_dataset_builder(cfg["dataset"]["name"])
        ds_val = build_ds(cfg, split='val')
        self.loader = DataLoader(ds_val, batch_size=cfg["trainer"]["batch_size"], shuffle=False,
                                 num_workers=cfg["trainer"]["num_workers"])
        self.work_dir = cfg.get("work_dir", "./runs/default")
        os.makedirs(self.work_dir, exist_ok=True)
        self.tb_dir = os.path.join(self.work_dir, "tb_eval")
        os.makedirs(self.tb_dir, exist_ok=True)
        self.writer = SummaryWriter(self.tb_dir)
        self.model = build_model(cfg).to(self.device)
        ckpt = cfg["trainer"].get("ckpt_path")
        if ckpt:
            state = torch.load(ckpt, map_location=self.device)
            self.model.load_state_dict(state["state_dict"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @torch.no_grad()
    def run(self):
        self.model.eval()
        device = self.device
        n = 0
        epe_all_list   = []
        epe_occ_list   = []
        epe_nonocc_list= []
        epe_line_list  = []
        epe_flat_list  = []
        epe_s10_list   = []
        epe_s1050_list = []
        epe_s50_list   = []
        # epe_sum = 0.0
        for batch in tqdm(self.loader, desc="Evaluating"):
            img1 = batch["image1"].to(self.device)
            img2 = batch["image2"].to(self.device)
            flow_gt = batch["flow"].to(self.device)

            padder = InputPadder(img1.shape)     # pad sao cho bội số 8/16
            img1_p, img2_p = padder.pad(img1, img2)
            flow_gt = batch["flow"].to(self.device)

            B, _, H, W = flow_gt.shape
            out = self.model(img1_p, img2_p)
            flow_pred = out["flow"]
            flow_pred = padder.unpad(out["flow"])  # unpad model output
            Hp, Wp = flow_pred.shape[-2:]
            Hg, Wg = flow_gt.shape[-2:]

            if (Hp, Wp) != (Hg, Wg):
                flow_pred = F.interpolate(flow_pred, size=(Hg, Wg), mode='bilinear', align_corners=False)
                # scale theo tỉ lệ kích thước
                flow_pred[:, 0] *= (Wg / Wp)
                flow_pred[:, 1] *= (Hg / Hp)

            # giờ flow_pred và flow_gt cùng shape
            assert flow_pred.shape[-2:] == flow_gt.shape[-2:], f"Shape mismatch: pred={flow_pred.shape}, gt={flow_gt.shape}"


            print(f"flow_pred.shape after unpad: {flow_pred.shape}")
            n += 1

            epe_map = torch.norm(flow_pred - flow_gt, dim=1)         # [B,H,W]
            print(f"epe_map.shape: {epe_map.shape}")
            mag     = torch.norm(flow_gt, dim=1) 

            valid = batch.get("valid", torch.ones((B, H, W), device=device, dtype=torch.bool)).bool()
            occ   = batch.get("occ",   None)  # 1 = non-occluded, 0 = occluded (AnimeRun)
            flat  = batch.get("line",  None)  # >0 = flat, 0 = line
            Ht, Wt = flow_gt.shape[-2:]
            valid = resize_mask(valid, (Ht, Wt), device)
            occ   = resize_mask(occ,   (Ht, Wt), device)
            flat  = resize_mask(flat,  (Ht, Wt), device)

            # print(f"occ: {occ}")
            # print(f"flat: {flat}")
            epe_all_list.append(epe_map[valid].detach().cpu().numpy())

            if occ is not None:
                occ = occ.to(device)
                print(occ.shape)
                print(valid.shape)
                epe_occ    = epe_map[(occ == 0) & valid]
                epe_nonocc = epe_map[(occ == 1) & valid]
                if epe_occ.numel()    > 0: epe_occ_list.append(epe_occ.detach().cpu().numpy())
                if epe_nonocc.numel() > 0: epe_nonocc_list.append(epe_nonocc.detach().cpu().numpy())

            if flat is not None:
                flat = flat.to(device)
                epe_flat = epe_map[(flat > 0) & valid]
                epe_line = epe_map[(flat == 0) & valid]
                if epe_flat.numel() > 0: epe_flat_list.append(epe_flat.detach().cpu().numpy())
                if epe_line.numel() > 0: epe_line_list.append(epe_line.detach().cpu().numpy())

            # motion bins theo |flow_gt|
            epe_s10   = epe_map[(mag <= 10.0) & valid]
            epe_s1050 = epe_map[(mag > 10.0) & (mag <= 50.0) & valid]
            epe_s50   = epe_map[(mag > 50.0) & valid]
            if epe_s10.numel()   > 0: epe_s10_list.append(epe_s10.detach().cpu().numpy())
            if epe_s1050.numel() > 0: epe_s1050_list.append(epe_s1050.detach().cpu().numpy())
            if epe_s50.numel()   > 0: epe_s50_list.append(epe_s50.detach().cpu().numpy())

            # ========================== aggregate & log ==========================
            metrics = {}
            if epe_all_list:
                epe_all_np = np.concatenate(epe_all_list)
                metrics["epe"] = float(np.mean(epe_all_np))
                metrics["1px"] = float(np.mean(epe_all_np < 1))
                metrics["3px"] = float(np.mean(epe_all_np < 3))
                metrics["5px"] = float(np.mean(epe_all_np < 5))
            else:
                metrics["epe"] = metrics["1px"] = metrics["3px"] = metrics["5px"] = float("nan")

            # breakdowns (có thể là NaN nếu mask không có trong loader)
            metrics["epe_occ"]     = concat_mean(epe_occ_list)
            metrics["epe_nonocc"]  = concat_mean(epe_nonocc_list)
            metrics["epe_line"]    = concat_mean(epe_line_list)
            metrics["epe_flat"]    = concat_mean(epe_flat_list)
            metrics["epe_s<10"]    = concat_mean(epe_s10_list)
            metrics["epe_s10-50"]  = concat_mean(epe_s1050_list)
            metrics["epe_s>50"]    = concat_mean(epe_s50_list)

        # ── save: JSON + CSV + TensorBoard ────────────────────────────────────
        ts = time.strftime("%Y%m%d-%H%M%S")
        json_path = os.path.join(self.work_dir, f"metrics_val_{ts}.json")
        csv_path  = os.path.join(self.work_dir, f"metrics_val_{ts}.csv")

        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # simple CSV (k,v per line)
        with open(csv_path, "w") as f:
            f.write("metric,value\n")
            for k, v in metrics.items():
                f.write(f"{k},{v}\n")

        # TensorBoard scalars (1 global step for whole-val pass)
        for k, v in metrics.items():
            if v == v:  # not NaN
                self.writer.add_scalar(f"val/{k}", v, global_step=0)
        self.writer.flush()

        print(f"[Eval] saved metrics to:\n  {json_path}\n  {csv_path}\n  TB dir: {self.tb_dir}")
        print(f"Validation results: {metrics}")
        print(f"Validation results: {metrics}")
        return metrics