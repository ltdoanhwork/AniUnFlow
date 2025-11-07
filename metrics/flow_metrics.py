import torch
import torch.nn.functional as F
import numpy as np

def compute_epe(pred, gt=None):
    # pred: (B,2,H,W). If gt None, compute EPE to zeros for monitoring
    if gt is None:
        gt = torch.zeros_like(pred)
    return ((pred - gt).pow(2).sum(1).sqrt()).mean(dim=(1,2)) # (B,)


def concat_mean(np_arrays):
    return float(np.mean(np.concatenate(np_arrays))) if np_arrays else float("nan")

@torch.no_grad()
def compute_metrics(outputs, target):
    H, W = target.shape[-2:]
    flow = outputs["flow"]
    flow = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=False)
    return {"epe": compute_epe(flow, target)}

@torch.no_grad()
def compute_mag(flow_gt):
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    return mag