# file: losses/unsup_flow_losses.py
from __future__ import annotations
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


from utils.warp import flow_warp, make_coords_grid, normalize_grid




def charbonnier(x, eps=1e-3, alpha=0.45):
    return torch.pow(x * x + eps * eps, alpha)




def image_gradients(img):
    # img: (B,3,H,W) in [0,1]
    sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=img.dtype, device=img.device).view(1,1,3,3)
    sobel_y = sobel_x.transpose(2,3)
    gx = F.conv2d(img, sobel_x.repeat(img.shape[1],1,1,1), padding=1, groups=img.shape[1])
    gy = F.conv2d(img, sobel_y.repeat(img.shape[1],1,1,1), padding=1, groups=img.shape[1])
    mag = (gx.abs() + gy.abs()).mean(1, keepdim=True) # (B,1,H,W)
    return gx, gy, mag




def ssim(x, y, C1=0.01**2, C2=0.03**2):
    # x,y: (B,3,H,W) in [0,1]
    pad = 1
    mu_x = F.avg_pool2d(x, 3, 1, pad)
    mu_y = F.avg_pool2d(y, 3, 1, pad)
    sigma_x = F.avg_pool2d(x * x, 3, 1, pad) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, 3, 1, pad) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, 3, 1, pad) - mu_x * mu_y
    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    ssim_map = torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1) # 0 (good) .. 1 (bad)
    return ssim_map




def gradient_smoothness(flow, img, alpha=10.0):
    # edge-aware first and second order smoothness
    gx, gy, mag = image_gradients(img)
    wx = torch.exp(-alpha * mag)
    wy = torch.exp(-alpha * mag)
    fx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    fy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    loss1 = (charbonnier(fx).mean(dim=1, keepdim=True) * wx[:,:,:,1:]).mean() + \
    (charbonnier(fy).mean(dim=1, keepdim=True) * wy[:,:,1:,:]).mean()


    # Second order (Laplacian-like)
    fxx = fx[:, :, :, 1:] - fx[:, :, :, :-1]
    fyy = fy[:, :, 1:, :] - fy[:, :, :-1, :]
    loss2 = (charbonnier(fxx).mean() + charbonnier(fyy).mean())
    return loss1, loss2




def fb_occlusion_mask(f12, f21, alpha=0.01, beta=0.5):
    # forward-backward consistency based mask (1 = visible)
    # warp f21 to 1->2 domain then check |f12 + warp(f21)| < alpha*(|f12|+|warp(f21)|)+beta
    f21_w = flow_warp(f21, f12) # (B,2,H,W)
    lhs = (f12 + f21_w).pow(2).sum(1, keepdim=True).sqrt()
    rhs = alpha * (f12.pow(2).sum(1, keepdim=True).sqrt() + f21_w.pow(2).sum(1, keepdim=True).sqrt()) + beta
    vis = (lhs <= rhs).float()
    return vis, f21_w




class UnsupervisedFlowLoss(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        self.w_photo = cfg.get("w_photo", 1.0)
        self.w_ssim = cfg.get("w_ssim", 0.15)
        self.w_smooth1 = cfg.get("w_smooth1", 0.2)
        self.w_smooth2 = cfg.get("w_smooth2", 0.05)
        self.w_fb = cfg.get("w_fb", 0.2)
        self.eps = cfg.get("charbonnier_eps", 1e-3)
        self.fb_alpha = cfg.get("fb_alpha", 0.01)
        self.fb_beta = cfg.get("fb_beta", 0.5)

    def forward(self, im1, im2, f12, extra: Dict | None = None):
        # Warp I2->1 with f12
        I2w = flow_warp(im2, f12)
        l1 = charbonnier((im1 - I2w), self.eps).mean()
        ssim_map = ssim(im1, I2w)
        l_ssim = ssim_map.mean()


        # Smoothness (stronger far from edges)
        l_sm1, l_sm2 = gradient_smoothness(f12, im1)


        # Forward/Backward occlusion-aware photo loss (need f21); if not supplied, create approx by -f12
        if extra is not None and "f21" in extra:
            f21 = extra["f21"]
        else:
        # cheap approx for mask only; does not add its photo term
            f21 = -f12.detach()


        vis12, f21w = fb_occlusion_mask(f12, f21, self.fb_alpha, self.fb_beta)
        # Masked photometric (ignore occlusions)
        l_photo = (charbonnier((im1 - I2w), self.eps) * vis12).sum() / (vis12.sum() + 1e-6)
        l_ssim_m = (ssim_map * vis12).sum() / (vis12.sum() + 1e-6)


        # FB consistency penalty (encourage cycle-consistency on visible)
        l_fb = charbonnier((f12 + f21w), self.eps)
        l_fb = (l_fb * vis12).sum() / (vis12.sum() + 1e-6)


        total = self.w_photo * l_photo + self.w_ssim * l_ssim_m + \
        self.w_smooth1 * l_sm1 + self.w_smooth2 * l_sm2 + \
        self.w_fb * l_fb


        logs = {
        "l_photo": l_photo.detach(),
        "l_ssim": l_ssim_m.detach(),
        "l_sm1": l_sm1.detach(),
        "l_sm2": l_sm2.detach(),
        "l_fb": l_fb.detach(),
        }
        return total, logs