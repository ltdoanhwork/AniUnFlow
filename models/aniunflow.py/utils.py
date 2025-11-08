from __future__ import annotations
import torch
import torch.nn.functional as F
from einops import rearrange


class SSIM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
        self.pool = torch.nn.AvgPool2d(3, 1)

    def forward(self, x, y):
        mu_x = self.pool(x)
        mu_y = self.pool(y)
        sig_x = self.pool(x*x) - mu_x*mu_x
        sig_y = self.pool(y*y) - mu_y*mu_y
        sig_xy = self.pool(x*y) - mu_x*mu_y
        ssim = ((2*mu_x*mu_y + self.C1) * (2*sig_xy + self.C2)) / ((mu_x*mu_x + mu_y*mu_y + self.C1) * (sig_x + sig_y + self.C2) + 1e-8)
        return torch.clamp((1-ssim)/2, 0, 1)


def _meshgrid_xy(B, H, W, device):
    ys, xs = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, device=device),
        torch.linspace(-1.0, 1.0, W, device=device), 
        indexing='ij')
    grid = torch.stack([xs, ys], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
    return grid


def flow_to_norm_grid(flow):
    B, _, H, W = flow.shape
    dx = flow[:, 0:1] * (2.0 / max(W-1, 1))
    dy = flow[:, 1:2] * (2.0 / max(H-1, 1))
    return rearrange(torch.cat([dx, dy], dim=1), 'b c h w -> b h w c')


def warp(img, flow):
    B, C, H, W = img.shape
    base = _meshgrid_xy(B, H, W, img.device)
    grid = base + flow_to_norm_grid(flow)
    return F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=True)


def image_gradients(img):
    dx = img[..., :, 1:] - img[..., :, :-1]
    dy = img[..., 1:, :] - img[..., :-1, :]
    dx = F.pad(dx, (0,1,0,0))
    dy = F.pad(dy, (0,0,0,1))
    return dx, dy