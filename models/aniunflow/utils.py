from __future__ import annotations
import torch
import torch.nn.functional as F
from einops import rearrange


class SSIM(torch.nn.Module):
    def __init__(self, kernel_size: int = 3, K1: float = 0.01, K2: float = 0.03, L: float = 1.0):
        """
        L: dynamic range của dữ liệu ảnh. Ảnh đã chuẩn hóa [0,1] => L=1.
        K1, K2: hằng số chuẩn của SSIM (mặc định 0.01, 0.03).
        """
        super().__init__()
        self.C1 = (K1 * L) ** 2
        self.C2 = (K2 * L) ** 2
        # 'same' output size: padding = k//2, và không tính phần pad vào trung bình
        self.pool = torch.nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size // 2,
                                       count_include_pad=False)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x, y: [B, 3, H, W] trong [0,1]
        eps = 1e-6

        mu_x = self.pool(x)
        mu_y = self.pool(y)

        sigma_x = self.pool(x * x) - mu_x * mu_x
        sigma_y = self.pool(y * y) - mu_y * mu_y
        sigma_xy = self.pool(x * y) - mu_x * mu_y

        # đảm bảo dương tính số nhỏ
        sigma_x = torch.clamp(sigma_x, min=0.0)
        sigma_y = torch.clamp(sigma_y, min=0.0)

        num = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        den = (mu_x * mu_x + mu_y * mu_y + self.C1) * (sigma_x + sigma_y + self.C2) + eps
        ssim = num / den  # kỳ vọng trong [-1, 1]

        # trả về "SSIM loss map" trong [0,1], gộp kênh về [B,1,H,W]
        ssim_loss_map = (1.0 - ssim) * 0.5
        ssim_loss_map = torch.clamp(ssim_loss_map, 0.0, 1.0).mean(dim=1, keepdim=True)
        return ssim_loss_map



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