import torch
import torch.nn.functional as F




def make_coords_grid(B, H, W, device, dtype):
    y, x = torch.meshgrid(torch.arange(H, device=device, dtype=dtype),
    torch.arange(W, device=device, dtype=dtype), indexing='ij')
    grid = torch.stack((x, y), dim=0)[None].repeat(B, 1, 1, 1) # (B,2,H,W)
    return grid


def normalize_grid(grid, H, W):
    # grid (B,2,H,W) in pixel coords -> normalized [-1,1] for grid_sample (x,y)
    x = 2.0 * (grid[:,0] / max(W-1,1)) - 1.0
    y = 2.0 * (grid[:,1] / max(H-1,1)) - 1.0
    return torch.stack([x, y], dim=1).permute(0,2,3,1) # (B,H,W,2)


def flow_warp(img, flow):
    # img: (B,3,H,W) or (B,C,H,W)
    # flow: (B,2,H,W) (dx, dy) in pixels mapping img2->img1 (when used as I2 warped to I1)
    B, C, H, W = img.shape
    coords = make_coords_grid(B, H, W, img.device, img.dtype)
    tgt = coords + flow # pixel coords in img2
    grid = normalize_grid(tgt, H, W)
    out = F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=True)
    return out