"""
SAM-Guided Loss Functions
=========================
Unsupervised losses that leverage SAM segmentation.
Based on UnSAMFlow (CVPR 2024).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


def compute_boundary_map(masks: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    Compute boundary maps from segmentation masks.
    
    Args:
        masks: (B, H, W) integer label map or (B, S, H, W) one-hot
        kernel_size: Dilation/erosion kernel size
        
    Returns:
        boundary: (B, 1, H, W) boundary probability
    """
    if masks.dim() == 3:
        # Integer label map (B, H, W)
        masks = masks.float().unsqueeze(1)  # (B, 1, H, W)
        
    if masks.dim() == 4 and masks.shape[1] == 1:
        # Single channel label map
        labels = masks
        padding = kernel_size // 2
        
        # Boundary = where dilation != erosion
        dilated = F.max_pool2d(labels, kernel_size, stride=1, padding=padding)
        eroded = -F.max_pool2d(-labels, kernel_size, stride=1, padding=padding)
        
        boundary = (dilated != eroded).float()
    else:
        # Multi-channel masks (B, S, H, W)
        B, S, H, W = masks.shape
        padding = kernel_size // 2
        
        # Compute boundary for each segment and combine
        boundaries = []
        for s in range(S):
            seg = masks[:, s:s+1].float()
            dilated = F.max_pool2d(seg, kernel_size, stride=1, padding=padding)
            eroded = -F.max_pool2d(-seg, kernel_size, stride=1, padding=padding)
            boundary = dilated - eroded
            boundaries.append(boundary)
        
        boundary = torch.stack(boundaries, dim=1).max(dim=1, keepdim=True)[0]
        boundary = boundary.clamp(0, 1)
    
    return boundary


class HomographySmoothLoss(nn.Module):
    """
    Homography Smoothness Loss from UnSAMFlow.
    
    Fits affine transformation per segment and penalizes residuals.
    Encourages piece-wise smooth flow within objects.
    """
    
    def __init__(self, min_segment_pixels: int = 100):
        super().__init__()
        self.min_segment_pixels = min_segment_pixels
    
    def fit_affine(
        self,
        flow: torch.Tensor,      # (N, 2) flow vectors
        coords: torch.Tensor,    # (N, 2) pixel coordinates
    ) -> torch.Tensor:
        """Fit affine transformation: flow = A @ coords + b"""
        N = flow.shape[0]
        if N < 6:
            return flow  # Not enough points, return original
        
        # Build linear system: [x, y, 1, 0, 0, 0] and [0, 0, 0, x, y, 1]
        ones = torch.ones(N, 1, device=flow.device)
        zeros = torch.zeros(N, 3, device=flow.device)
        
        A_top = torch.cat([coords, ones, zeros], dim=1)  # (N, 6)
        A_bot = torch.cat([zeros, coords, ones], dim=1)  # (N, 6)
        A = torch.cat([A_top, A_bot], dim=0)  # (2N, 6)
        
        b = torch.cat([flow[:, 0], flow[:, 1]], dim=0)  # (2N,)
        
        # Solve least squares
        try:
            params, _ = torch.lstsq(b.unsqueeze(1), A)
            params = params[:6, 0]  # (6,)
        except:
            return flow
        
        # Compute predicted flow
        pred_u = coords[:, 0] * params[0] + coords[:, 1] * params[1] + params[2]
        pred_v = coords[:, 0] * params[3] + coords[:, 1] * params[4] + params[5]
        pred_flow = torch.stack([pred_u, pred_v], dim=1)
        
        return pred_flow
    
    def forward(
        self,
        flow: torch.Tensor,      # (B, 2, H, W)
        masks: torch.Tensor,     # (B, H, W) integer labels
    ) -> torch.Tensor:
        """
        Compute homography smoothness loss.
        """
        B, _, H, W = flow.shape
        device = flow.device
        
        # Create coordinate grid
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        coords = torch.stack([x_coords, y_coords], dim=-1)  # (H, W, 2)
        
        total_loss = 0.0
        num_segments = 0
        
        for b in range(B):
            mask_b = masks[b]  # (H, W)
            flow_b = flow[b].permute(1, 2, 0)  # (H, W, 2)
            
            for seg_id in mask_b.unique():
                if seg_id == 0:
                    continue  # Skip background
                
                seg_mask = (mask_b == seg_id)
                num_pixels = seg_mask.sum().item()
                
                if num_pixels < self.min_segment_pixels:
                    continue
                
                # Get flow and coords for this segment
                seg_flow = flow_b[seg_mask]  # (N, 2)
                seg_coords = coords[seg_mask]  # (N, 2)
                
                # Fit affine and get residuals
                pred_flow = self.fit_affine(seg_flow, seg_coords)
                residual = (seg_flow - pred_flow).norm(dim=-1)
                
                total_loss = total_loss + residual.mean()
                num_segments += 1
        
        if num_segments > 0:
            return total_loss / num_segments
        else:
            return torch.tensor(0.0, device=device)


class BoundarySharpnessLoss(nn.Module):
    """
    Boundary Sharpness Loss.
    
    Encourages flow gradients to align with segment boundaries.
    """
    
    def __init__(self):
        super().__init__()
        
        # Sobel filters for gradient
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def forward(
        self,
        flow: torch.Tensor,      # (B, 2, H, W)
        boundary: torch.Tensor,  # (B, 1, H, W)
    ) -> torch.Tensor:
        """
        Flow gradients should be high where boundaries are high.
        """
        # Compute flow gradient magnitude
        flow_mag = flow.norm(dim=1, keepdim=True)  # (B, 1, H, W)
        
        grad_x = F.conv2d(flow_mag, self.sobel_x, padding=1)
        grad_y = F.conv2d(flow_mag, self.sobel_y, padding=1)
        flow_grad = (grad_x.abs() + grad_y.abs())  # (B, 1, H, W)
        
        # Normalize
        flow_grad = flow_grad / (flow_grad.max() + 1e-6)
        boundary = boundary / (boundary.max() + 1e-6)
        
        # Loss: where boundary is high, flow_grad should be high
        # Penalize low flow gradient at boundaries
        loss = boundary * (1 - flow_grad)
        
        return loss.mean()


class ObjectVarianceLoss(nn.Module):
    """
    Object Variance Loss.
    
    Penalizes high flow variance within each segment.
    Encourages consistent motion per object.
    """
    
    def __init__(self, min_segment_pixels: int = 50):
        super().__init__()
        self.min_segment_pixels = min_segment_pixels
    
    def forward(
        self,
        flow: torch.Tensor,      # (B, 2, H, W)
        masks: torch.Tensor,     # (B, H, W) integer labels
    ) -> torch.Tensor:
        """
        Compute intra-segment flow variance.
        """
        B, _, H, W = flow.shape
        device = flow.device
        
        total_loss = 0.0
        num_segments = 0
        
        for b in range(B):
            mask_b = masks[b]  # (H, W)
            flow_b = flow[b]  # (2, H, W)
            
            for seg_id in mask_b.unique():
                if seg_id == 0:
                    continue
                
                seg_mask = (mask_b == seg_id)
                num_pixels = seg_mask.sum().item()
                
                if num_pixels < self.min_segment_pixels:
                    continue
                
                # Get flow for this segment
                seg_flow_u = flow_b[0][seg_mask]  # (N,)
                seg_flow_v = flow_b[1][seg_mask]  # (N,)
                
                # Compute variance
                var_u = seg_flow_u.var()
                var_v = seg_flow_v.var()
                
                total_loss = total_loss + var_u + var_v
                num_segments += 1
        
        if num_segments > 0:
            return total_loss / num_segments
        else:
            return torch.tensor(0.0, device=device)


class BoundaryAwareSmoothLoss(nn.Module):
    """
    Boundary-Aware Smoothness Loss.
    
    Standard edge-aware smoothness but suppressed at segment boundaries.
    """
    
    def __init__(self, boundary_suppress: float = 0.1):
        super().__init__()
        self.boundary_suppress = boundary_suppress
    
    def forward(
        self,
        flow: torch.Tensor,      # (B, 2, H, W)
        image: torch.Tensor,     # (B, 3, H, W)
        boundary: torch.Tensor,  # (B, 1, H, W)
    ) -> torch.Tensor:
        """
        Edge-aware smoothness with boundary suppression.
        """
        # Flow gradients
        flow_dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
        flow_dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
        
        # Image gradients (for edge-awareness)
        img_gray = image.mean(dim=1, keepdim=True)
        img_dx = img_gray[:, :, :, 1:] - img_gray[:, :, :, :-1]
        img_dy = img_gray[:, :, 1:, :] - img_gray[:, :, :-1, :]
        
        # Edge weights
        weight_x = torch.exp(-torch.abs(img_dx).mean(dim=1, keepdim=True) * 10)
        weight_y = torch.exp(-torch.abs(img_dy).mean(dim=1, keepdim=True) * 10)
        
        # Boundary suppression
        boundary_x = F.interpolate(boundary, size=weight_x.shape[2:], mode='nearest')
        boundary_y = F.interpolate(boundary, size=weight_y.shape[2:], mode='nearest')
        
        # Suppress smoothness at boundaries
        weight_x = weight_x * (1 - (1 - self.boundary_suppress) * boundary_x)
        weight_y = weight_y * (1 - (1 - self.boundary_suppress) * boundary_y)
        
        # Weighted smoothness
        loss_x = (flow_dx.abs() * weight_x).mean()
        loss_y = (flow_dy.abs() * weight_y).mean()
        
        return loss_x + loss_y


class SAMGuidedLossBundle(nn.Module):
    """
    Bundle of all SAM-guided losses with configurable weights.
    """
    
    def __init__(
        self,
        homography_weight: float = 0.0,
        boundary_sharpness_weight: float = 0.0,
        object_variance_weight: float = 0.0,
        boundary_smooth_weight: float = 0.0,
        boundary_suppress: float = 0.1,
    ):
        super().__init__()
        
        self.weights = {
            'homography': homography_weight,
            'boundary_sharpness': boundary_sharpness_weight,
            'object_variance': object_variance_weight,
            'boundary_smooth': boundary_smooth_weight,
        }
        
        self.homography_loss = HomographySmoothLoss()
        self.boundary_sharpness_loss = BoundarySharpnessLoss()
        self.object_variance_loss = ObjectVarianceLoss()
        self.boundary_smooth_loss = BoundaryAwareSmoothLoss(boundary_suppress)
    
    def forward(
        self,
        flow: torch.Tensor,
        masks: torch.Tensor,
        images: torch.Tensor,
        boundary: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all enabled SAM-guided losses.
        
        Returns:
            losses: Dict of loss name -> value
        """
        losses = {}
        
        # Compute boundary if not provided
        if boundary is None:
            boundary = compute_boundary_map(masks)
        
        if self.weights['homography'] > 0:
            losses['homography'] = self.weights['homography'] * self.homography_loss(flow, masks)
        
        if self.weights['boundary_sharpness'] > 0:
            losses['boundary_sharpness'] = self.weights['boundary_sharpness'] * self.boundary_sharpness_loss(flow, boundary)
        
        if self.weights['object_variance'] > 0:
            losses['object_variance'] = self.weights['object_variance'] * self.object_variance_loss(flow, masks)
        
        if self.weights['boundary_smooth'] > 0:
            losses['boundary_smooth'] = self.weights['boundary_smooth'] * self.boundary_smooth_loss(flow, images, boundary)
        
        return losses
