"""
Segment-Aware Losses V3
========================
Enhanced losses based on UnSAMFlow (CVPR 2024) research.

New losses:
1. HomographySmoothnessLoss: Fit homography per segment, penalize residual
2. SegmentMotionConsistency: Low variance within segments
3. BoundarySharpnessLoss: Flow gradients align with SAM boundaries
4. CrossSegmentDiscontinuity: Allow discontinuity at boundaries

Does not modify existing segment_aware_losses.py.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def charbonnier(x: torch.Tensor, eps: float = 1e-3, alpha: float = 0.45) -> torch.Tensor:
    """Robust Charbonnier penalty function."""
    return torch.pow(x * x + eps * eps, alpha)


def compute_flow_gradients(flow: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute flow gradients in x and y directions."""
    fx = flow[:, :, :, 1:] - flow[:, :, :, :-1]  # (B, 2, H, W-1)
    fy = flow[:, :, 1:, :] - flow[:, :, :-1, :]  # (B, 2, H-1, W)
    return fx, fy


class HomographySmoothnessLoss(nn.Module):
    """
    Homography-based Smoothness Loss (UnSAMFlow style).
    
    For each segment:
    1. Fit a homography to the flow within the segment
    2. Compute residual = actual_flow - homography_predicted_flow
    3. Penalize residual
    
    This allows for smooth parametric motion within segments while
    preserving sharp boundaries.
    
    Simplified version: Use affine model instead of full homography.
    """
    
    def __init__(
        self,
        weight: float = 0.1,
        use_affine: bool = True,  # Affine is more stable than homography
        min_segment_pixels: int = 100,  # Minimum pixels to fit model
    ):
        super().__init__()
        self.weight = weight
        self.use_affine = use_affine
        self.min_segment_pixels = min_segment_pixels
    
    def _fit_affine(
        self,
        flow: torch.Tensor,  # (B, 2, H, W)
        mask: torch.Tensor,  # (B, 1, H, W) in [0, 1]
    ) -> torch.Tensor:
        """
        Fit affine flow model to masked region and return residual.
        
        Affine flow: f(x, y) = [a1*x + a2*y + a3, b1*x + b2*y + b3]
        """
        B, C, H, W = flow.shape
        device = flow.device
        dtype = flow.dtype
        
        # Create coordinate grids
        y_coords = torch.arange(H, device=device, dtype=dtype).view(1, 1, H, 1).expand(B, 1, H, W)
        x_coords = torch.arange(W, device=device, dtype=dtype).view(1, 1, 1, W).expand(B, 1, H, W)
        
        # Normalize coordinates to [-1, 1]
        y_norm = (y_coords / (H - 1) - 0.5) * 2
        x_norm = (x_coords / (W - 1) - 0.5) * 2
        
        # Flatten
        mask_flat = mask.view(B, -1)  # (B, HW)
        flow_flat = flow.view(B, 2, -1)  # (B, 2, HW)
        x_flat = x_norm.view(B, -1)  # (B, HW)
        y_flat = y_norm.view(B, -1)  # (B, HW)
        
        residuals = []
        
        for b in range(B):
            m = mask_flat[b]  # (HW,)
            valid_idx = m > 0.5
            n_valid = valid_idx.sum().item()
            
            if n_valid < self.min_segment_pixels:
                # Not enough pixels, skip this segment
                residuals.append(torch.zeros_like(flow[b]))
                continue
            
            # Extract valid pixels
            f = flow_flat[b, :, valid_idx]  # (2, N)
            x = x_flat[b, valid_idx]  # (N,)
            y = y_flat[b, valid_idx]  # (N,)
            ones = torch.ones_like(x)
            
            # Build design matrix: A = [x, y, 1] for each channel
            # f_u = a1*x + a2*y + a3
            # f_v = b1*x + b2*y + b3
            A = torch.stack([x, y, ones], dim=1)  # (N, 3)
            
            # Solve least squares for each flow channel
            try:
                # Use pseudo-inverse for stability
                AtA = A.T @ A  # (3, 3)
                AtA_reg = AtA + 1e-6 * torch.eye(3, device=device, dtype=dtype)
                AtA_inv = torch.linalg.inv(AtA_reg)
                
                # Solve for u channel: params_u = (A^T A)^-1 A^T f_u
                params_u = AtA_inv @ (A.T @ f[0])  # (3,)
                params_v = AtA_inv @ (A.T @ f[1])  # (3,)
                
                # Predict affine flow for all pixels
                x_full = x_flat[b]
                y_full = y_flat[b]
                ones_full = torch.ones_like(x_full)
                
                pred_u = params_u[0] * x_full + params_u[1] * y_full + params_u[2]
                pred_v = params_v[0] * x_full + params_v[1] * y_full + params_v[2]
                
                pred_flow = torch.stack([pred_u, pred_v], dim=0).view(2, H, W)
                
                # Residual
                res = flow[b] - pred_flow
                residuals.append(res)
                
            except Exception:
                # Fallback if solve fails
                residuals.append(torch.zeros_like(flow[b]))
        
        return torch.stack(residuals, dim=0)  # (B, 2, H, W)
    
    def forward(
        self,
        flow: torch.Tensor,           # (B, 2, H, W)
        segment_masks: torch.Tensor,  # (B, S, H, W)
    ) -> torch.Tensor:
        """
        Compute homography smoothness loss.
        
        For each segment, fit affine model and penalize residual.
        """
        B, S, H, W = segment_masks.shape
        
        # Resize masks if needed
        if segment_masks.shape[-2:] != flow.shape[-2:]:
            segment_masks = F.interpolate(
                segment_masks.float(), size=flow.shape[-2:],
                mode='bilinear', align_corners=False
            )
        
        total_loss = 0.0
        total_weight = 0.0
        
        for s in range(S):
            mask = segment_masks[:, s:s+1]  # (B, 1, H, W)
            mask_sum = mask.sum()
            
            if mask_sum < self.min_segment_pixels * B:
                continue
            
            # Fit affine and get residual
            residual = self._fit_affine(flow, mask)
            
            # Penalize residual within mask
            loss_map = charbonnier(residual).mean(dim=1, keepdim=True)  # (B, 1, H, W)
            segment_loss = (loss_map * mask).sum() / mask_sum.clamp(min=1e-6)
            
            total_loss = total_loss + segment_loss * mask_sum
            total_weight = total_weight + mask_sum
        
        if total_weight > 0:
            loss = total_loss / total_weight
        else:
            loss = torch.zeros(1, device=flow.device, dtype=flow.dtype)
        
        return self.weight * loss


class SegmentMotionConsistencyLoss(nn.Module):
    """
    Encourage consistent motion within each segment.
    
    Penalizes high variance of flow vectors within the same segment.
    Similar to segment consistency in V2 but with improvements:
    - Uses robust Charbonnier penalty
    - Better handling of small segments
    """
    
    def __init__(
        self,
        weight: float = 0.05,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.weight = weight
        self.eps = eps
    
    def forward(
        self,
        flow: torch.Tensor,           # (B, 2, H, W)
        segment_masks: torch.Tensor,  # (B, S, H, W)
    ) -> torch.Tensor:
        """Compute motion consistency loss."""
        B, S, H, W = segment_masks.shape
        
        # Resize masks if needed
        if segment_masks.shape[-2:] != flow.shape[-2:]:
            segment_masks = F.interpolate(
                segment_masks.float(), size=flow.shape[-2:],
                mode='bilinear', align_corners=False
            )
        
        total_loss = 0.0
        total_weight = 0.0
        
        for s in range(S):
            mask = segment_masks[:, s:s+1]  # (B, 1, H, W)
            mask_sum = mask.sum(dim=(-2, -1), keepdim=True).clamp(min=self.eps)
            
            # Weighted mean flow
            flow_weighted = flow * mask
            mean_flow = flow_weighted.sum(dim=(-2, -1), keepdim=True) / mask_sum  # (B, 2, 1, 1)
            
            # Variance
            diff = flow - mean_flow
            diff_mag = (diff ** 2).sum(dim=1, keepdim=True).sqrt()  # (B, 1, H, W)
            
            # Weighted variance
            variance = (charbonnier(diff_mag) * mask).sum() / mask_sum.sum().clamp(min=self.eps)
            segment_weight = mask_sum.sum()
            
            total_loss = total_loss + variance * segment_weight
            total_weight = total_weight + segment_weight
        
        if total_weight > self.eps:
            loss = total_loss / total_weight
        else:
            loss = torch.zeros(1, device=flow.device, dtype=flow.dtype)
        
        return self.weight * loss


class BoundarySharpnessLoss(nn.Module):
    """
    Encourage flow gradients to align with segment boundaries.
    
    Where SAM detects a boundary, flow should have high gradients.
    Where SAM detects interior, flow should be smooth.
    """
    
    def __init__(
        self,
        weight: float = 0.05,
        boundary_threshold: float = 0.3,
    ):
        super().__init__()
        self.weight = weight
        self.boundary_threshold = boundary_threshold
    
    def forward(
        self,
        flow: torch.Tensor,           # (B, 2, H, W)
        boundary_map: torch.Tensor,   # (B, 1, H, W) in [0, 1]
    ) -> torch.Tensor:
        """Compute boundary sharpness loss."""
        # Resize boundary if needed
        if boundary_map.shape[-2:] != flow.shape[-2:]:
            boundary_map = F.interpolate(
                boundary_map, size=flow.shape[-2:],
                mode='bilinear', align_corners=False
            )
        
        # Compute flow gradients
        fx, fy = compute_flow_gradients(flow)
        
        # Flow gradient magnitude
        # Pad to match original size
        fx_mag = (fx ** 2).sum(dim=1, keepdim=True).sqrt()  # (B, 1, H, W-1)
        fy_mag = (fy ** 2).sum(dim=1, keepdim=True).sqrt()  # (B, 1, H-1, W)
        
        # Pad gradients
        fx_mag = F.pad(fx_mag, (0, 1, 0, 0))  # (B, 1, H, W)
        fy_mag = F.pad(fy_mag, (0, 0, 0, 1))  # (B, 1, H, W)
        
        flow_grad_mag = (fx_mag + fy_mag) / 2
        
        # Boundary: high gradient is good (no penalty)
        # Interior: high gradient is bad (penalize)
        # Loss = gradient * (1 - boundary)
        interior_mask = (1 - boundary_map).clamp(0, 1)
        
        # Penalize gradients in interior regions
        loss = (charbonnier(flow_grad_mag) * interior_mask).mean()
        
        return self.weight * loss


class CrossSegmentDiscontinuityLoss(nn.Module):
    """
    Allow and encourage flow discontinuity at segment boundaries.
    
    This is the opposite of smoothness: we WANT discontinuity at boundaries.
    Implemented as a negative smoothness penalty at boundaries.
    """
    
    def __init__(
        self,
        weight: float = 0.02,
        min_discontinuity: float = 0.5,  # Minimum desired discontinuity in pixels
    ):
        super().__init__()
        self.weight = weight
        self.min_discontinuity = min_discontinuity
    
    def forward(
        self,
        flow: torch.Tensor,           # (B, 2, H, W)
        boundary_map: torch.Tensor,   # (B, 1, H, W) in [0, 1]
    ) -> torch.Tensor:
        """Compute cross-segment discontinuity loss."""
        # Resize boundary if needed
        if boundary_map.shape[-2:] != flow.shape[-2:]:
            boundary_map = F.interpolate(
                boundary_map, size=flow.shape[-2:],
                mode='bilinear', align_corners=False
            )
        
        # Compute flow gradients
        fx, fy = compute_flow_gradients(flow)
        
        # Gradient magnitude
        fx_mag = (fx ** 2).sum(dim=1, keepdim=True).sqrt()
        fy_mag = (fy ** 2).sum(dim=1, keepdim=True).sqrt()
        
        # Pad
        fx_mag = F.pad(fx_mag, (0, 1, 0, 0))
        fy_mag = F.pad(fy_mag, (0, 0, 0, 1))
        
        flow_grad_mag = (fx_mag + fy_mag) / 2
        
        # At boundaries, encourage gradient >= min_discontinuity
        # Loss = max(0, min_discontinuity - gradient) * boundary
        discontinuity_deficit = F.relu(self.min_discontinuity - flow_grad_mag)
        
        loss = (discontinuity_deficit * boundary_map).mean()
        
        return self.weight * loss


class SegmentAwareLossModuleV3(nn.Module):
    """
    Combined V3 Segment-Aware Loss Module.
    
    Aggregates all V3 losses:
    - HomographySmoothnessLoss
    - SegmentMotionConsistencyLoss
    - BoundarySharpnessLoss
    - CrossSegmentDiscontinuityLoss
    """
    
    def __init__(self, cfg: Dict):
        super().__init__()
        loss_cfg = cfg.get('loss', {})
        
        # Homography Smoothness
        homo_cfg = loss_cfg.get('homography_smooth', {})
        self.use_homography = homo_cfg.get('enabled', True)
        if self.use_homography:
            self.homography_loss = HomographySmoothnessLoss(
                weight=homo_cfg.get('weight', 0.1),
            )
        
        # Motion Consistency
        motion_cfg = loss_cfg.get('segment_motion_consistency', {})
        self.use_motion_consistency = motion_cfg.get('enabled', True)
        if self.use_motion_consistency:
            self.motion_loss = SegmentMotionConsistencyLoss(
                weight=motion_cfg.get('weight', 0.05),
            )
        
        # Boundary Sharpness
        boundary_cfg = loss_cfg.get('boundary_sharpness', {})
        self.use_boundary_sharpness = boundary_cfg.get('enabled', True)
        if self.use_boundary_sharpness:
            self.boundary_loss = BoundarySharpnessLoss(
                weight=boundary_cfg.get('weight', 0.05),
            )
        
        # Cross-Segment Discontinuity
        discont_cfg = loss_cfg.get('cross_segment_discontinuity', {})
        self.use_discontinuity = discont_cfg.get('enabled', False)  # Off by default
        if self.use_discontinuity:
            self.discontinuity_loss = CrossSegmentDiscontinuityLoss(
                weight=discont_cfg.get('weight', 0.02),
            )
    
    def forward(
        self,
        flows: List[torch.Tensor],          # List of (B, 2, H, W)
        segment_masks: torch.Tensor,        # (B, T, S, H, W) or (B, S, H, W)
        boundary_maps: Optional[torch.Tensor] = None,  # (B, T, 1, H, W)
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all V3 losses.
        
        Returns:
            Dictionary with individual losses and total_v3_loss
        """
        device = flows[0].device
        dtype = flows[0].dtype
        
        losses = {}
        total = torch.zeros(1, device=device, dtype=dtype)
        
        T = len(flows)
        
        for k, flow in enumerate(flows):
            # Get masks for this time step
            if segment_masks.dim() == 5:
                masks_t = segment_masks[:, k] if k < segment_masks.shape[1] else segment_masks[:, -1]
            else:
                masks_t = segment_masks
            
            # Get boundary map
            boundary_t = None
            if boundary_maps is not None:
                if boundary_maps.dim() == 5:
                    boundary_t = boundary_maps[:, k] if k < boundary_maps.shape[1] else boundary_maps[:, -1]
                else:
                    boundary_t = boundary_maps
            
            # Homography Smoothness
            if self.use_homography:
                homo_loss = self.homography_loss(flow, masks_t)
                losses[f'homography_smooth_{k}'] = homo_loss
                total = total + homo_loss
            
            # Motion Consistency
            if self.use_motion_consistency:
                motion_loss = self.motion_loss(flow, masks_t)
                losses[f'motion_consistency_{k}'] = motion_loss
                total = total + motion_loss
            
            # Boundary Sharpness
            if self.use_boundary_sharpness and boundary_t is not None:
                boundary_loss = self.boundary_loss(flow, boundary_t)
                losses[f'boundary_sharpness_{k}'] = boundary_loss
                total = total + boundary_loss
            
            # Cross-Segment Discontinuity
            if self.use_discontinuity and boundary_t is not None:
                discont_loss = self.discontinuity_loss(flow, boundary_t)
                losses[f'discontinuity_{k}'] = discont_loss
                total = total + discont_loss
        
        # Average over time steps
        losses['total_v3_loss'] = total / max(T, 1)
        
        return losses


# ============= Builder =============
def build_segment_aware_losses_v3(cfg: Dict) -> SegmentAwareLossModuleV3:
    """Build V3 loss module from config."""
    return SegmentAwareLossModuleV3(cfg)
