# file: losses/segment_aware_losses.py
"""
Segment-Aware Unsupervised Optical Flow Losses
===============================================
Components:
1. SegmentConsistencyFlowLoss - Encourages flow coherence within segments
2. BoundaryAwareSmoothnessLoss - Strong smoothness inside, weak across boundaries
3. TemporalMemoryRegularization - Temporal consistency across frames

All losses are independently weighted and toggleable via config.
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def charbonnier(x: torch.Tensor, eps: float = 1e-3, alpha: float = 0.45) -> torch.Tensor:
    """Robust Charbonnier penalty function."""
    return torch.pow(x * x + eps * eps, alpha)


def compute_segment_boundary(masks: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    Compute segment boundaries from segment masks.
    
    Args:
        masks: (B, S, H, W) segment masks (soft or hard)
        kernel_size: dilation kernel size for boundary detection
    
    Returns:
        boundary: (B, 1, H, W) boundary map (1 at boundaries, 0 inside)
    """
    B, S, H, W = masks.shape
    # Sum over segments to get occupancy
    occupancy = masks.sum(dim=1, keepdim=True)  # (B, 1, H, W)
    
    # Detect boundaries via morphological gradient
    padding = kernel_size // 2
    dilated = F.max_pool2d(occupancy, kernel_size, stride=1, padding=padding)
    eroded = -F.max_pool2d(-occupancy, kernel_size, stride=1, padding=padding)
    
    # Boundary = dilated - eroded
    boundary = (dilated - eroded).clamp(0, 1)
    
    # Also detect inter-segment boundaries
    for s in range(S):
        m = masks[:, s:s+1]
        m_dilated = F.max_pool2d(m, kernel_size, stride=1, padding=padding)
        m_eroded = -F.max_pool2d(-m, kernel_size, stride=1, padding=padding)
        boundary = boundary + (m_dilated - m_eroded).clamp(0, 1)
    
    return boundary.clamp(0, 1)


class SegmentConsistencyFlowLoss(nn.Module):
    """
    Segment Consistency Flow Loss
    ==============================
    Encourages flow coherence within the same segment by penalizing
    high variance of flow vectors inside each segment.
    
    Loss = sum_s [ Var(flow | segment_s) * area(segment_s) ]
    
    This is enforced softly (not hard constraints).
    """
    
    def __init__(
        self,
        weight: float = 0.1,
        eps: float = 1e-6,
        use_charbonnier: bool = True,
    ):
        super().__init__()
        self.weight = weight
        self.eps = eps
        self.use_charbonnier = use_charbonnier
    
    def forward(
        self,
        flow: torch.Tensor,           # (B, 2, H, W)
        segment_masks: torch.Tensor,  # (B, S, H, W) soft masks in [0,1]
    ) -> torch.Tensor:
        """
        Compute segment consistency loss.
        
        Args:
            flow: Predicted optical flow (B, 2, H, W)
            segment_masks: Soft segment masks (B, S, H, W), each pixel sums to ~1
        
        Returns:
            Weighted loss scalar
        """
        B, C, H, W = flow.shape
        _, S, _, _ = segment_masks.shape
        
        # Resize masks to flow resolution if needed
        if segment_masks.shape[-2:] != (H, W):
            segment_masks = F.interpolate(
                segment_masks, size=(H, W), mode='bilinear', align_corners=False
            )
        
        total_loss = 0.0
        total_weight = 0.0
        
        for s in range(S):
            mask = segment_masks[:, s:s+1]  # (B, 1, H, W)
            mask_sum = mask.sum(dim=(-2, -1), keepdim=True).clamp(min=self.eps)
            
            # Weighted mean flow within segment
            flow_weighted = flow * mask  # (B, 2, H, W)
            mean_flow = flow_weighted.sum(dim=(-2, -1), keepdim=True) / mask_sum  # (B, 2, 1, 1)
            
            # Variance within segment
            diff = flow - mean_flow  # (B, 2, H, W)
            diff_sq = (diff ** 2).sum(dim=1, keepdim=True)  # (B, 1, H, W)
            
            if self.use_charbonnier:
                variance_map = charbonnier(diff_sq.sqrt())
            else:
                variance_map = diff_sq
            
            # Weight by mask (soft assignment)
            segment_loss = (variance_map * mask).sum() / mask_sum.sum().clamp(min=self.eps)
            segment_weight = mask_sum.sum()
            
            total_loss = total_loss + segment_loss * segment_weight
            total_weight = total_weight + segment_weight
        
        if total_weight > self.eps:
            loss = total_loss / total_weight
        else:
            loss = torch.zeros(1, device=flow.device, dtype=flow.dtype)
        
        return self.weight * loss


class BoundaryAwareSmoothnessLoss(nn.Module):
    """
    Boundary-Aware Smoothness Loss
    ================================
    - Strong smoothness INSIDE segments
    - Weak or no smoothness ACROSS segment boundaries
    
    Uses segment boundary maps from SAM to modulate smoothness weights.
    """
    
    def __init__(
        self,
        weight: float = 0.15,
        edge_alpha: float = 10.0,
        boundary_suppress: float = 0.1,  # Smoothness weight at boundaries
        use_second_order: bool = True,
    ):
        super().__init__()
        self.weight = weight
        self.edge_alpha = edge_alpha
        self.boundary_suppress = boundary_suppress
        self.use_second_order = use_second_order
    
    def _image_gradients(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute image gradients using Sobel filters."""
        sobel_x = torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
            dtype=img.dtype, device=img.device
        ).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3)
        
        C = img.shape[1]
        gx = F.conv2d(img, sobel_x.repeat(C, 1, 1, 1), padding=1, groups=C)
        gy = F.conv2d(img, sobel_y.repeat(C, 1, 1, 1), padding=1, groups=C)
        
        return gx, gy
    
    def forward(
        self,
        flow: torch.Tensor,           # (B, 2, H, W)
        image: torch.Tensor,          # (B, 3, H, W) reference image in [0,1]
        segment_masks: torch.Tensor,  # (B, S, H, W) or None
        boundary_map: Optional[torch.Tensor] = None,  # (B, 1, H, W) pre-computed
    ) -> torch.Tensor:
        """
        Compute boundary-aware smoothness loss.
        
        Args:
            flow: Predicted optical flow (B, 2, H, W)
            image: Reference image for edge-aware weighting (B, 3, H, W)
            segment_masks: Segment masks for boundary computation
            boundary_map: Pre-computed boundary map (optional)
        
        Returns:
            Weighted loss scalar
        """
        B, C, H, W = flow.shape
        
        # Resize image to flow resolution if needed
        if image.shape[-2:] != (H, W):
            image = F.interpolate(image, size=(H, W), mode='bilinear', align_corners=False)
        
        # Compute image gradients for edge-aware weighting
        gx_img, gy_img = self._image_gradients(image)
        img_grad_mag = (gx_img.abs() + gy_img.abs()).mean(dim=1, keepdim=True)  # (B, 1, H, W)
        edge_weights = torch.exp(-self.edge_alpha * img_grad_mag)
        
        # Compute segment boundary weights
        if boundary_map is None and segment_masks is not None:
            if segment_masks.shape[-2:] != (H, W):
                segment_masks = F.interpolate(
                    segment_masks, size=(H, W), mode='bilinear', align_corners=False
                )
            boundary_map = compute_segment_boundary(segment_masks)
        
        if boundary_map is not None:
            # Reduce smoothness at boundaries
            # boundary_map: 1 at boundaries, 0 inside
            # We want: high weight inside (1.0), low weight at boundaries (boundary_suppress)
            boundary_weights = 1.0 - boundary_map * (1.0 - self.boundary_suppress)
        else:
            boundary_weights = torch.ones(B, 1, H, W, device=flow.device, dtype=flow.dtype)
        
        # Combined weights
        weights = edge_weights * boundary_weights
        
        # Flow gradients (first order)
        fx = flow[:, :, :, 1:] - flow[:, :, :, :-1]  # (B, 2, H, W-1)
        fy = flow[:, :, 1:, :] - flow[:, :, :-1, :]  # (B, 2, H-1, W)
        
        # Weighted first-order smoothness
        wx = weights[:, :, :, 1:]
        wy = weights[:, :, 1:, :]
        
        loss1 = (charbonnier(fx).mean(dim=1, keepdim=True) * wx).mean()
        loss1 = loss1 + (charbonnier(fy).mean(dim=1, keepdim=True) * wy).mean()
        
        total_loss = loss1
        
        # Second-order smoothness (Laplacian)
        if self.use_second_order:
            fxx = fx[:, :, :, 1:] - fx[:, :, :, :-1]
            fyy = fy[:, :, 1:, :] - fy[:, :, :-1, :]
            loss2 = charbonnier(fxx).mean() + charbonnier(fyy).mean()
            total_loss = total_loss + 0.5 * loss2
        
        return self.weight * total_loss


class TemporalMemoryRegularization(nn.Module):
    """
    Temporal Memory Regularization Loss
    =====================================
    Encourages consistency of flow predictions across multiple frames
    using AniFlowFormer-T's temporal memory features.
    
    This is optional and used for multi-frame consistency.
    """
    
    def __init__(
        self,
        weight: float = 0.05,
        consistency_type: str = 'l2',  # 'l2', 'cosine', 'charbonnier'
    ):
        super().__init__()
        self.weight = weight
        self.consistency_type = consistency_type
    
    def forward(
        self,
        flows: list,  # List of (B, 2, H, W) for consecutive pairs
        memory_features: Optional[torch.Tensor] = None,  # Optional memory tensor
    ) -> torch.Tensor:
        """
        Compute temporal consistency loss across flow predictions.
        
        Args:
            flows: List of flow tensors for consecutive pairs
            memory_features: Optional temporal memory features
        
        Returns:
            Weighted loss scalar
        """
        if len(flows) < 2:
            return torch.zeros(1, device=flows[0].device, dtype=flows[0].dtype)
        
        total_loss = 0.0
        count = 0
        
        for i in range(len(flows) - 1):
            f_curr = flows[i]      # (B, 2, H, W)
            f_next = flows[i + 1]  # (B, 2, H, W)
            
            # Ensure same resolution
            if f_curr.shape != f_next.shape:
                f_next = F.interpolate(
                    f_next, size=f_curr.shape[-2:],
                    mode='bilinear', align_corners=False
                )
            
            if self.consistency_type == 'l2':
                diff = (f_curr - f_next).pow(2).mean()
            elif self.consistency_type == 'cosine':
                # Cosine similarity between flow vectors
                f_curr_flat = f_curr.flatten(2)  # (B, 2, HW)
                f_next_flat = f_next.flatten(2)
                cos_sim = F.cosine_similarity(f_curr_flat, f_next_flat, dim=1)
                diff = 1.0 - cos_sim.mean()
            else:  # charbonnier
                diff = charbonnier(f_curr - f_next).mean()
            
            total_loss = total_loss + diff
            count += 1
        
        if count > 0:
            total_loss = total_loss / count
        
        return self.weight * total_loss


class SegmentAwareLossModule(nn.Module):
    """
    Combined Segment-Aware Loss Module
    ====================================
    Aggregates all segment-aware losses with configurable weights and toggles.
    
    Usage:
        loss_module = SegmentAwareLossModule(config)
        loss_dict = loss_module(flow, clip, segment_masks, ...)
    """
    
    def __init__(self, cfg: Dict):
        super().__init__()
        
        # Segment Consistency Loss
        seg_cons_cfg = cfg.get('segment_consistency', {})
        self.use_segment_consistency = seg_cons_cfg.get('enabled', True)
        if self.use_segment_consistency:
            self.segment_consistency = SegmentConsistencyFlowLoss(
                weight=seg_cons_cfg.get('weight', 0.1),
            )
        
        # Boundary-Aware Smoothness Loss
        boundary_cfg = cfg.get('boundary_aware_smooth', {})
        self.use_boundary_smooth = boundary_cfg.get('enabled', True)
        if self.use_boundary_smooth:
            self.boundary_smooth = BoundaryAwareSmoothnessLoss(
                weight=boundary_cfg.get('weight', 0.15),
                boundary_suppress=boundary_cfg.get('boundary_suppress', 0.1),
            )
        
        # Temporal Memory Regularization
        temp_cfg = cfg.get('temporal_memory', {})
        self.use_temporal = temp_cfg.get('enabled', False)
        if self.use_temporal:
            self.temporal_reg = TemporalMemoryRegularization(
                weight=temp_cfg.get('weight', 0.05),
            )
    
    def forward(
        self,
        flows: list,                   # List of (B, 2, H, W)
        clip: torch.Tensor,            # (B, T, 3, H, W)
        segment_masks: Optional[torch.Tensor] = None,  # (B, T, S, H, W) or None
        boundary_maps: Optional[torch.Tensor] = None,   # (B, T, 1, H, W) or None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all segment-aware losses.
        
        Returns:
            Dictionary with individual losses and 'total_segment_loss'
        """
        B, T, C, H, W = clip.shape
        device = clip.device
        dtype = clip.dtype
        
        losses = {}
        total = torch.zeros(1, device=device, dtype=dtype)
        
        # Segment Consistency Loss
        if self.use_segment_consistency and segment_masks is not None:
            seg_cons_loss = torch.zeros(1, device=device, dtype=dtype)
            for k, flow in enumerate(flows):
                if k < T - 1:
                    masks_t = segment_masks[:, k] if segment_masks.dim() == 5 else segment_masks
                    seg_cons_loss = seg_cons_loss + self.segment_consistency(flow, masks_t)
            if len(flows) > 0:
                seg_cons_loss = seg_cons_loss / len(flows)
            losses['segment_consistency'] = seg_cons_loss
            total = total + seg_cons_loss
        
        # Boundary-Aware Smoothness Loss
        if self.use_boundary_smooth:
            boundary_loss = torch.zeros(1, device=device, dtype=dtype)
            for k, flow in enumerate(flows):
                if k < T - 1:
                    img = clip[:, k]  # Reference image
                    masks_t = segment_masks[:, k] if (segment_masks is not None and segment_masks.dim() == 5) else segment_masks
                    boundary_k = boundary_maps[:, k] if (boundary_maps is not None and boundary_maps.dim() == 5) else boundary_maps
                    boundary_loss = boundary_loss + self.boundary_smooth(
                        flow, img, masks_t, boundary_k
                    )
            if len(flows) > 0:
                boundary_loss = boundary_loss / len(flows)
            losses['boundary_smooth'] = boundary_loss
            total = total + boundary_loss
        
        # Temporal Memory Regularization
        if self.use_temporal and len(flows) > 1:
            temp_loss = self.temporal_reg(flows)
            losses['temporal_reg'] = temp_loss
            total = total + temp_loss
        
        losses['total_segment_loss'] = total
        return losses


# ============= Factory function for easy instantiation =============
def build_segment_aware_losses(cfg: Dict) -> SegmentAwareLossModule:
    """Build segment-aware loss module from config."""
    return SegmentAwareLossModule(cfg.get('loss', cfg))
