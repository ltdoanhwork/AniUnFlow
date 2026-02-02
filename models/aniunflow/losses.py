from __future__ import annotations
from typing import Dict, List
import torch
import torch.nn.functional as F

from .utils import SSIM, warp, image_gradients


class UnsupervisedFlowLoss:
    """
    Bidirectional unsupervised flow loss:
    - Uses forward and backward flows for each frame pair.
    - Components:
        * Photometric loss (L1 + SSIM loss) in both directions.
        * Edge-aware smoothness on forward flow.
        * Forward-backward consistency.
        * Flow magnitude regularization (prevents collapse to zero).
    Assumes:
        * clip in [0,1], shape [B,T,3,H,W]
        * flows_fw, flows_bw: lists of [B,2,h,w], len = T-1
    """

    def __init__(
        self,
        alpha_ssim: float = 0.2,
        w_smooth: float = 0.1,
        w_cons: float = 0.05,
        smooth_alpha: float = 10.0,
        # Anti-collapse regularization
        w_mag_reg: float = 0.01,      # Weight for magnitude regularization
        min_flow_mag: float = 0.5,    # Minimum expected average flow magnitude
        use_photo_gradient: bool = True,  # Use photometric gradients as flow prior
    ):
        self.ssim = SSIM()
        self.alpha_ssim = alpha_ssim
        self.w_smooth = w_smooth
        self.w_cons = w_cons
        self.smooth_alpha = smooth_alpha
        
        # Anti-collapse parameters
        self.w_mag_reg = w_mag_reg
        self.min_flow_mag = min_flow_mag
        self.use_photo_gradient = use_photo_gradient


    def _upsample_flow(self, f: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Upsample flow from (h,w) to (H,W) and scale u,v correctly.
        f: [B,2,h,w] in pixels.
        """
        B, C, h, w = f.shape
        f_up = F.interpolate(f, size=(H, W), mode="bilinear", align_corners=True)
        sx = W / float(w)
        sy = H / float(h)
        f_up[:, 0] *= sx
        f_up[:, 1] *= sy
        return f_up

    def _get_occu_mask_bidirection(
        self,
        flow_fw: torch.Tensor,
        flow_bw: torch.Tensor,
        alpha: float = 0.01,
        beta: float = 0.5,
    ) -> torch.Tensor:
        """
        Compute occlusion mask using forward-backward consistency.
        Pixels where |fw + warp(bw, fw)| > alpha * (|fw|^2 + |bw_warp|^2) + beta are occluded.
        Returns: occlusion mask [B,1,H,W] where 1 = occluded, 0 = visible.
        """
        flow_bw_warp = warp(flow_bw, flow_fw)
        flow_diff = flow_fw + flow_bw_warp
        
        mag_sq_fw = (flow_fw ** 2).sum(dim=1, keepdim=True)
        mag_sq_bw = (flow_bw_warp ** 2).sum(dim=1, keepdim=True)
        
        occ_thresh = alpha * (mag_sq_fw + mag_sq_bw) + beta
        occ_mask = ((flow_diff ** 2).sum(dim=1, keepdim=True) > occ_thresh).float()
        
        return occ_mask

    def _photometric_pair(
        self,
        I_ref: torch.Tensor,
        I_tgt: torch.Tensor,
        flow_up: torch.Tensor,
        vis_mask: torch.Tensor = None,
        eps: float = 1e-3,
    ) -> torch.Tensor:
        """
        Photometric loss between I_ref and I_tgt warped by flow_up.
        I_ref, I_tgt: [B,3,H,W] in [0,1]
        flow_up: [B,2,H,W] in pixels
        vis_mask: [B,1,H,W] visibility mask (1=visible, 0=occluded)
        
        Uses:
            L1 Charbonnier + SSIM loss map (already "1-SSIM" in [0,1])
            Only penalizes visible pixels (if vis_mask provided)
        Returns scalar loss.
        """
        I_tgt_w = warp(I_tgt, flow_up)

        # Charbonnier L1
        l1_map = torch.sqrt((I_ref - I_tgt_w) ** 2 + eps**2)  # [B,3,H,W]
        l1_map = l1_map.mean(dim=1, keepdim=True)             # [B,1,H,W]

        # SSIM returns a loss map in [0,1] (0 = perfect match)
        ssim_loss_map = self.ssim(I_ref, I_tgt_w)             # [B,1,H,W]

        photo_map = (1.0 - self.alpha_ssim) * l1_map + self.alpha_ssim * ssim_loss_map
        
        if vis_mask is not None:
            # Normalize by visible area to avoid pushing toward zero flow
            photo = (photo_map * vis_mask).sum() / (vis_mask.sum() + 1e-6)
        else:
            photo = photo_map.mean()
        
        return photo


    def _smoothness_pair(self, flow_up: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        """
        Edge-aware smoothness on a single flow and reference image.
        flow_up: [B,2,H,W]
        img: [B,3,H,W] in [0,1]
        Returns scalar loss (already multiplied by w_smooth).
        """
        # image gradients for edge weights
        gx_img, gy_img = image_gradients(img)
        img_grad_mag = (gx_img.abs() + gy_img.abs()).mean(1, keepdim=True)  # [B,1,H,W]

        # larger gradients -> smaller weights
        weights = torch.exp(-self.smooth_alpha * img_grad_mag)

        # flow gradients
        dx_f, dy_f = image_gradients(flow_up)  # [B,2,H,W] each
        flow_grad_mag = dx_f.abs() + dy_f.abs()   # [B,2,H,W]
        flow_grad_mag = flow_grad_mag.mean(1, keepdim=True)  # [B,1,H,W]

        smooth = (weights * flow_grad_mag).mean()
        return self.w_smooth * smooth

    def _magnitude_regularization(
        self,
        flow_up: torch.Tensor,
        I1: torch.Tensor,
        I2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Flow magnitude regularization to prevent collapse to zero.
        
        Strategy: Penalize when flow magnitude is significantly below what
        photometric differences suggest it should be.
        
        flow_up: [B,2,H,W] in pixels
        I1, I2: [B,3,H,W] reference and target images
        Returns scalar loss (already multiplied by w_mag_reg).
        """
        if self.w_mag_reg <= 0:
            return torch.tensor(0.0, device=flow_up.device)
        
        # Compute actual flow magnitude
        flow_mag = torch.norm(flow_up, dim=1, keepdim=True)  # [B,1,H,W]
        mean_flow_mag = flow_mag.mean()
        
        if self.use_photo_gradient:
            # Use image difference as a soft prior for where flow should exist
            # High difference = likely motion = should have flow
            img_diff = (I1 - I2).abs().mean(dim=1, keepdim=True)  # [B,1,H,W]
            
            # Expected flow magnitude scales with image difference
            # (rough heuristic: 10 pixels of flow causes ~0.1 intensity change)
            expected_mag = img_diff * 100.0  # Scale factor
            expected_mag = expected_mag.clamp(min=self.min_flow_mag, max=50.0)
            
            # Penalize if flow is much less than expected
            mag_deficit = F.relu(expected_mag - flow_mag)
            loss = mag_deficit.mean()
        else:
            # Simple minimum magnitude penalty
            # Penalize if average flow magnitude is below threshold
            loss = F.relu(self.min_flow_mag - mean_flow_mag)
        
        return self.w_mag_reg * loss

    def _fb_consistency_pair(
        self,
        flow_fw_up: torch.Tensor,
        flow_bw_up: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward-backward consistency:
            F_fw(x) + F_bw(x + F_fw(x)) ~ 0
        flow_fw_up, flow_bw_up: [B,2,H,W] in pixels
        Returns scalar loss (already multiplied by w_cons).
        """
        flow_bw_warp = warp(flow_bw_up, flow_fw_up)
        fb = flow_fw_up + flow_bw_warp
        return self.w_cons * fb.abs().mean()

    # ---------- main bidirectional unsupervised loss ---------- #
    def unsup_bidirectional(
        self,
        clip: torch.Tensor,              # [B,T,3,H,W]
        flows_fw: List[torch.Tensor],    # len = T-1, each [B,2,h,w]
        flows_bw: List[torch.Tensor],    # len = T-1, each [B,2,h,w]
        use_occ_mask: bool = True,       # Whether to use occlusion-aware masking
    ) -> Dict[str, torch.Tensor]:
        """
        Compute bidirectional unsupervised loss for a clip:
        - Forward & backward photometric losses with occlusion masking.
        - Edge-aware smoothness on forward flows.
        - Forward-backward consistency.
        Returns a dict with keys: 'photo', 'smooth', 'cons', 'total'.
        """
        B, T, C, H, W = clip.shape
        assert len(flows_fw) == len(flows_bw) == T - 1, "flows length must be T-1"

        total_photo = 0.0
        total_smooth = 0.0
        total_cons = 0.0
        total_mag_reg = 0.0
        count = 0

        for k in range(T - 1):
            I1 = clip[:, k]     # [B,3,H,W]
            I2 = clip[:, k + 1]

            F_fw = flows_fw[k]  # [B,2,h,w]
            F_bw = flows_bw[k]

            # upsample & rescale to image size
            F_fw_up = self._upsample_flow(F_fw, H, W)
            F_bw_up = self._upsample_flow(F_bw, H, W)
            
            # Compute occlusion masks using FB-consistency
            if use_occ_mask:
                occ_fw = self._get_occu_mask_bidirection(F_fw_up, F_bw_up)
                occ_bw = self._get_occu_mask_bidirection(F_bw_up, F_fw_up)
                vis_mask_fw = 1 - occ_fw  # Visibility = 1 - occlusion
                vis_mask_bw = 1 - occ_bw
            else:
                vis_mask_fw = None
                vis_mask_bw = None

            # photometric forward & backward with occlusion masking
            photo_fw = self._photometric_pair(I1, I2, F_fw_up, vis_mask_fw)
            photo_bw = self._photometric_pair(I2, I1, F_bw_up, vis_mask_bw)
            photo = 0.5 * (photo_fw + photo_bw)


            # smoothness (on forward; can optionally also add backward if needed)
            smooth = self._smoothness_pair(F_fw_up, I1)

            # forward-backward consistency
            cons = self._fb_consistency_pair(F_fw_up, F_bw_up)
            
            # magnitude regularization (prevents zero-flow collapse)
            mag_reg = self._magnitude_regularization(F_fw_up, I1, I2)

            total_photo += photo
            total_smooth += smooth
            total_cons += cons
            total_mag_reg += mag_reg
            count += 1

        if count > 0:
            total_photo = total_photo / count
            total_smooth = total_smooth / count
            total_cons = total_cons / count
            total_mag_reg = total_mag_reg / count

        total = total_photo + total_smooth + total_cons + total_mag_reg

        return {
            "photo": total_photo,
            "smooth": total_smooth,
            "cons": total_cons,
            "mag_reg": total_mag_reg,
            "total": total,
        }


    # Optional: keep a forward-only version if you still want it for debugging
    def unsup_forward_only(self, clip: torch.Tensor, flows: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Simple forward-only unsupervised loss (for debugging or ablation).
        Not used in the main bidirectional training loop.
        """
        B, T, C, H, W = clip.shape
        total_photo = 0.0
        total_smooth = 0.0
        count = 0

        for k, F_fw in enumerate(flows):
            if k >= T - 1:
                break
            I1 = clip[:, k]
            I2 = clip[:, k + 1]
            F_fw_up = self._upsample_flow(F_fw, H, W)

            photo = self._photometric_pair(I1, I2, F_fw_up)
            smooth = self._smoothness_pair(F_fw_up, I1)

            total_photo += photo
            total_smooth += smooth
            count += 1

        if count > 0:
            total_photo /= count
            total_smooth /= count

        total = total_photo + total_smooth
        return {
            "photo": total_photo,
            "smooth": total_smooth,
            "total": total,
        }
