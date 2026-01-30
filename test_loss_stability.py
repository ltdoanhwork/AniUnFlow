
import torch
import torch.nn as nn
from losses.segment_aware_losses import SegmentConsistencyFlowLoss, BoundaryAwareSmoothnessLoss

def test_stability():
    print("Testing SegmentConsistencyFlowLoss stability...")
    loss_fn = SegmentConsistencyFlowLoss(weight=0.1, use_charbonnier=True)
    
    # simulate small flow and masks
    B, C, H, W = 2, 2, 64, 64
    S = 4
    
    # Case 1: Identical flow (variance 0) - this triggered sqrt(0) NaN before
    flow = torch.zeros(B, C, H, W, requires_grad=True)
    masks = torch.rand(B, S, H, W)
    masks = masks / masks.sum(dim=1, keepdim=True) # normalize
    
    loss = loss_fn(flow, masks)
    print(f"Loss (zero variance): {loss.item()}")
    
    loss.backward()
    print("Backward pass successful (no crash)")
    if torch.isnan(flow.grad).any():
        print("FAIL: NaN gradients detected!")
    else:
        print("PASS: Gradients are finite")
        
    # Case 2: Boundary smoothness
    print("\nTesting BoundaryAwareSmoothnessLoss stability...")
    smooth_fn = BoundaryAwareSmoothnessLoss(weight=0.1)
    img = torch.rand(B, 3, H, W)
    boundary = torch.rand(B, 1, H, W)
    
    loss2 = smooth_fn(flow, img, segment_masks=None, boundary_map=boundary)
    print(f"Smoothness Loss: {loss2.item()}")
    loss2.backward()
    print("Backward pass successful")

if __name__ == "__main__":
    test_stability()
