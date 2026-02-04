"""
Test script for GlobalMatchingTokenizer integration
====================================================
Tests forward pass with realistic inputs before full training.
"""
import sys
sys.path.insert(0, '/home/serverai/ltdoanh/AniUnFlow')

import torch
from models.aniunflow.model import AniFlowFormerT, ModelConfig

def test_global_matcher():
    print("=" * 60)
    print("Testing GlobalMatchingTokenizer Integration")
    print("=" * 60)
    
    # Create model config (lite version)
    cfg = ModelConfig(
        enc_channels=32,
        token_dim=128,
        lcm_depth=4,
        lcm_heads=4,
        gtr_depth=1,
        gtr_heads=4,
        iters_per_level=3,
        use_sam=False,
    )
    
    # Create model
    print("\n[1] Creating model...")
    model = AniFlowFormerT(cfg).cuda()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Create dummy input
    print("\n[2] Creating dummy input (B=2, T=3, H=384, W=768)...")
    B, T, H, W = 2, 3, 384, 768
    clip = torch.randn(B, T, 3, H, W).cuda()
    
    # Forward pass
    print("\n[3] Forward pass...")
    try:
        with torch.no_grad():
            output = model(clip)
        
        flows = output['flows']
        occ = output['occ']
        
        print(f"  ✅ Forward pass successful!")
        print(f"  Number of flows: {len(flows)}")
        print(f"  Flow[0] shape: {flows[0].shape}")
        
        # Check flow magnitude
        flow_mag = (flows[0]**2).sum(dim=1).sqrt()  # [B, H, W]
        mean_mag = flow_mag.mean().item()
        max_mag = flow_mag.max().item()
        
        print(f"\n[4] Flow magnitude check:")
        print(f"  Mean magnitude: {mean_mag:.4f} pixels")
        print(f"  Max magnitude: {max_mag:.4f} pixels")
        
        if mean_mag > 0.01:
            print(f"  ✅ Flow magnitude is reasonable (> 0.01)")
        else:
            print(f"  ⚠️  Flow magnitude very small - may collapse during training")
        
        print("\n" + "=" * 60)
        print("Test Summary: PASSED")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n  ❌ Forward pass FAILED!")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_global_matcher()
    sys.exit(0 if success else 1)
