"""
Test Script for Global Matching V3
===================================
Verifies all V3 components work correctly before training.

Tests:
1. SAMGuidanceAdapterV3 - 4 modes
2. SAMGuidedGlobalMatcher - boundary-aware matching
3. LatentCostMemoryV3 - segment cross-attention
4. AniFlowFormerTV3 - full forward pass
5. V3 Losses - homography smoothness etc.
"""
import sys
sys.path.insert(0, '/home/serverai/ltdoanh/AniUnFlow')

import torch
import torch.nn as nn


def test_sam_adapter_v3():
    """Test SAMGuidanceAdapterV3 with all 4 modes."""
    print("=" * 60)
    print("Testing SAMGuidanceAdapterV3")
    print("=" * 60)
    
    from models.aniunflow.sam_adapter_v3 import SAMGuidanceAdapterV3
    
    # Create adapter
    adapter = SAMGuidanceAdapterV3(
        feat_dim=64,
        token_dim=128,
        num_segments=16,
        num_heads=4,
        use_feature_concat=True,
        use_attention_bias=True,
        use_cost_modulation=True,
        use_object_pooling=True,
    ).cuda()
    
    # Create dummy inputs
    B, T, S, H, W = 2, 3, 16, 46, 96
    masks = torch.rand(B, T, S, H, W).cuda()
    features = [torch.randn(B, 64, 46, 96).cuda() for _ in range(T)]
    
    # Forward pass
    outputs = adapter(masks, features)
    
    # Check outputs
    print(f"  boundary_maps shape: {outputs['boundary_maps'].shape}")
    print(f"  enhanced_features: {len(outputs['enhanced_features'])} tensors")
    print(f"  attn_bias: {len(outputs['attn_bias'])} tensors")
    print(f"  seg_tokens shape: {outputs['seg_tokens'].shape}")
    print(f"  edge_maps shape: {outputs['edge_maps'].shape}")
    
    # Assertions
    assert outputs['boundary_maps'].shape == (B, T, 1, H, W)
    assert len(outputs['enhanced_features']) == T
    assert outputs['seg_tokens'].shape == (B, T, S, 128)
    
    print("  ✅ SAMGuidanceAdapterV3 PASSED")
    return True


def test_sam_guided_matcher():
    """Test SAMGuidedGlobalMatcher."""
    print("\n" + "=" * 60)
    print("Testing SAMGuidedGlobalMatcher")
    print("=" * 60)
    
    from models.aniunflow.global_matcher_v3 import SAMGuidedGlobalMatcher
    
    matcher = SAMGuidedGlobalMatcher(
        dim=64,
        token_dim=128,
        num_heads=4,
        topk=64,
        use_boundary_modulation=True,
        use_segment_affinity=True,
    ).cuda()
    
    B, C, H, W = 2, 64, 46, 96
    feat1 = torch.randn(B, C, H, W).cuda()
    feat2 = torch.randn(B, C, H, W).cuda()
    boundary = torch.rand(B, 1, H, W).cuda()
    
    # Forward pass
    tokens = matcher(feat1, feat2, boundary_map=boundary)
    
    print(f"  Input shape: ({B}, {C}, {H}, {W})")
    print(f"  Output tokens shape: {tokens.shape}")
    
    assert tokens.shape == (B, 128, H, W)
    
    print("  ✅ SAMGuidedGlobalMatcher PASSED")
    return True


def test_lcm_v3():
    """Test LatentCostMemoryV3 with segment cross-attention."""
    print("\n" + "=" * 60)
    print("Testing LatentCostMemoryV3")
    print("=" * 60)
    
    from models.aniunflow.lcm_v3 import LatentCostMemoryV3
    
    lcm = LatentCostMemoryV3(
        token_dim=128,
        depth=4,
        heads=4,
        use_segment_cross_attn=True,
    ).cuda()
    
    # Create dummy tokens (3 levels, 2 time steps each)
    B, D, H, W = 2, 128, 23, 48
    tokens_per_level = [
        [torch.randn(B, D, H, W).cuda() for _ in range(2)]
        for _ in range(3)
    ]
    
    # Segment tokens
    seg_tokens = torch.randn(B, 2, 16, 128).cuda()
    
    # Forward pass
    out = lcm(tokens_per_level, seg_tokens=seg_tokens)
    
    print(f"  Input: 3 levels x 2 time steps, shape ({B}, {D}, {H}, {W})")
    print(f"  Segment tokens: ({B}, 2, 16, 128)")
    print(f"  Output: {len(out)} levels, each with {len(out[0])} tensors")
    print(f"  Output[0][0] shape: {out[0][0].shape}")
    
    assert len(out) == 3
    assert len(out[0]) == 2
    assert out[0][0].shape == (B, D, H * W)
    
    print("  ✅ LatentCostMemoryV3 PASSED")
    return True


def test_model_v3():
    """Test full AniFlowFormerTV3 forward pass."""
    print("\n" + "=" * 60)
    print("Testing AniFlowFormerTV3 (Full Model)")
    print("=" * 60)
    
    from models.aniunflow.model_v3 import AniFlowFormerTV3, ModelConfigV3
    
    cfg = ModelConfigV3(
        enc_channels=32,
        token_dim=128,
        lcm_depth=4,
        lcm_heads=4,
        gtr_depth=1,
        gtr_heads=4,
        iters_per_level=3,
        use_sam=True,
        sam_version=3,
        num_segments=16,
    )
    
    model = AniFlowFormerTV3(cfg).cuda()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Create inputs
    B, T, H, W = 2, 3, 384, 768
    clip = torch.randn(B, T, 3, H, W).cuda()
    sam_masks = torch.rand(B, T, 16, H, W).cuda()
    
    # Forward pass (without SAM first)
    print("\n  [1] Testing without SAM...")
    with torch.no_grad():
        output = model(clip, sam_masks=None, use_sam=False)
    
    print(f"      flows: {len(output['flows'])} tensors, shape {output['flows'][0].shape}")
    print(f"      occ: {len(output['occ'])} tensors")
    
    # Forward pass (with SAM)
    print("\n  [2] Testing with SAM...")
    with torch.no_grad():
        output = model(clip, sam_masks=sam_masks, use_sam=True, debug=True)
    
    print(f"      flows: {len(output['flows'])} tensors, shape {output['flows'][0].shape}")
    print(f"      sam_outputs keys: {list(output.get('sam_outputs', {}).keys())}")
    
    # Check flow magnitude
    flow_mag = (output['flows'][0] ** 2).sum(dim=1).sqrt()
    mean_mag = flow_mag.mean().item()
    
    print(f"\n  Flow magnitude: mean={mean_mag:.4f}")
    
    if mean_mag > 0.01:
        print("  ✅ Flow magnitude reasonable (> 0.01)")
    else:
        print("  ⚠️  Flow magnitude very small")
    
    print("  ✅ AniFlowFormerTV3 PASSED")
    return True


def test_losses_v3():
    """Test V3 loss functions."""
    print("\n" + "=" * 60)
    print("Testing Segment-Aware Losses V3")
    print("=" * 60)
    
    from losses.segment_aware_losses_v3 import (
        HomographySmoothnessLoss,
        SegmentMotionConsistencyLoss,
        BoundarySharpnessLoss,
        SegmentAwareLossModuleV3,
    )
    
    B, H, W = 2, 92, 192
    S = 16
    
    flow = torch.randn(B, 2, H, W).cuda()
    masks = torch.rand(B, S, H, W).cuda()
    boundary = torch.rand(B, 1, H, W).cuda()
    
    # Test individual losses
    print("  Testing HomographySmoothnessLoss...")
    homo_loss = HomographySmoothnessLoss(weight=0.1).cuda()
    loss_val = homo_loss(flow, masks)
    print(f"      Loss value: {loss_val.item():.6f}")
    
    print("  Testing SegmentMotionConsistencyLoss...")
    motion_loss = SegmentMotionConsistencyLoss(weight=0.05).cuda()
    loss_val = motion_loss(flow, masks)
    print(f"      Loss value: {loss_val.item():.6f}")
    
    print("  Testing BoundarySharpnessLoss...")
    boundary_loss = BoundarySharpnessLoss(weight=0.05).cuda()
    loss_val = boundary_loss(flow, boundary)
    print(f"      Loss value: {loss_val.item():.6f}")
    
    # Test combined module
    print("\n  Testing SegmentAwareLossModuleV3...")
    cfg = {
        'loss': {
            'homography_smooth': {'enabled': True, 'weight': 0.1},
            'segment_motion_consistency': {'enabled': True, 'weight': 0.05},
            'boundary_sharpness': {'enabled': True, 'weight': 0.05},
            'cross_segment_discontinuity': {'enabled': False},
        }
    }
    loss_module = SegmentAwareLossModuleV3(cfg).cuda()
    
    flows = [flow, flow]
    masks_5d = masks.unsqueeze(1).expand(-1, 2, -1, -1, -1)
    boundary_5d = boundary.unsqueeze(1).expand(-1, 2, -1, -1, -1)
    
    losses = loss_module(flows, masks_5d, boundary_5d)
    print(f"      Total V3 loss: {losses['total_v3_loss'].item():.6f}")
    print(f"      Loss keys: {list(losses.keys())}")
    
    print("  ✅ Segment-Aware Losses V3 PASSED")
    return True


def run_all_tests():
    """Run all V3 component tests."""
    print("\n" + "=" * 70)
    print("   GLOBAL MATCHING V3 TEST SUITE")
    print("=" * 70)
    
    results = {}
    
    try:
        results['SAMGuidanceAdapterV3'] = test_sam_adapter_v3()
    except Exception as e:
        print(f"  ❌ SAMGuidanceAdapterV3 FAILED: {e}")
        results['SAMGuidanceAdapterV3'] = False
    
    try:
        results['SAMGuidedGlobalMatcher'] = test_sam_guided_matcher()
    except Exception as e:
        print(f"  ❌ SAMGuidedGlobalMatcher FAILED: {e}")
        results['SAMGuidedGlobalMatcher'] = False
    
    try:
        results['LatentCostMemoryV3'] = test_lcm_v3()
    except Exception as e:
        print(f"  ❌ LatentCostMemoryV3 FAILED: {e}")
        results['LatentCostMemoryV3'] = False
    
    try:
        results['AniFlowFormerTV3'] = test_model_v3()
    except Exception as e:
        print(f"  ❌ AniFlowFormerTV3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['AniFlowFormerTV3'] = False
    
    try:
        results['SegmentAwareLossesV3'] = test_losses_v3()
    except Exception as e:
        print(f"  ❌ SegmentAwareLossesV3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['SegmentAwareLossesV3'] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("   TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {name}: {status}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    print("=" * 70)
    
    return all(results.values())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, default='all', 
                       choices=['all', 'adapter', 'matcher', 'lcm', 'model', 'losses'])
    args = parser.parse_args()
    
    if args.test == 'all':
        success = run_all_tests()
    elif args.test == 'adapter':
        success = test_sam_adapter_v3()
    elif args.test == 'matcher':
        success = test_sam_guided_matcher()
    elif args.test == 'lcm':
        success = test_lcm_v3()
    elif args.test == 'model':
        success = test_model_v3()
    elif args.test == 'losses':
        success = test_losses_v3()
    
    sys.exit(0 if success else 1)
