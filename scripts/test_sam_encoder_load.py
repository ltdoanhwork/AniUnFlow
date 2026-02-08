#!/usr/bin/env python3
"""
Test SAM2 encoder loading with corrected paths.
"""
import sys
import os
sys.path.insert(0, os.getcwd())

from models.aniunflow_v4.sam_encoder import SAMEncoderWrapper
import torch

print("=== Testing SAM2 Encoder Loading ===")
print(f"CWD: {os.getcwd()}")

try:
    print("\n1. Creating SAMEncoderWrapper with default params...")
    encoder = SAMEncoderWrapper(device='cuda')
    print(f"   ✓ Wrapper created")
    print(f"   Checkpoint: {encoder.checkpoint}")
    print(f"   Config: {encoder.config}")
    
    print("\n2. Testing lazy load (first forward call)...")
    dummy_input = torch.randn(1, 3, 1024, 1024).cuda()
    features = encoder(dummy_input)
    print(f"   ✓ Forward pass successful")
    print(f"   Features: {list(features.keys())}")
    
    print("\n✅ SAM2 Encoder loaded successfully!")
    
except Exception as e:
    print(f"\n❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
