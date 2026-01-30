# Training Configuration Improvements

## Key Changes Based on Research

### 1. Learning Rate & Scheduler
**Problem**: Original config used `lr=2e-4` with cosine scheduler
**Solution**: 
- Reduced to `lr=1.25e-4` (following FlowFormer/RAFT papers)
- Switched to **OneCycleLR** scheduler (proven superior for optical flow)
- Extended warmup from 5 to 10 epochs
- Start LR: `1.25e-4 / 25 = 5e-6` (gradual ramp-up prevents instability)

### 2. Photometric Loss Rebalancing
**Problem**: `alpha_ssim=0.2` was too low, causing pixel-level noise sensitivity
**Solution**:
- Increased `alpha_ssim` to `0.85` during warmup (following ARFlow)
- Plan: Switch to Census transform after warmup for better robustness

### 3. Stronger Regularization
**Problem**: Weak smoothness allowed noisy predictions
**Solution**:
- Increased `w_smooth` from `0.1` to `0.5`
- Increased `w_cons` from `0.05` to `0.1`
- Reduced segment losses initially (they were too strong relative to photometric loss)

### 4. Enhanced Data Augmentation
**Problem**: Insufficient augmentation led to overfitting to static patterns
**Solution**:
- Increased color jitter: `[0.4, 0.4, 0.4, 0.15]`
- Doubled grayscale probability: `0.2`
- This forces the model to learn motion rather than color/texture cues

## Expected Improvements

1. **Avoid Zero-Flow Collapse**: OneCycleLR + higher SSIM weight + stronger smoothness
2. **Faster Convergence**: Proper warmup prevents early training instability
3. **Better Generalization**: Enhanced augmentation + Census transform robustness
4. **Higher Flow Magnitudes**: Better photometric matching will allow model to predict larger, more accurate flows

## Usage

```bash
# With improved config
python scripts/train_unsup_animerun.py --config configs/train_unsup_animerun_sam_v2.yaml

# Monitor these metrics:
# - mean |flow| should increase from ~0.05 to ~0.5-1.0 over first 20 epochs
# - loss should decrease more smoothly (no spikes)
# - val/epe should improve steadily
```

## References
- RAFT unsupervised: OneCycleLR with pct_start=0.5, max_lr tuned per dataset
- ARFlow: SSIM-heavy photometric loss (0.85) + Census transform
- FlowFormer: max_lr=1.25e-4 for FlyingThings/Sintel
- Teacher-Student frameworks: Reduced segment loss weights to avoid overfitting
