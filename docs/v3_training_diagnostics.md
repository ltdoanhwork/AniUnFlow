# V3 Training Diagnostics Report

**Date:** 2026-02-07  
**Workspace:** `workspaces/global_matching_v3`

## Critical Issues Found

### 1. **Photometric Loss = 0** (CRITICAL)
```
train_loss/photo: last=0.0000, mean=0.0000
```
The photometric loss (main unsupervised signal) is **exactly zero**. This means:
- Either the loss is not being computed
- Or the flow warping is producing perfect matches (impossible)
- **Most likely cause**: Some masking or gating is zeroing out this loss

### 2. **Flow Magnitude Extremely Low** (CRITICAL)
```
train_flow/mag_mean: last=0.1253, mean=0.1308
train_flow/mag_std: last=0.0016
train_flow/mag_max: last=0.1287
```
- Average flow: **~0.13 pixels** (should be 1-10+ for animation)
- Max flow: **~0.13 pixels** (should be 10-50+)
- Std: **~0.001** (almost no variation - model outputting uniform near-zero flow!)

**Diagnosis**: The model is collapsing to predicting near-zero flow everywhere.

### 3. **Segment Detection = 1** (CRITICAL)
```
train_segment/num_segments: last=1.0000, mean=1.0000
train_segment/avg_size: last=174086.0938
```
- Only **1 segment** detected per image (should be 16)
- Segment covers entire image (174k pixels = 368×768×0.62)

**Diagnosis**: SAM masks are not being loaded correctly, or the mask expansion is failing.

---

## Root Cause Analysis

### Hypothesis A: Loss Masking Issue
The `w_photo=1.0` in config, but actual logged value is 0. Check:
1. Is occlusion mask zeroing out all pixels?
2. Is warmup disabling photo loss? (`disable_occ_during_warmup: true`, `warmup_steps=1000`)

### Hypothesis B: Flow Collapse (Mode Collapse)
The model is learning to output near-zero flow because:
1. Zero flow gives low smoothness loss (no gradients)
2. Zero flow in flat regions gives low photo loss if images are similar
3. `w_mag_reg=0.05` with `min_flow_mag=0.5` may not be strong enough

### Hypothesis C: SAM Mask Loading Failure
Only 1 segment detected suggests:
1. Masks are all zeros (background only)
2. One-hot expansion is not working
3. Path resolution in dataset is failing

---

## Proposed Fixes

### Fix 1: Debug Mask Loading
Add print statements to verify masks are loaded:
```python
# In trainer_segment_aware.py, after mask loading:
if "sam_masks" in batch:
    print(f"[DEBUG] Loaded masks: shape={batch['sam_masks'].shape}, unique_values={batch['sam_masks'].unique()}")
```

### Fix 2: Check Photometric Loss Computation
Verify in `train_step` that:
1. `loss_photo` is computed before any masking
2. Occlusion mask is not all-zeros

### Fix 3: Increase Flow Magnitude Regularization
```yaml
loss:
  w_mag_reg: 0.15  # Increase from 0.05
  min_flow_mag: 1.0  # Increase from 0.5
```

### Fix 4: Disable Segment Losses Initially
To isolate the problem:
```yaml
sam_guidance:
  feature_concat: false
  attention_bias: false
  cost_modulation: false
  object_pooling: false
```
Train baseline first, then re-enable one by one.

### Fix 5: Verify Warmup Behavior
Check if warmup is incorrectly disabling all losses, not just occlusion.

---

## Recommended Next Steps

1. **Immediate**: Add debug prints to verify mask loading with `batch['sam_masks'].unique()`
2. **Quick test**: Run 100 iterations with all SAM modes disabled to confirm baseline works
3. **If masks are broken**: Fix path resolution in `clip_dataset_unsup.py`
4. **If flow still collapses**: Increase `w_mag_reg` to 0.2 and `min_flow_mag` to 2.0
