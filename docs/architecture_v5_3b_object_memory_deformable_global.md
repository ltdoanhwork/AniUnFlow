# V5.3b Deformable Ramp Global

`v5_3b_object_memory_deformable_global` is a stabilization follow-up to V5.3.

The key lesson from the first V5.3 run was that deformable slot motion was not collapsing, but it was entering too early and too strongly. That caused the branch to plateau around the V5.3 baseline without improving the large-motion tail.

V5.3b keeps the same high-level architecture:

- SAM-guided object memory
- lightweight coarse global matcher
- local dense refinement
- boundary-aware residual refinement
- deformable slot motion

The difference is how deformable motion is activated.

## What Changed

1. Epoch ramp for slot-basis scale
- `slot_basis_start_epoch`
- `slot_basis_ramp_epochs`
- `slot_basis_scale` is now treated as the maximum scale, not the always-on scale.

2. Confidence-gated deformation
- Slot basis contributions are modulated by slot matching confidence.
- Low-confidence slots stay close to the affine prior.

3. Temporal SAM support from neighboring frames
- Slot matching now uses SAM masks from the previous and next frames as an extra temporal support signal.
- This support reweights correspondence confidence before dense/global/deformable updates consume it.

4. Stronger regularization
- `slot_deformation_reg` is increased in the V5.3b configs.
- The intent is to only spend deformation capacity when the match signal supports it.

## Motivation

The stabilization target is specific:

- keep the early-epoch behavior as close as possible to the stronger affine/global V5 family
- let deformation help only after object identity and coarse motion have already become reliable
- reduce the chance that deformable slots become extra capacity without improving `epe_s>50`

## Configs

- `configs/v5_3b_object_memory_deformable_global.yaml`
- `configs/v5_3b_object_memory_deformable_global_finetune.yaml`

## Diagnostics To Watch

When training V5.3b, the most useful TensorBoard tags are:

- `train_v5_3b/slot_mag_mean`
- `train_v5_3b/dense_prior_mag_mean`
- `train_v5_3b/global_mag_mean`
- `train_v5_3b/global_conf_mean`
- `train_v5_3b/match_conf_mean`
- `train_v5_3b/temporal_support_mean`
- `train_v5_3b/slot_basis_energy`
- `train_v5_3b/deform_basis_scale`

If `slot_basis_energy` rises sharply before `deform_basis_scale` has ramped or while `match_conf_mean` remains weak, the branch is still entering too aggressively.
