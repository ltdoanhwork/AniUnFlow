# V5.2 Object-Memory Global Branch

`v5_2_object_memory_global` is the follow-up to V5.1 after analyzing why the heavier V6 branch underperformed. The main lesson from V6 was that a coarse global search signal is useful, but the full combination of non-rigid slot flow, visibility compositing, and extra heavy objectives made optimization much less stable than V5.1.

## Motivation

V5.1 is strong because it keeps a clear hierarchy:

1. object-memory prior,
2. local dense recovery,
3. boundary-aware residual refinement.

V6 tried to solve large motion by changing too many parts of the model at once. V5.2 keeps the stable V5.1 backbone and only imports the single most useful idea from V6: a lightweight coarse global matcher at the level-2 feature scale.

## Design

V5.2 differs from V5.1 in one focused place.

- The object-memory slot pipeline remains unchanged.
- The affine slot prior remains unchanged.
- The local level-2 and level-1 dense matchers remain unchanged.
- A new coarse global matcher predicts a low-resolution large-motion proposal and confidence map from level-2 features.
- The slot prior and global proposal are fused with confidence-aware gating before the local dense matchers refine them.

This gives V5.2 a simpler philosophy than V6:

- keep the stable object-first V5.1 backbone,
- add one extra large-motion cue,
- avoid introducing new heavy branches unless they earn their complexity.

## Training

V5.2 keeps the V5.1 trainer path and loss bundle, but adds two light constraints for the new global branch:

- `global_photo`: an auxiliary unsupervised photometric term on the global coarse flow,
- `global_dense_consistency`: a structural regularizer that keeps the dense prior close to the global branch in confident interior regions.

These terms are designed to stop the new global branch from collapsing or overpowering the slot prior, which was one of the main failure patterns observed in V6.

## Configs

- `configs/v5_2_object_memory_global.yaml`
- `configs/v5_2_object_memory_global_finetune.yaml`

## Reporting

`scripts/collect_animerun_results.py` classifies this branch as `V5.2 Object Memory Global` so it can be compared directly against `V5.1 Object Memory Dense`, `V5 Object Memory`, and `V6 Global Slot Search` in the generated result tables.
