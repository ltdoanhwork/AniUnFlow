# V5.1 Object-Memory Dense Branch

`v5_1_object_memory_dense` is the follow-up to the original V5 object-memory branch. The motivation is simple: V5's slot-affine prior gives good object structure, but it still under-serves large displacement and non-rigid motion inside the same object. V5.1 keeps the object-memory core and adds a dense large-motion recovery path.

## Design

V5.1 keeps the V5 slot pipeline and extends it in three places:

1. SAM masks still define object slots pooled from level-1 and level-2 image features.
2. Temporal slot memory still reasons across the full clip (`T=5`).
3. Slot correspondence still predicts object matches, affine motion, and layered order priors.
4. A new level-2 dense correlation matcher refines the slot-affine prior before it reaches the finest level.
5. The refined level-2 flow is upsampled and fused with the slot-affine level-1 prior.
6. A second level-1 dense correlation matcher performs local dense recovery around the fused prior.
7. The boundary-aware residual refiner now works on top of this dense prior instead of only the affine rasterization.

The result is a hybrid object-first model:

- Object slots give identity, occlusion, and structural bias.
- Dense correlation gives large-motion and intra-object recovery.
- Multi-scale refinement bridges the two.

## Output contract

V5.1 preserves all V5 outputs and adds extra dense-prior diagnostics:

- `slot_flow_fw`
- `dense_prior_flow_fw`
- `corr_confidence_fw`

These are useful for debugging whether the dense branch is genuinely improving over the slot-affine prior.

## Training strategy

V5.1 keeps the V5 loss bundle and adds one more structural term:

- Dense-slot consistency: inside confident object interiors, the learned dense prior should stay close to the slot motion unless the dense evidence strongly disagrees.

This keeps the new dense matcher from throwing away the object-centric inductive bias while still allowing large-motion corrections.

## Config

Main config:

- `configs/v5_1_object_memory_dense_parallel.yaml`

Suggested comparison set:

- `configs/v5_object_memory_sam_parallel.yaml`
- `configs/v5_1_object_memory_dense_parallel.yaml`
- `configs/v4_6_hybrid_sam_sub7_v2.yaml`

## Reporting

`scripts/collect_animerun_results.py` now tags this branch as `V5.1 Object Memory Dense`, so it shows up separately from `V5 Object Memory` and `V4 Hybrid SAM` in the CSV/Markdown reports.
