# V5.3 Deformable Object-Memory Global Branch

`v5_3_object_memory_deformable_global` is a large follow-up to V5.2. The goal is to keep the stability of the V5 family while adding a genuinely bigger motion component: deformable slot motion.

## Motivation

V5.1 and V5.2 still rely on affine object motion at the slot level. That works well for rigid or near-rigid regions, but it remains too limited for anime-specific motion such as hair, cloth, articulated limbs, and expressive shape deformation inside the same segment.

V6 tried to address this with a much heavier redesign, but that branch proved difficult to optimize. V5.3 takes a more targeted route: keep the stable V5.2 stack and upgrade the slot motion model itself.

## Design

V5.3 keeps all major V5.2 components:

- SAM-guided object slots,
- temporal slot memory,
- coarse global matcher,
- local dense refinement,
- boundary-aware residual refinement.

The large new component is the slot-motion head.

Instead of predicting only 6 affine parameters per slot, V5.3 predicts:

- 6 affine parameters, and
- a set of low-rank deformation basis coefficients.

These basis coefficients modulate a fixed bank of spatial basis maps, allowing each slot to express non-rigid motion while preserving the slot-centric inductive bias.

In practice, V5.3 behaves like this:

- affine motion still carries the coarse object displacement,
- basis deformation captures intra-object warping,
- the global matcher helps large displacement,
- the dense branch refines local detail on top of the stronger slot prior.

## Training

V5.3 keeps the V5.2 auxiliary objectives and adds one more structural regularizer:

- `slot_deformation_reg`: penalizes deformation-basis energy in low-confidence slots, so the deformable head does not explode or replace the whole motion model with uncontrolled basis noise.

This keeps the new component expressive but still disciplined.

## Configs

- `configs/v5_3_object_memory_deformable_global.yaml`
- `configs/v5_3_object_memory_deformable_global_finetune.yaml`

## Reporting

The current collector already supports this family once a workspace is created; it will appear as `V5.3 Deformable Object Memory Global` after results are gathered from real runs.
