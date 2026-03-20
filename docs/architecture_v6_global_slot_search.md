# V6 Global-Slot-Search Branch

`v6_global_slot_search` is the next research branch after V5/V5.1, aimed at the failure mode that remained stubborn in earlier runs: very large displacement. The design keeps SAM-guided object memory, but stops treating dense recovery as a purely local correction around an affine prior.

## Motivation

Earlier branches taught us two things:

1. Object-centric priors improve structure and threshold accuracy, but affine slot motion alone is too rigid for anime articulation, cloth, and hair.
2. Local dense refinement helps medium motion, but it still under-serves the large-motion tail when the initial prior is too weak or too local.

V6 addresses both limitations by introducing a coarse global search path before local refinement and by replacing purely affine slot motion with non-rigid slot flow.

## Design

V6 has four major stages.

1. SAM masks define object slots pooled from level-1 and level-2 image features.
2. A temporal slot-memory encoder reasons over the full clip (`T=5`) so slot identity is not estimated from isolated pairs alone.
3. Each slot predicts an affine component plus a low-rank non-rigid deformation basis. This produces a dense slot-conditioned flow prior.
4. A coarse global matcher operates on pooled level-2 features to produce a large-displacement proposal and confidence map.
5. The slot prior and global proposal are fused using confidence-aware blending.
6. Two local dense correlation stages refine this fused prior at level-2 and level-1.
7. Visibility-aware compositing and a final boundary-sensitive residual refiner produce the final dense flow.

This makes V6 explicitly hierarchical:

- object memory provides identity and structure,
- global search handles large displacement,
- local refinement restores fine detail,
- visibility and residual refinement sharpen difficult boundaries.

## Output Contract

The V6 model exports the standard forward and backward flow lists and adds intermediate diagnostics so we can inspect failure modes directly:

- `slot_flow_fw`
- `global_flow_fw`
- `fused_coarse_flow_fw`
- `dense_prior_flow_fw`
- `residual_flow_fw`
- `match_confidence_fw`
- `global_corr_confidence_fw`
- `local_corr_confidence_fw`
- `slot_visibility_fw`
- `dense_occlusion_fw`
- `slot_basis_coeffs_fw`

These tensors are intentionally logged because V5.1 debugging showed that research progress depends on seeing whether the object prior, global search, and local refinement are cooperating or fighting each other.

## Training Strategy

V6 uses a staged curriculum.

- Stage 0: slot memory plus non-rigid slot flow only.
- Stage 1: enable the global matcher and train slot/global fusion.
- Stage 2: enable local dense refinement.
- Stage 3: enable visibility compositing, long-gap reasoning, and stronger structural losses.
- Stage 4: expand stride and turn on hard-motion reweighting.
- Final fine-tune: low learning rate cosine schedule with validation every epoch.

The V6 loss bundle keeps the unsupervised photometric base and adds structural terms for:

- segment warp consistency,
- dense-slot consistency,
- global/fused coarse consistency,
- visibility consistency,
- occlusion-aware compositing,
- slot deformation regularization,
- hard-motion reweighting.

The branch also keeps `slot_photo` and `global_photo` auxiliary supervision in the trainer so both the object prior and the coarse global search remain grounded during training.

## Configs

Main configs:

- `configs/v6_global_slot_search_parallel.yaml`
- `configs/v6_global_slot_search_finetune.yaml`

Ablations:

- `configs/v6_no_global_search.yaml`
- `configs/v6_affine_only_slots.yaml`
- `configs/v6_no_visibility_compositing.yaml`
- `configs/v6_no_hard_motion_reweight.yaml`

## Reporting

`scripts/collect_animerun_results.py` now tags this branch as `V6 Global Slot Search`, so it appears separately from `V5.1 Object Memory Dense`, `V5 Object Memory`, and `V4 Hybrid SAM` in the generated CSV and Markdown summaries.
