# V5.4 SAM Propagation Memory

`V5.4` is a stabilization-oriented follow-up to `V5.3b`. The main idea is simple: neighboring SAM masks should not only reweight slot correspondences at the object level, but also provide a dense agreement field that tells the refiner where propagated object structure is trustworthy.

## Motivation

`V5.3b` showed that delayed deformable slots and temporal SAM support were enough to keep training stable, but they did not solve the main generalization failure. The model could still optimize the training objective while drifting away from validation EPE late in training. The root issue was that temporal SAM information only entered as slot-level confidence. That was too weak to constrain dense refinement, residual correction, and deformation where the model most needed guidance: object interiors, boundaries, and re-association regions after motion.

## Core Change

`V5.4` adds a dense **SAM propagation memory** pathway:

1. Take the previous frame's SAM labels.
2. Warp them into the current frame using the previously predicted flow.
3. Compare the warped labels with the current frame's SAM labels.
4. Convert that overlap into a dense agreement map.

This agreement map is then used in three places:

- to boost confidence where propagated SAM structure agrees with the current mask,
- to raise boundary sensitivity where the propagated mask disagrees,
- to regularize the model through a dedicated `sam_memory_consistency` loss.

## Practical Design

`V5.4` keeps the strongest parts of `V5.3b`:

- object-memory slot matching,
- lightweight coarse global matcher,
- local dense refinement,
- delayed deformable slot basis,
- boundary-aware residual refinement.

What changes is how temporal SAM evidence is used. `V5.3b` used neighboring masks only to reweight slot matching. `V5.4` reuses them again after flow prediction as a pixel-level memory signal. This creates a much tighter loop between SAM structure and dense flow refinement.

## Training Strategy

The schedule is deliberately more conservative than `V5.3b`:

- no `stride 1..3` stage,
- no late layered-order stage,
- long-gap and cycle losses are delayed,
- SAM memory is given an explicit loss weight.

The goal is to preserve the early-epoch gains instead of chasing harder curriculum stages that previously caused late-run drift.

## Diagnostics

`V5.4` logs all `V5.3b` diagnostics plus:

- `train_v5_4/sam_memory_agreement_mean`

This value is intended to tell us whether the propagated SAM structure is actually staying aligned with the next frame as training progresses.
