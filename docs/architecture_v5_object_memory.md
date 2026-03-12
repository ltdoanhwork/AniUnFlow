# V5 Object-Memory SAM Branch

`AniFlowFormerTV5` is the second major AnimeRun research branch. It is intentionally different from the V4 hybrid matcher line: V4 starts from dense matching and injects SAM as guidance, while V5 starts from SAM-defined objects and builds dense flow from object motion first.

## Design

The branch is organized around five stages:

1. `PyramidEncoder` extracts frame features for the full clip.
2. SAM masks are converted into per-object slots by pooling level-1 and level-2 features inside each segment.
3. A temporal memory encoder mixes slot tokens across the entire clip (`T=5` by default) so correspondences are built from track-like object context rather than only pairwise appearance.
4. For each adjacent pair, V5 predicts segment correspondences, affine motion parameters, per-slot confidence, and layered occlusion/order scores.
5. The affine object motion field is rasterized into a dense coarse flow and a boundary-aware residual branch adds non-rigid corrections.

The output contract includes:

- `flows_fw` / `flows_bw`
- `flows_long`
- `segment_params_fw`
- `match_probs_fw`
- `match_confidence_fw`
- `layer_order_fw`
- `occlusion_slots_fw`
- `residual_flow_fw`

## Training strategy

V5 keeps the base unsupervised photometric loss, but adds object-memory specific regularization:

- Segment warp consistency: warped source segments should match target segments under learned correspondence.
- Piecewise residual loss: residual flow inside confident object interiors should stay small.
- Segment cycle consistency: adjacent correspondences should agree with long-gap matches.
- Layered order consistency: occlusion scores should agree with learned segment ordering.
- Boundary residual specialization: residual energy is encouraged to live near boundaries instead of re-learning the full field.

The default runtime schedule is staged:

- Early stage: object correspondences + coarse affine motion only.
- Middle stage: enable residual refinement and layered ordering.
- Late stage: enable long-gap and cycle terms.

## Configs

Main experiment config:

- `configs/v5_object_memory_sam_parallel.yaml`

Ablations:

- `configs/v5_object_memory_sam_no_layer_order.yaml`
- `configs/v5_object_memory_sam_no_temporal_memory.yaml`
- `configs/v5_object_memory_sam_affine_only.yaml`

## Reporting

`scripts/collect_animerun_results.py` now emits both CSV and Markdown summaries and tags V5 runs as `V5 Object Memory` so they can be compared directly against `V4 Hybrid SAM`.
