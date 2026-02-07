# Global Matching V3 Architecture

Global Matching V3 is a significant upgrade to the AniFlowFormer-T model, integrating **Segment Anything Model (SAM-2)** guidance to improve optical flow estimation, particularly at object boundaries and in flat regions common in anime.

## Core Philosophy
1.  **Structure-Aware Flow**: Uses SAM segments as a strong structural prior.
2.  **Boundary Preservation**: Explicitly models motion discontinuities at segment boundaries.
3.  **Region Consistency**: Enforces consistent motion within segments using affinity and cross-attention.

## Architecture Overview

The V3 architecture builds upon the V1/V2 baseline but introduces specialized modules for SAM integration.

### 1. SAM Guidance Adapter V3 (`models/aniunflow/sam_adapter_v3.py`)
Matches SAM masks with image features using 4 complementary modes:
*   **Feature Concatenation**: Concatenates boundary-aware features with image features (inspired by SAMFlow).
*   **Attention Bias**: Generates an attention bias matrix where pixels within the same segment have higher affinity.
*   **Cost Modulation**: Modulates the matching cost volume to reduce penalties for motion discontinuities at segment boundaries.
*   **Object Pooling**: Pools features within each segment to create "Segment Tokens" that represent object-level motion.

### 2. SAM-Guided Global Matcher (`models/aniunflow/global_matcher_v3.py`)
Enhances the global matching process:
*   **Boundary-Aware Tokenizer**: Uses the attention bias from SAM to guide the tokenization of cost volume features.
*   **Segment Affinity**: Ensures that global matching respects object boundaries.

### 3. Latent Cost Memory V3 (`models/aniunflow/lcm_v3.py`)
Updates the latent state using segment context:
*   **Segment Cross-Attention**: The latent motion state attends to "Segment Tokens" to incorporate object-level motion information.
*   **Gated Fusion**: Adaptively fuses segment information with the pixel-level motion state.

## Loss Functions (`losses/segment_aware_losses_v3.py`)

V3 introduces UnSAMFlow-inspired unsupervised losses:
*   **Homography Smoothness**: Fits an affine/homography model to each segment and penalizes deviations.
*   **Segment Motion Consistency**: Enforces low variance of flow vectors within each segment.
*   **Boundary Sharpness**: Aligns flow gradients with SAM boundaries to promote sharp motion edges.
*   **Cross-Segment Discontinuity**: Encourages flow to be different across boundaries (optional).

## Configuration

Training is controlled via `configs/train_unsup_animerun_sam_v3.yaml`.

```yaml
model:
  name: AniFlowFormerTV3
  args:
    sam_version: 3
    use_sam: true
    # ... V3 options ...
```

## Performance Optimization
*   **Precomputed Masks**: To avoid slow online SAM inference, masks are precomputed using `scripts/precompute_sam_masks.py` and stored as integer label maps.
*   **Efficient Loading**: The dataset loads these label maps, and `SAMGuidanceAdapterV3` efficiently expands them to one-hot encodings on the GPU.
