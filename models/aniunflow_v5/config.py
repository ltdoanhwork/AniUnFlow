from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import List


@dataclass
class V5SAMConfig:
    enabled: bool = True
    use_encoder_features: bool = True
    use_mask_guidance: bool = True
    encoder_checkpoint: str = "models/sam2/checkpoints/sam2.1_hiera_tiny.pt"
    encoder_config: str = "configs/sam2.1/sam2.1_hiera_t.yaml"
    encoder_freeze: bool = True
    feature_dim: int = 256
    feature_scales: List[int] = field(default_factory=lambda: [8, 16])
    num_segments: int = 32
    mask_cache_dir: str | None = None


@dataclass
class V5LossConfig:
    photo: float = 1.0
    photo_ssim_alpha: float = 0.85
    smooth: float = 0.02
    consistency: float = 0.02
    mag_reg: float = 0.05
    min_flow_mag: float = 0.5
    warmup_steps: int = 1000
    disable_occ_during_warmup: bool = True
    occ_aware_start_epoch: int = 1

    segment_warp: float = 0.12
    slot_photo: float = 0.0
    piecewise_residual: float = 0.03
    segment_cycle: float = 0.04
    layered_order: float = 0.03
    boundary_residual: float = 0.02
    dense_slot_consistency: float = 0.05

    long_gap_photo_weight: float = 0.08
    long_gap_consistency_weight: float = 0.03
    long_gap_start_epoch: int = 12
    long_gap_ramp_epochs: int = 8


@dataclass
class V5ModelConfig:
    name: str = "AniFlowFormerTV5"
    backbone: str = "v5_object_memory_sam"
    enc_channels: int = 32
    slot_dim: int = 128
    slot_hidden_dim: int = 192
    temporal_memory_depth: int = 2
    temporal_memory_heads: int = 4
    temporal_memory_dropout: float = 0.0
    num_slots: int = 32
    overlap_prior_weight: float = 2.0
    use_long_gap_matching: bool = True
    predict_layer_order: bool = True
    dense_match_hidden_dim: int = 128
    dense_match_radius_l2: int = 6
    dense_match_radius_l1: int = 4
    dense_delta_scale_l2: float = 6.0
    dense_delta_scale_l1: float = 4.0
    dense_prior_mix: float = 0.65
    dense_update_scale_l2: float = 0.75
    dense_update_scale_l1: float = 0.5
    dense_confidence_floor: float = 0.1
    residual_hidden_dim: int = 96
    residual_blocks: int = 3
    residual_boundary_scale: float = 0.8
    residual_base_scale: float = 0.2
    residual_confidence_floor: float = 0.35


@dataclass
class V5Config:
    model: V5ModelConfig = field(default_factory=V5ModelConfig)
    sam: V5SAMConfig = field(default_factory=V5SAMConfig)
    loss: V5LossConfig = field(default_factory=V5LossConfig)
    data: dict = field(default_factory=dict)
    optim: dict = field(default_factory=dict)
    logging: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)
    ckpt: dict = field(default_factory=dict)
    viz: dict = field(default_factory=dict)
    runtime: dict = field(default_factory=dict)
    teacher: dict = field(default_factory=dict)
    workspace: str = "workspaces/v5_object_memory_sam"
    seed: int = 1337

    @classmethod
    def from_dict(cls, payload: dict) -> "V5Config":
        def _build_dc(dc_cls, data):
            data = data or {}
            valid = {f.name for f in fields(dc_cls)}
            filtered = {k: v for k, v in data.items() if k in valid}
            return dc_cls(**filtered)

        return cls(
            model=_build_dc(V5ModelConfig, payload.get("model", {})),
            sam=_build_dc(V5SAMConfig, payload.get("sam", {})),
            loss=_build_dc(V5LossConfig, payload.get("loss", {})),
            data=payload.get("data", {}),
            optim=payload.get("optim", {}),
            logging=payload.get("logging", {}),
            validation=payload.get("validation", {}),
            ckpt=payload.get("ckpt", {}),
            viz=payload.get("viz", {}),
            runtime=payload.get("runtime", {}),
            teacher=payload.get("teacher", {}),
            workspace=payload.get("workspace", "workspaces/v5_object_memory_sam"),
            seed=payload.get("seed", 1337),
        )
