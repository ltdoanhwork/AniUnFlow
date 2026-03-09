"""
V4 Configuration with Ablation Toggles
======================================
All SAM components can be enabled/disabled via config.
"""
from dataclasses import dataclass, field, fields
from typing import List, Optional


@dataclass
class SAMConfig:
    """SAM integration configuration with ablation toggles."""
    
    # Master switch
    enabled: bool = True
    
    # === Component Toggles (for ablation) ===
    use_encoder_features: bool = True   # Extract SAM ViT features
    use_mask_guidance: bool = True      # Use segmentation masks
    
    # === Feature Integration Modes ===
    feature_concat: bool = True         # Concatenate SAM features
    attention_bias: bool = True         # Same-segment attention boost
    cost_modulation: bool = True        # Boundary-aware cost volume
    object_pooling: bool = True         # Segment-level tokens
    
    # === Encoder Settings ===
    encoder_checkpoint: str = "models/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    encoder_config: str = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    encoder_freeze: bool = True         # Freeze SAM encoder weights
    feature_dim: int = 256              # SAM feature dimension
    feature_scales: List[int] = field(default_factory=lambda: [8, 16])
    
    # === Mask Settings ===
    num_segments: int = 32              # Max segments per frame
    mask_cache_dir: Optional[str] = None  # Precomputed masks directory
    
    # === UnSAMFlow-style mask-correlation branch ===
    add_mask_corr: bool = False
    mask_corr_aggregation: str = "concat"  # concat | residual
    mask_corr_weight: float = 1.0
    mask_corr_min_pixels: int = 8
    matcher_topk: int = 96
    segment_cross_attn_every: int = 2
    mask_corr_weight_init: float = 0.5
    boundary_gate_strength: float = 0.3


@dataclass
class LossConfig:
    """Loss weights with ablation toggles."""
    
    # === Base Unsupervised Losses ===
    photo: float = 1.0                  # Photometric loss
    photo_ssim_alpha: float = 0.85      # SSIM weight in photo loss
    smooth: float = 0.02                # Edge-aware smoothness
    consistency: float = 0.02           # Forward-backward consistency
    
    # === Anti-Collapse ===
    mag_reg: float = 0.05               # Flow magnitude regularization
    min_flow_mag: float = 0.5           # Target minimum magnitude
    warmup_steps: int = 1000            # Steps before occlusion masking
    disable_occ_during_warmup: bool = False
    
    # === SAM-Guided Losses (toggleable) ===
    homography_smooth: float = 0.0      # 0=disabled, >0=enabled with weight
    boundary_sharpness: float = 0.0     # Align flow gradients with boundaries
    object_variance: float = 0.0        # Penalize intra-segment flow variance
    mask_flow_consistency: float = 0.0  # Warped mask alignment
    segment_consistency: float = 0.0    # Within-segment flow coherence
    boundary_aware_smooth: float = 0.0  # Suppress smoothness at boundaries


@dataclass 
class ModelConfig:
    """Model architecture configuration."""

    # Backbone mode
    backbone: str = "v4"  # v4 | v4_5_matcher_lcm | v4_5_hybrid_sam
    
    # Encoder
    enc_channels: int = 32
    enc_out_channels: int = 64
    
    # Tokenizer/Matcher
    token_dim: int = 128
    num_heads: int = 4
    
    # LCM (Latent Cost Memory)
    lcm_depth: int = 4
    lcm_heads: int = 4
    
    # GTR (Global Token Refinement)
    gtr_depth: int = 1
    gtr_heads: int = 4
    
    # Decoder
    iters_per_level: int = 3

    # V4.5 iterative refiner
    refiner_iters: int = 10
    use_convex_upsampler: bool = True
    
    # Dropout
    dropout: float = 0.0


@dataclass
class V4Config:
    """Complete V4 configuration."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    sam: SAMConfig = field(default_factory=SAMConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    
    # Training Loop Configs (dicts for flexibility)
    data: dict = field(default_factory=dict)
    optim: dict = field(default_factory=dict)
    logging: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)
    ckpt: dict = field(default_factory=dict)
    viz: dict = field(default_factory=dict)
    workspace: str = "workspaces/default"
    
    # Training
    seed: int = 1337
    
    @classmethod
    def from_dict(cls, d: dict) -> "V4Config":
        """Create config from dictionary (e.g., from YAML)."""
        def _build_dc(dc_cls, payload):
            payload = payload or {}
            valid = {f.name for f in fields(dc_cls)}
            filtered = {k: v for k, v in payload.items() if k in valid}
            return dc_cls(**filtered)

        model = _build_dc(ModelConfig, d.get("model", {}))
        sam = _build_dc(SAMConfig, d.get("sam", {}))
        loss = _build_dc(LossConfig, d.get("loss", {}))
        
        return cls(
            model=model, 
            sam=sam, 
            loss=loss, 
            seed=d.get("seed", 1337),
            data=d.get("data", {}),
            optim=d.get("optim", {}),
            logging=d.get("logging", {}),
            validation=d.get("validation", {}),
            ckpt=d.get("ckpt", {}),
            viz=d.get("viz", {}),
            workspace=d.get("workspace", "workspaces/default"),
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        from dataclasses import asdict
        return asdict(self)


# === Preset Configs for Ablation ===

def get_baseline_config() -> V4Config:
    """Baseline: No SAM integration."""
    cfg = V4Config()
    cfg.sam.enabled = False
    return cfg


def get_masks_only_config() -> V4Config:
    """Ablation: Use masks but not encoder features."""
    cfg = V4Config()
    cfg.sam.use_encoder_features = False
    cfg.sam.use_mask_guidance = True
    return cfg


def get_encoder_only_config() -> V4Config:
    """Ablation: Use encoder features but not masks."""
    cfg = V4Config()
    cfg.sam.use_encoder_features = True
    cfg.sam.use_mask_guidance = False
    return cfg


def get_full_config() -> V4Config:
    """Full integration: All SAM components enabled."""
    cfg = V4Config()
    # Enable all SAM-guided losses
    cfg.loss.homography_smooth = 0.15
    cfg.loss.boundary_sharpness = 0.05
    cfg.loss.object_variance = 0.08
    cfg.loss.segment_consistency = 0.03
    cfg.loss.boundary_aware_smooth = 0.08
    return cfg
