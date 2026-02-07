"""
AniFlowFormerT V4: Full SAM Integration
========================================
Clean, modular architecture with ablation-friendly config.
"""
from .config import V4Config, SAMConfig, LossConfig
from .model import AniFlowFormerTV4

__all__ = [
    "V4Config",
    "SAMConfig", 
    "LossConfig",
    "AniFlowFormerTV4",
]
