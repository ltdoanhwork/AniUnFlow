# file: losses/__init__.py
"""
Loss Functions for Unsupervised Optical Flow Training
======================================================
"""
from .unsup_flow_losses import UnsupervisedFlowLoss
from .segment_aware_losses import (
    SegmentConsistencyFlowLoss,
    BoundaryAwareSmoothnessLoss,
    TemporalMemoryRegularization,
    SegmentAwareLossModule,
    build_segment_aware_losses,
)

__all__ = [
    "UnsupervisedFlowLoss",
    "SegmentConsistencyFlowLoss",
    "BoundaryAwareSmoothnessLoss",
    "TemporalMemoryRegularization",
    "SegmentAwareLossModule",
    "build_segment_aware_losses",
]
