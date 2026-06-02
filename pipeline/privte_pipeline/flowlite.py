"""Backward-compatible FlowLite import.

The implementation now lives in `privte_pipeline.algorithms.flowlite`.
"""

from __future__ import annotations

from .algorithms.flowlite import PriVTEFlowLiteExtractor

__all__ = ["PriVTEFlowLiteExtractor"]
