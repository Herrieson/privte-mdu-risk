"""Backward-compatible exports for PriVTE algorithms.

New code should import from `privte_pipeline.algorithms` or concrete modules
under `privte_pipeline.algorithms`. This file is intentionally thin so future
algorithm implementations do not accumulate here.
"""

from __future__ import annotations

from .algorithms import available_extractors, build_extractor
from .algorithms.behavior_v1 import PriVTEBehaviorV1Extractor
from .algorithms.common import (
    collect_clips,
    count_available,
    count_nonempty_json,
    duration_bucket,
    evenly_sample,
    fps_bucket,
    probe_mp4_box_metadata,
    resolution_bucket,
    round_mb,
    safe_ratio,
    size_bucket,
    video_file_entries,
)
from .algorithms.flowlite import PriVTEFlowLiteExtractor
from .algorithms.manifest_only import ManifestOnlyExtractor
from .algorithms.simple_video_quality import SimpleVideoQualityExtractor
from .core import EvidenceExtractor

__all__ = [
    "EvidenceExtractor",
    "ManifestOnlyExtractor",
    "SimpleVideoQualityExtractor",
    "PriVTEFlowLiteExtractor",
    "PriVTEBehaviorV1Extractor",
    "available_extractors",
    "build_extractor",
    "collect_clips",
    "count_available",
    "count_nonempty_json",
    "duration_bucket",
    "evenly_sample",
    "fps_bucket",
    "probe_mp4_box_metadata",
    "resolution_bucket",
    "round_mb",
    "safe_ratio",
    "size_bucket",
    "video_file_entries",
]
