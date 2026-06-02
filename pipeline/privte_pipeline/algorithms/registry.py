"""Algorithm registry for replaceable PriVTE extractors."""

from __future__ import annotations

from typing import Any

from ..core import EvidenceExtractor
from .flowlite import PriVTEFlowLiteExtractor
from .manifest_only import ManifestOnlyExtractor
from .simple_video_quality import SimpleVideoQualityExtractor

EXTRACTOR_REGISTRY: dict[str, type[EvidenceExtractor]] = {}


def register_extractor(extractor_cls: type[EvidenceExtractor]) -> type[EvidenceExtractor]:
    """Register an extractor class and return it for decorator-style use."""

    if extractor_cls.name in EXTRACTOR_REGISTRY:
        raise ValueError(f"Duplicate extractor name: {extractor_cls.name}")
    EXTRACTOR_REGISTRY[extractor_cls.name] = extractor_cls
    return extractor_cls


def available_extractors() -> list[str]:
    return sorted(EXTRACTOR_REGISTRY)


def build_extractor(name: str, config: dict[str, Any] | None = None) -> EvidenceExtractor:
    try:
        extractor_cls = EXTRACTOR_REGISTRY[name]
    except KeyError as exc:
        options = ", ".join(available_extractors())
        raise ValueError(f"Unknown extractor {name!r}. Available: {options}") from exc
    return extractor_cls(config)


for _extractor_cls in (
    ManifestOnlyExtractor,
    SimpleVideoQualityExtractor,
    PriVTEFlowLiteExtractor,
):
    register_extractor(_extractor_cls)
