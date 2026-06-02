"""Base contracts for replaceable PriVTE algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class EvidenceExtractor(ABC):
    """Base interface for PriVTE evidence extraction algorithms.

    Algorithm modules should implement this interface and return the standard
    PriVTE evidence contract. The pipeline then handles label separation,
    LLM-package assembly, rendering, and writing.
    """

    name = "base"
    version = "v0"
    feature_schema_version = "feature_schema.v0"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "feature_schema_version": self.feature_schema_version,
            "config": self.config,
        }

    @abstractmethod
    def extract(self, person_record: dict[str, Any]) -> dict[str, Any]:
        """Return privacy-filtered evidence for one person record."""
