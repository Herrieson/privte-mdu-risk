"""Evidence record assembly for the PriVTE evidence pipeline."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .extractors import EvidenceExtractor
from .renderers import build_text_evidence


def target_label_from_person_record(person_record: dict[str, Any]) -> dict[str, Any]:
    label = person_record.get("label", {})
    return {
        "risk_level": label.get("risk_level"),
        "available": label.get("available", False),
        "label_source": label.get("label_source"),
        "aggregation_method": label.get("aggregation_method"),
        "risk_label_counts": label.get("risk_label_counts", {}),
    }


def build_evidence_record(
    person_record: dict[str, Any],
    subset_name: str,
    extractor: EvidenceExtractor,
) -> dict[str, Any]:
    sample_id = f"evidence_{person_record['person_uid']}"
    extracted_evidence = extractor.extract(person_record)
    record = {
        "schema_version": "mdu_risktext_evidence_pipeline.v0",
        "sample_id": sample_id,
        "source": {
            "source_subset": subset_name,
            "source_person_uid": person_record["person_uid"],
        },
        "split": person_record.get("split"),
        "target_label": target_label_from_person_record(person_record),
        "input_policy": {
            "input_type": "text_only_privte_evidence",
            "uses_raw_video": False,
            "uses_raw_images": False,
            "uses_raw_audio": False,
            "uses_questionnaire_answers_as_input": False,
            "uses_exact_heart_rate_as_input": False,
            "uses_app_names_as_input": False,
        },
        "evidence": extracted_evidence,
    }
    record["text_evidence"] = build_text_evidence(record)
    return record


def build_report(
    records: list[dict[str, Any]],
    *,
    subset_name: str,
    output_jsonl: Path,
    extractor: EvidenceExtractor,
) -> dict[str, Any]:
    label_counts = Counter(
        record["target_label"].get("risk_level") or "<missing>" for record in records
    )
    return {
        "schema_version": "privte_pipeline_report.v0",
        "source_subset": subset_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_jsonl": output_jsonl.as_posix(),
        "extractor": extractor.metadata,
        "records": len(records),
        "target_label_counts": dict(label_counts),
        "records_with_video": sum(
            1 for record in records if record["evidence"]["modality_availability"]["has_video"]
        ),
        "records_with_heart_rate": sum(
            1
            for record in records
            if record["evidence"]["modality_availability"]["has_heart_rate"]
        ),
        "records_with_questionnaire": sum(
            1
            for record in records
            if record["evidence"]["modality_availability"]["has_questionnaire"]
        ),
        "scope_note": "Extractor-swappable PriVTE evidence pipeline. The selected extractor defines the current feature depth.",
    }
