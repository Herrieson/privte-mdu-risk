#!/usr/bin/env python3
"""Audit PriVTE evidence JSONL for obvious public-input privacy leaks."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


BLOCKED_TEXT_PATTERNS = {
    "raw_data_path": re.compile(r"data/[^\s\"']+|/home/|\\\\"),
    "video_directory_id": re.compile(r"\b\d{6}_VID_\d{4}_\d{2}_\d{2}_"),
    "raw_clip_filename": re.compile(r"\bclip_\d+\.mp4\b"),
    "image_filename": re.compile(r"\.(png|jpg|jpeg|bmp)\b", re.IGNORECASE),
}

MUST_BE_FALSE_FLAGS = {
    "raw_video_included",
    "raw_images_included",
    "raw_audio_included",
    "ocr_text_included",
    "asr_text_included",
    "exact_timestamps_included",
    "high_frequency_coordinates_included",
    "face_embeddings_included",
    "screen_content_included",
    "questionnaire_answers_included",
    "exact_heart_rate_values_included",
    "app_names_included",
    "raw_paths_included",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit PriVTE public evidence fields for blocked privacy leaks."
    )
    parser.add_argument("evidence_jsonl", type=Path)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=10,
        help="Maximum violation examples to print.",
    )
    return parser.parse_args()


def iter_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def public_payload(record: dict[str, Any]) -> dict[str, Any]:
    evidence = record.get("evidence", {})
    feature_blocks = evidence.get("feature_blocks", {})
    return {
        "sample_id": record.get("sample_id"),
        "llm_evidence_package": record.get("llm_evidence_package", {}),
        "text_evidence": record.get("text_evidence", ""),
        "preprocessor_evidence": feature_blocks.get("preprocessor_evidence", {}),
    }


def audit_text(sample_id: str, payload: dict[str, Any]) -> list[str]:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    violations = []
    for name, pattern in BLOCKED_TEXT_PATTERNS.items():
        match = pattern.search(serialized)
        if match:
            violations.append(f"{sample_id}: blocked text pattern {name}: {match.group(0)}")
    return violations


def audit_privacy_flags(sample_id: str, payload: dict[str, Any]) -> list[str]:
    preprocessor = payload.get("preprocessor_evidence", {})
    privacy = preprocessor.get("privacy_processing_summary", {})
    violations = []
    for flag in MUST_BE_FALSE_FLAGS:
        if privacy.get(flag) is not False:
            violations.append(f"{sample_id}: privacy flag {flag} is not false")
    return violations


def main() -> int:
    args = parse_args()
    if not args.evidence_jsonl.exists():
        raise FileNotFoundError(args.evidence_jsonl)

    records = iter_records(args.evidence_jsonl)
    violations = []
    for record in records:
        sample_id = str(record.get("sample_id", "<missing_sample_id>"))
        payload = public_payload(record)
        violations.extend(audit_text(sample_id, payload))
        violations.extend(audit_privacy_flags(sample_id, payload))

    print(
        json.dumps(
            {
                "records": len(records),
                "violation_count": len(violations),
                "examples": violations[: args.max_examples],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
