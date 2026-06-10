#!/usr/bin/env python3
"""Add PriVTE-Trace v1 fields to an existing Behavior v3 temporal evidence JSONL."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
if PIPELINE_DIR.as_posix() not in sys.path:
    sys.path.insert(0, PIPELINE_DIR.as_posix())

from privte_pipeline.algorithms.trace_v1 import PriVTETraceV1Extractor  # noqa: E402
from privte_pipeline.llm_package import build_llm_evidence_package  # noqa: E402
from privte_pipeline.renderers import build_text_evidence  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/algorithms/privte_trace.v1.json"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL record") from exc
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            file.write("\n")
    tmp_path.replace(path)


def add_trace_fields(
    record: dict[str, Any],
    extractor: PriVTETraceV1Extractor,
) -> dict[str, Any]:
    refreshed = copy.deepcopy(record)
    evidence = refreshed["evidence"]
    feature_blocks = evidence["feature_blocks"]
    video = feature_blocks["video_proxy_summary"]
    temporal_summary = video.get("temporal_sequence_summary")
    if not temporal_summary:
        raise ValueError(f"{refreshed['sample_id']}: missing temporal_sequence_summary")

    aggregate_for_trace = dict(video)
    aggregate_for_trace["quality_summary"] = feature_blocks.get("quality_summary", {})
    trace_summary = extractor._build_trace_risk_summary(aggregate_for_trace)
    video["privte_trace_features"] = {
        "schema_note": (
            "Trace v1 fields derived from existing Behavior v3 temporal evidence "
            "without rerunning raw-video feature extraction."
        ),
        "derived_from_existing_temporal_evidence": True,
        "trace_risk_summary": trace_summary,
    }
    video["trace_risk_summary"] = trace_summary
    video["trace_behavior_narrative"] = extractor._build_trace_narrative(trace_summary)
    refreshed["llm_evidence_package"] = build_llm_evidence_package(
        refreshed["sample_id"],
        evidence,
    )
    refreshed["text_evidence"] = build_text_evidence(refreshed)
    return refreshed


def main() -> int:
    args = parse_args()
    if args.output_jsonl.exists() and not args.overwrite:
        raise SystemExit(
            f"output already exists: {args.output_jsonl} (pass --overwrite to replace)"
        )
    extractor = PriVTETraceV1Extractor(read_config(args.config))
    records = [add_trace_fields(record, extractor) for record in read_jsonl(args.input_jsonl)]
    write_jsonl(args.output_jsonl, records)
    print(f"records={len(records)}")
    print(f"output={args.output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
