"""Reusable evidence-run orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .algorithms import build_extractor
from .evidence import build_evidence_record, build_report
from .io import read_jsonl, write_json, write_jsonl, write_text_files


def subset_slug(subset_name: str) -> str:
    return subset_name.replace(".", "_").replace("-", "_")


def write_evidence_run(
    *,
    person_manifest: Path,
    output_dir: Path,
    subset_name: str,
    extractor_name: str,
    extractor_config: dict[str, Any] | None = None,
    evidence_stem: str = "evidence",
    report_stem: str = "pipeline_report",
) -> dict[str, Any]:
    """Build and write one person-level evidence run."""

    if not person_manifest.exists():
        raise FileNotFoundError(f"Person manifest not found: {person_manifest}")

    extractor = build_extractor(extractor_name, extractor_config)
    person_records = read_jsonl(person_manifest)
    evidence_records = [
        build_evidence_record(person_record, subset_name, extractor)
        for person_record in person_records
    ]

    slug = subset_slug(subset_name)
    output_jsonl = output_dir / f"{evidence_stem}.{slug}.jsonl"
    output_report = output_dir / f"{report_stem}.{slug}.json"
    text_dir = output_dir / "evidence_text"

    write_jsonl(output_jsonl, evidence_records)
    write_json(
        output_report,
        build_report(
            evidence_records,
            subset_name=subset_name,
            output_jsonl=output_jsonl,
            extractor=extractor,
        ),
    )
    write_text_files(evidence_records, text_dir)

    return {
        "extractor": extractor,
        "output_jsonl": output_jsonl,
        "output_report": output_report,
        "text_dir": text_dir,
        "records": len(evidence_records),
    }


def print_run_result(result: dict[str, Any]) -> None:
    extractor = result["extractor"]
    print(f"extractor={extractor.name}:{extractor.version}")
    print(f"output_jsonl={result['output_jsonl']}")
    print(f"output_report={result['output_report']}")
    print(f"text_dir={result['text_dir']}")
    print(f"records={result['records']}")
