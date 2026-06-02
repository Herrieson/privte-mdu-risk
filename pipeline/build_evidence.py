#!/usr/bin/env python3
"""CLI for building PriVTE evidence packages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from privte_pipeline.evidence import build_evidence_record, build_report
from privte_pipeline.extractors import available_extractors, build_extractor
from privte_pipeline.io import read_jsonl, write_json, write_jsonl, write_text_files


def read_extractor_config(path: Path | None) -> dict:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a MDU-RiskText evidence JSONL from a person manifest."
    )
    parser.add_argument("--person-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("pipeline/outputs/default"))
    parser.add_argument("--subset-name", default="dataset")
    parser.add_argument(
        "--extractor",
        default="manifest_only",
        choices=available_extractors(),
        help="Evidence extractor implementation to use.",
    )
    parser.add_argument(
        "--extractor-config",
        type=Path,
        default=None,
        help="Optional JSON config passed to the extractor.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.person_manifest.exists():
        raise SystemExit(f"Person manifest not found: {args.person_manifest}")

    extractor = build_extractor(args.extractor, read_extractor_config(args.extractor_config))
    person_records = read_jsonl(args.person_manifest)
    evidence_records = [
        build_evidence_record(person_record, args.subset_name, extractor)
        for person_record in person_records
    ]

    subset_slug = args.subset_name.replace(".", "_").replace("-", "_")
    output_jsonl = args.output_dir / f"evidence.{subset_slug}.jsonl"
    output_report = args.output_dir / f"pipeline_report.{subset_slug}.json"
    text_dir = args.output_dir / "evidence_text"

    write_jsonl(output_jsonl, evidence_records)
    write_json(
        output_report,
        build_report(
            evidence_records,
            subset_name=args.subset_name,
            output_jsonl=output_jsonl,
            extractor=extractor,
        ),
    )
    write_text_files(evidence_records, text_dir)

    print(f"extractor={extractor.name}:{extractor.version}")
    print(f"output_jsonl={output_jsonl}")
    print(f"output_report={output_report}")
    print(f"text_dir={text_dir}")
    print(f"records={len(evidence_records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
