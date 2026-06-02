#!/usr/bin/env python3
"""Run the simple PriVTE algorithm MVP."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
sys.path.insert(0, PIPELINE_DIR.as_posix())

from privte_pipeline.evidence import build_evidence_record, build_report  # noqa: E402
from privte_pipeline.extractors import build_extractor  # noqa: E402
from privte_pipeline.io import read_jsonl, write_json, write_jsonl, write_text_files  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the PriVTE simple video-quality MVP."
    )
    parser.add_argument(
        "--person-manifest",
        type=Path,
        default=Path("data/manifests/internal_person_manifest.6_1data.v0.labeled.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mvp/outputs/6_1data_simple_video_quality"),
    )
    parser.add_argument("--subset-name", default="6.1data")
    parser.add_argument(
        "--max-metadata-clips",
        type=int,
        default=16,
        help="Number of videos to probe per participant. Use --metadata-all for all clips.",
    )
    parser.add_argument(
        "--metadata-all",
        action="store_true",
        help="Probe all video files instead of a deterministic sample.",
    )
    parser.add_argument(
        "--disable-opencv",
        action="store_true",
        help="Skip OpenCV metadata probing and use file-level features only.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.person_manifest.exists():
        raise SystemExit(f"Person manifest not found: {args.person_manifest}")

    extractor_config = {
        "max_metadata_clips": 0 if args.metadata_all else args.max_metadata_clips,
        "disable_opencv": args.disable_opencv,
    }
    extractor = build_extractor("simple_video_quality", extractor_config)
    person_records = read_jsonl(args.person_manifest)
    evidence_records = [
        build_evidence_record(person_record, args.subset_name, extractor)
        for person_record in person_records
    ]

    subset_slug = args.subset_name.replace(".", "_").replace("-", "_")
    output_jsonl = args.output_dir / f"simple_mvp_evidence.{subset_slug}.jsonl"
    output_report = args.output_dir / f"simple_mvp_report.{subset_slug}.json"
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
