#!/usr/bin/env python3
"""Run the simple PriVTE algorithm MVP."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
sys.path.insert(0, PIPELINE_DIR.as_posix())

from privte_pipeline.run import print_run_result, write_evidence_run  # noqa: E402


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
        default=Path("outputs/quickstart/6_1data_simple_video_quality"),
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
    extractor_config = {
        "max_metadata_clips": 0 if args.metadata_all else args.max_metadata_clips,
        "disable_opencv": args.disable_opencv,
    }
    result = write_evidence_run(
        person_manifest=args.person_manifest,
        output_dir=args.output_dir,
        subset_name=args.subset_name,
        extractor_name="simple_video_quality",
        extractor_config=extractor_config,
        evidence_stem="simple_mvp_evidence",
        report_stem="simple_mvp_report",
    )
    print_run_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
