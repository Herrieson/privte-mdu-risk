#!/usr/bin/env python3
"""Run the PriVTE-FlowLite frame-level MVP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
sys.path.insert(0, PIPELINE_DIR.as_posix())

from privte_pipeline.run import print_run_result, write_evidence_run  # noqa: E402


DEFAULT_CONFIG = Path("configs/algorithms/privte_flowlite.v0.json")


def read_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the PriVTE-FlowLite frame-level MVP."
    )
    parser.add_argument(
        "--person-manifest",
        type=Path,
        default=Path("data/manifests/internal_person_manifest.6_1data.v0.labeled.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/quickstart/6_1data_privte_flowlite"),
    )
    parser.add_argument("--subset-name", default="6.1data")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--max-video-clips",
        type=int,
        default=None,
        help="Override config max_video_clips per participant.",
    )
    parser.add_argument(
        "--frames-per-clip",
        type=int,
        default=None,
        help="Override config sampled frames per selected clip.",
    )
    parser.add_argument(
        "--process-all-clips",
        action="store_true",
        help="Analyze all available video clips for each participant.",
    )
    parser.add_argument(
        "--allow-metadata-fallback",
        action="store_true",
        help="Do not fail when OpenCV is unavailable; emit metadata-only fallback evidence.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.person_manifest.exists():
        raise SystemExit(f"Person manifest not found: {args.person_manifest}")
    if not args.config.exists():
        raise SystemExit(f"FlowLite config not found: {args.config}")

    extractor_config = read_config(args.config)
    if args.max_video_clips is not None:
        extractor_config["max_video_clips"] = args.max_video_clips
    if args.frames_per_clip is not None:
        extractor_config["frames_per_clip"] = args.frames_per_clip
    if args.process_all_clips:
        extractor_config["max_video_clips"] = 0
    if args.allow_metadata_fallback:
        extractor_config["require_opencv"] = False

    result = write_evidence_run(
        person_manifest=args.person_manifest,
        output_dir=args.output_dir,
        subset_name=args.subset_name,
        extractor_name="privte_flowlite",
        extractor_config=extractor_config,
        evidence_stem="flowlite_mvp_evidence",
        report_stem="flowlite_mvp_report",
    )
    print_run_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
