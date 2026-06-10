#!/usr/bin/env python3
"""Run the PriVTE-Behavior v3 temporal MVP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
sys.path.insert(0, PIPELINE_DIR.as_posix())

from privte_pipeline.run import print_run_result, write_evidence_run  # noqa: E402


DEFAULT_CONFIG = Path("configs/algorithms/privte_behavior.v3_temporal.json")


def read_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the PriVTE-Behavior v3 temporal episode MVP."
    )
    parser.add_argument(
        "--person-manifest",
        type=Path,
        default=Path("data/manifests/internal_person_manifest.6_1data.v0.labeled.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/quickstart/6_1data_privte_behavior_v3_temporal"),
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
        "--max-temporal-steps",
        type=int,
        default=None,
        help="Override the maximum number of selected temporal episodes per participant.",
    )
    parser.add_argument(
        "--process-all-clips",
        action="store_true",
        help="Analyze all available video clips for each participant.",
    )
    parser.add_argument(
        "--require-yolo",
        action="store_true",
        help="Fail if Ultralytics YOLO cannot be loaded.",
    )
    parser.add_argument(
        "--disable-yolo",
        action="store_true",
        help="Use MediaPipe plus heuristic device/screen proxy without YOLO.",
    )
    parser.add_argument(
        "--allow-metadata-fallback",
        action="store_true",
        help="Do not fail when behavior dependencies are unavailable.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.person_manifest.exists():
        raise SystemExit(f"Person manifest not found: {args.person_manifest}")
    if not args.config.exists():
        raise SystemExit(f"Behavior v3 temporal config not found: {args.config}")

    extractor_config = read_config(args.config)
    if args.max_video_clips is not None:
        extractor_config["max_video_clips"] = args.max_video_clips
    if args.frames_per_clip is not None:
        extractor_config["frames_per_clip"] = args.frames_per_clip
    if args.max_temporal_steps is not None:
        extractor_config["max_temporal_steps"] = args.max_temporal_steps
    if args.process_all_clips:
        extractor_config["max_video_clips"] = 0
    if args.require_yolo:
        extractor_config["require_yolo"] = True
    if args.disable_yolo:
        extractor_config["disable_yolo"] = True
        extractor_config["device_backend"] = "heuristic"
    if args.allow_metadata_fallback:
        extractor_config["require_behavior_dependencies"] = False
        extractor_config["require_yolo"] = False

    try:
        result = write_evidence_run(
            person_manifest=args.person_manifest,
            output_dir=args.output_dir,
            subset_name=args.subset_name,
            extractor_name="privte_behavior_v3_temporal",
            extractor_config=extractor_config,
            evidence_stem="behavior_v3_temporal_mvp_evidence",
            report_stem="behavior_v3_temporal_mvp_report",
        )
    except RuntimeError as exc:
        raise SystemExit(
            f"{exc}\n"
            "For a plumbing check without behavior models, rerun with "
            "--allow-metadata-fallback. For real Behavior v3 temporal output, "
            "provide the configured MediaPipe task models under models/mediapipe/."
        ) from exc
    print_run_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
