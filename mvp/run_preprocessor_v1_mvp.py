#!/usr/bin/env python3
"""Run the PriVTE preprocessor v1 quickstart."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = REPO_ROOT / "pipeline"
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from privte_pipeline.run import print_run_result, write_evidence_run  # noqa: E402


DEFAULT_CONFIG = Path("configs/algorithms/privte_preprocessor.v1.json")


def read_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PriVTE preprocessor v1 calibrated frame-proxy demo."
    )
    parser.add_argument(
        "--person-manifest",
        type=Path,
        default=Path(
            "data/manifests/"
            "internal_person_manifest.6_1data.v0.labeled.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/quickstart/6_1data_privte_preprocessor_v1"),
    )
    parser.add_argument("--subset-name", default="6.1data")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output directory exists: {output_dir}. Use --overwrite to replace files."
        )

    result = write_evidence_run(
        person_manifest=args.person_manifest,
        output_dir=output_dir,
        subset_name=args.subset_name,
        extractor_name="privte_preprocessor_v1",
        extractor_config=read_config(args.config),
        evidence_stem="preprocessor_v1_evidence",
        report_stem="preprocessor_v1_report",
    )
    print_run_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
