#!/usr/bin/env python3
"""CLI for building PriVTE evidence packages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from privte_pipeline.algorithms import available_extractors
from privte_pipeline.run import print_run_result, write_evidence_run


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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pipeline/default"))
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
    result = write_evidence_run(
        person_manifest=args.person_manifest,
        output_dir=args.output_dir,
        subset_name=args.subset_name,
        extractor_name=args.extractor,
        extractor_config=read_extractor_config(args.extractor_config),
    )
    print_run_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
