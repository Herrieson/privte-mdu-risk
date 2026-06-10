#!/usr/bin/env python3
"""Refresh LLM evidence package and rendered text in an existing evidence JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from pipeline.privte_pipeline.llm_package import build_llm_evidence_package  # noqa: E402
from pipeline.privte_pipeline.renderers import build_text_evidence  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output JSONL.",
    )
    return parser.parse_args()


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


def refresh_record(record: dict[str, Any]) -> dict[str, Any]:
    if "sample_id" not in record:
        raise ValueError("record is missing sample_id")
    if "evidence" not in record:
        raise ValueError(f"{record['sample_id']}: record is missing evidence")
    refreshed = dict(record)
    refreshed["llm_evidence_package"] = build_llm_evidence_package(
        refreshed["sample_id"],
        refreshed["evidence"],
    )
    refreshed["text_evidence"] = build_text_evidence(refreshed)
    return refreshed


def main() -> int:
    args = parse_args()
    if args.output_jsonl.exists() and not args.overwrite:
        raise SystemExit(
            f"output already exists: {args.output_jsonl} (pass --overwrite to replace)"
        )
    records = [refresh_record(record) for record in read_jsonl(args.input_jsonl)]
    write_jsonl(args.output_jsonl, records)
    print(f"records={len(records)}")
    print(f"output={args.output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
