#!/usr/bin/env python3
"""Build one internal PriVTE manifest from multiple flat data roots."""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from build_dataset_manifest import (
    assign_uids,
    build_clip_record,
    build_dataset_manifest,
    build_person_records,
    parse_segment_dir_name,
    rel_path,
    safe_slug,
    write_json,
    write_jsonl,
)


def unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        normalized = path.resolve()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(path)
    return unique


def collect_segment_dirs(
    *,
    data_roots: list[Path],
    repo_root: Path,
) -> tuple[list[tuple[Path, dict[str, Any], Path]], list[str], list[dict[str, Any]]]:
    parsed_by_dir: list[tuple[Path, dict[str, Any], Path]] = []
    invalid_dirs: list[str] = []
    root_summaries: list[dict[str, Any]] = []

    for data_root in data_roots:
        segment_dirs = sorted(
            [path for path in data_root.iterdir() if path.is_dir()],
            key=lambda path: path.name,
        )
        root_valid = 0
        root_invalid = 0
        for segment_dir in segment_dirs:
            parsed = parse_segment_dir_name(segment_dir.name)
            if parsed is None:
                invalid_dirs.append(rel_path(segment_dir, repo_root))
                root_invalid += 1
                continue
            parsed_by_dir.append((segment_dir, parsed, data_root))
            root_valid += 1

        root_summaries.append(
            {
                "data_root": rel_path(data_root, repo_root),
                "valid_clip_dirs": root_valid,
                "invalid_dirs": root_invalid,
            }
        )

    parsed_by_dir.sort(
        key=lambda item: (
            item[1]["person_raw_id"],
            item[1]["session_raw_key"],
            item[1]["clip_index"],
            rel_path(item[0], repo_root),
        )
    )
    return parsed_by_dir, invalid_dirs, root_summaries


def build_combined_clip_records(
    *,
    parsed_by_dir: list[tuple[Path, dict[str, Any], Path]],
    source_subset: str,
    repo_root: Path,
    hash_json: bool,
    hash_video: bool,
) -> list[dict[str, Any]]:
    person_uids = assign_uids([parsed["person_raw_id"] for _, parsed, _ in parsed_by_dir])
    session_uid_by_key: dict[str, str] = {}
    session_counts_by_person: dict[str, int] = defaultdict(int)
    clip_counts_by_person: dict[str, int] = defaultdict(int)
    clip_records: list[dict[str, Any]] = []

    for segment_dir, parsed, data_root in parsed_by_dir:
        person_uid = person_uids[parsed["person_raw_id"]]
        if parsed["session_raw_key"] not in session_uid_by_key:
            session_counts_by_person[person_uid] += 1
            session_uid_by_key[parsed["session_raw_key"]] = (
                f"{person_uid}_s{session_counts_by_person[person_uid]:04d}"
            )
        clip_counts_by_person[person_uid] += 1
        clip_uid = f"{person_uid}_c{clip_counts_by_person[person_uid]:06d}"

        record = build_clip_record(
            segment_dir=segment_dir,
            parsed=parsed,
            source_subset=source_subset,
            person_uid=person_uid,
            session_uid=session_uid_by_key[parsed["session_raw_key"]],
            clip_uid=clip_uid,
            repo_root=repo_root,
            hash_json=hash_json,
            hash_video=hash_video,
        )
        record["source_data_root"] = rel_path(data_root, repo_root)
        clip_records.append(record)

    return clip_records


def common_data_root(data_roots: list[Path]) -> Path:
    resolved = [path.resolve().as_posix() for path in data_roots]
    return Path(os.path.commonpath(resolved))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one internal PriVTE manifest from multiple flat data roots."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        action="append",
        required=True,
        help="Flat raw data root. Pass this option multiple times for a combined manifest.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--subset-name", default="all_current")
    parser.add_argument("--manifest-version", default="v0")
    parser.add_argument(
        "--no-json-hash",
        action="store_true",
        help="Skip SHA-256 hashes for JSON files.",
    )
    parser.add_argument(
        "--hash-video",
        action="store_true",
        help="Hash video files too. This can be slow for large raw datasets.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd()
    data_roots = unique_paths(args.data_root)
    for data_root in data_roots:
        if not data_root.exists():
            print(f"Data root not found: {data_root}", file=sys.stderr)
            return 1
        if not data_root.is_dir():
            print(f"Data root is not a directory: {data_root}", file=sys.stderr)
            return 1

    subset_name = args.subset_name
    subset_slug = safe_slug(subset_name)
    version = args.manifest_version
    output_dir = args.output_dir

    dataset_path = output_dir / f"dataset_manifest.{subset_slug}.{version}.json"
    person_path = output_dir / f"internal_person_manifest.{subset_slug}.{version}.jsonl"
    clip_path = output_dir / f"internal_clip_manifest.{subset_slug}.{version}.jsonl"
    output_paths = {
        "dataset_manifest": dataset_path,
        "person_manifest": person_path,
        "clip_manifest": clip_path,
    }

    parsed_by_dir, invalid_dirs, root_summaries = collect_segment_dirs(
        data_roots=data_roots,
        repo_root=repo_root,
    )
    clip_records = build_combined_clip_records(
        parsed_by_dir=parsed_by_dir,
        source_subset=subset_name,
        repo_root=repo_root,
        hash_json=not args.no_json_hash,
        hash_video=args.hash_video,
    )
    person_records = build_person_records(clip_records)
    dataset_manifest = build_dataset_manifest(
        data_root=common_data_root(data_roots),
        repo_root=repo_root,
        subset_name=subset_name,
        manifest_version=version,
        person_records=person_records,
        clip_records=clip_records,
        invalid_dirs=invalid_dirs,
        output_paths=output_paths,
    )
    dataset_manifest["source_data_roots"] = [
        rel_path(data_root, repo_root) for data_root in data_roots
    ]
    dataset_manifest["source_root_summaries"] = root_summaries
    dataset_manifest["counts"]["source_roots"] = len(data_roots)

    write_json(dataset_path, dataset_manifest)
    write_jsonl(person_path, person_records)
    write_jsonl(clip_path, clip_records)

    print(f"dataset_manifest={dataset_path}")
    print(f"person_manifest={person_path}")
    print(f"clip_manifest={clip_path}")
    print(f"source_roots={len(data_roots)}")
    print(f"persons={len(person_records)}")
    print(f"sessions={dataset_manifest['counts']['sessions']}")
    print(f"clips={len(clip_records)}")
    print(f"invalid_directories={len(invalid_dirs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
