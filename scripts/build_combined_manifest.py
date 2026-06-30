#!/usr/bin/env python3
"""Build one internal PriVTE manifest from multiple flat data roots."""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
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
    recursive: bool,
    layout: str,
) -> tuple[list[tuple[Path, dict[str, Any], Path]], list[str], list[dict[str, Any]]]:
    parsed_by_dir: list[tuple[Path, dict[str, Any], Path]] = []
    invalid_dirs: list[str] = []
    root_summaries: list[dict[str, Any]] = []

    for data_root in data_roots:
        if layout == "batch-person-segment":
            candidate_dirs = batch_person_segment_dirs(data_root)
        elif recursive:
            candidate_dirs = recursive_segment_dirs(data_root)
        else:
            candidate_dirs = [path for path in data_root.iterdir() if path.is_dir()]
        segment_dirs = sorted(candidate_dirs, key=lambda path: path.as_posix())
        root_valid = 0
        for segment_dir in segment_dirs:
            parsed = parse_segment_dir_name(segment_dir.name)
            if parsed is None:
                if not recursive:
                    invalid_dirs.append(rel_path(segment_dir, repo_root))
                continue
            parsed_by_dir.append((segment_dir, parsed, data_root))
            root_valid += 1

        root_summaries.append(
            {
                "data_root": rel_path(data_root, repo_root),
                "scan_mode": layout if layout != "auto" else "recursive" if recursive else "flat",
                "valid_clip_dirs": root_valid,
                "invalid_dirs": 0 if recursive or layout != "auto" else len(segment_dirs) - root_valid,
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


def batch_person_segment_dirs(data_root: Path) -> list[Path]:
    segment_dirs: list[Path] = []
    with os.scandir(data_root) as people:
        for person_entry in people:
            if not person_entry.is_dir():
                continue
            person_name = person_entry.name
            if person_name in {"manifests", "$RECYCLE.BIN", "System Volume Information"}:
                continue
            with os.scandir(person_entry.path) as sessions:
                for session_entry in sessions:
                    if not session_entry.is_dir():
                        continue
                    if parse_segment_dir_name(session_entry.name) is not None:
                        segment_dirs.append(Path(session_entry.path))
    return segment_dirs


def recursive_segment_dirs(data_root: Path) -> list[Path]:
    segment_dirs: list[Path] = []
    for root, dirnames, _ in os.walk(data_root):
        root_path = Path(root)
        parsed = parse_segment_dir_name(root_path.name)
        if parsed is not None:
            segment_dirs.append(root_path)
            dirnames[:] = []
            continue
        dirnames[:] = [
            name
            for name in dirnames
            if name not in {"manifests", "$RECYCLE.BIN", "System Volume Information"}
        ]
    return segment_dirs


def fast_json_info(
    path: Path,
    repo_root: Path,
    public_policy: str,
    entries: dict[str, os.DirEntry[str]],
) -> dict[str, Any]:
    entry = entries.get(path.name)
    info: dict[str, Any] = {
        "available": entry is not None,
        "path": rel_path(path, repo_root) if entry is not None else None,
        "public_policy": public_policy,
        "json_status": "not_parsed_fast_manifest_mode",
        "size_bytes": None,
        "is_empty": None,
    }
    return info


def fast_image_info(
    path: Path,
    repo_root: Path,
    public_policy: str,
    entries: dict[str, os.DirEntry[str]],
) -> dict[str, Any]:
    entry = entries.get(path.name)
    return {
        "available": entry is not None,
        "path": rel_path(path, repo_root) if entry is not None else None,
        "size_bytes": None,
        "public_policy": public_policy,
    }


def fast_video_info(
    segment_dir: Path,
    repo_root: Path,
    entries: dict[str, os.DirEntry[str]],
) -> dict[str, Any]:
    files = sorted(
        [
            (name, entry)
            for name, entry in entries.items()
            if name.lower().endswith(".mp4")
        ],
        key=lambda item: item[0],
    )
    return {
        "available": bool(files),
        "count": len(files),
        "files": [
            {
                "path": rel_path(segment_dir / name, repo_root),
                "size_bytes": entry.stat().st_size,
                "sha256": None,
                "public_policy": "exclude",
            }
            for name, entry in files
        ],
        "public_policy": "exclude_raw_video",
    }


def fast_segment_entries(segment_dir: Path) -> dict[str, os.DirEntry[str]]:
    entries: dict[str, os.DirEntry[str]] = {}
    with os.scandir(segment_dir) as iterator:
        for entry in iterator:
            if entry.is_file():
                entries[entry.name] = entry
    return entries


def build_fast_clip_record(
    *,
    segment_dir: Path,
    parsed: dict[str, Any],
    source_subset: str,
    person_uid: str,
    session_uid: str,
    clip_uid: str,
    repo_root: Path,
    hash_video: bool,
) -> dict[str, Any]:
    if hash_video:
        raise ValueError("--fast cannot be combined with --hash-video")
    entries = fast_segment_entries(segment_dir)
    return {
        "schema_version": "internal_clip_manifest.v0",
        "source_subset": source_subset,
        "person_uid": person_uid,
        "person_raw_id": parsed["person_raw_id"],
        "session_uid": session_uid,
        "session_raw_key": parsed["session_raw_key"],
        "clip_uid": clip_uid,
        "clip_index": parsed["clip_index"],
        "clip_index_raw": parsed["clip_index_raw"],
        "segment_dir_name": segment_dir.name,
        "segment_dir_path": rel_path(segment_dir, repo_root),
        "capture_date_raw": parsed["capture_date_raw"],
        "capture_time_raw": parsed["capture_time_raw"],
        "captured_at_raw": parsed["captured_at_raw"],
        "split": None,
        "label": {
            "available": False,
            "risk_level": None,
            "label_source": None,
            "dimension_labels": {
                "video": None,
                "heart_rate": None,
                "questionnaire": None,
                "app_usage": None,
            },
        },
        "video": fast_video_info(segment_dir, repo_root, entries),
        "usage": fast_json_info(
            segment_dir / "usage.json",
            repo_root,
            "coarse_category_and_duration_bin_only",
            entries,
        ),
        "heart_rate": fast_json_info(
            segment_dir / "heartRate.json",
            repo_root,
            "exclude_exact_values_use_quality_or_trend_only",
            entries,
        ),
        "questionnaire_ref": fast_json_info(
            segment_dir / "questionnaire.json",
            repo_root,
            "exclude_raw_values_person_level_only",
            entries,
        ),
        "images": {
            "questionnaire_list": fast_image_info(
                segment_dir / "questionnaire_list.png",
                repo_root,
                "exclude",
                entries,
            ),
            "heart_rate_plot": fast_image_info(
                segment_dir / "heartRate.png",
                repo_root,
                "exclude",
                entries,
            ),
        },
        "privacy_flags": {
            "contains_raw_paths": True,
            "contains_raw_person_id": True,
            "contains_exact_timestamp": True,
            "public_release_allowed": False,
        },
    }


def build_data_quality_summary(
    clip_records: list[dict[str, Any]],
) -> dict[str, Any]:
    zero_byte_video_clips = []
    missing_modality_counts: Counter[str] = Counter()
    empty_modality_counts: Counter[str] = Counter()
    clip_counts_by_root: Counter[str] = Counter()
    person_counts_by_root: dict[str, set[str]] = defaultdict(set)
    session_counts_by_root: dict[str, set[str]] = defaultdict(set)

    for record in clip_records:
        source_root = str(record.get("source_data_root", "<unknown>"))
        clip_counts_by_root[source_root] += 1
        person_counts_by_root[source_root].add(str(record.get("person_raw_id")))
        session_counts_by_root[source_root].add(str(record.get("session_raw_key")))

        video_files = record.get("video", {}).get("files", [])
        if not video_files:
            missing_modality_counts["video"] += 1
        for file_info in video_files:
            if int(file_info.get("size_bytes") or 0) == 0:
                zero_byte_video_clips.append(
                    {
                        "clip_uid": record.get("clip_uid"),
                        "person_raw_id": record.get("person_raw_id"),
                        "segment_dir_path": record.get("segment_dir_path"),
                        "video_path": file_info.get("path"),
                    }
                )

        for key in ("usage", "heart_rate", "questionnaire_ref"):
            value = record.get(key, {})
            if not value.get("available"):
                missing_modality_counts[key] += 1
            elif value.get("is_empty"):
                empty_modality_counts[key] += 1

    return {
        "zero_byte_video_clip_count": len(zero_byte_video_clips),
        "zero_byte_video_clip_examples": zero_byte_video_clips[:50],
        "missing_modality_counts": dict(missing_modality_counts),
        "empty_modality_counts": dict(empty_modality_counts),
        "source_root_counts": {
            source_root: {
                "persons": len(person_counts_by_root[source_root]),
                "sessions": len(session_counts_by_root[source_root]),
                "clips": clip_count,
            }
            for source_root, clip_count in sorted(clip_counts_by_root.items())
        },
    }


def build_combined_clip_records(
    *,
    parsed_by_dir: list[tuple[Path, dict[str, Any], Path]],
    source_subset: str,
    repo_root: Path,
    hash_json: bool,
    hash_video: bool,
    fast: bool,
    progress_every: int,
) -> list[dict[str, Any]]:
    person_uids = assign_uids([parsed["person_raw_id"] for _, parsed, _ in parsed_by_dir])
    session_uid_by_key: dict[str, str] = {}
    session_counts_by_person: dict[str, int] = defaultdict(int)
    clip_counts_by_person: dict[str, int] = defaultdict(int)
    clip_records: list[dict[str, Any]] = []

    for index, (segment_dir, parsed, data_root) in enumerate(parsed_by_dir, start=1):
        person_uid = person_uids[parsed["person_raw_id"]]
        if parsed["session_raw_key"] not in session_uid_by_key:
            session_counts_by_person[person_uid] += 1
            session_uid_by_key[parsed["session_raw_key"]] = (
                f"{person_uid}_s{session_counts_by_person[person_uid]:04d}"
            )
        clip_counts_by_person[person_uid] += 1
        clip_uid = f"{person_uid}_c{clip_counts_by_person[person_uid]:06d}"

        builder = build_fast_clip_record if fast else build_clip_record
        record = builder(
            segment_dir=segment_dir,
            parsed=parsed,
            source_subset=source_subset,
            person_uid=person_uid,
            session_uid=session_uid_by_key[parsed["session_raw_key"]],
            clip_uid=clip_uid,
            repo_root=repo_root,
            hash_video=hash_video,
            **({} if fast else {"hash_json": hash_json}),
        )
        record["source_data_root"] = rel_path(data_root, repo_root)
        clip_records.append(record)
        if progress_every > 0 and index % progress_every == 0:
            print(
                f"processed_clip_records={index}/{len(parsed_by_dir)}",
                file=sys.stderr,
                flush=True,
            )

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
    parser.add_argument(
        "--recursive",
        action="store_true",
        help=(
            "Recursively search data roots for *_VID_* segment directories. "
            "Use this for roots organized as batch/person/segment."
        ),
    )
    parser.add_argument(
        "--layout",
        choices=["auto", "batch-person-segment"],
        default="auto",
        help=(
            "Use a known directory layout. batch-person-segment means "
            "data-root/person_id/session_dir and is much faster than recursive scan."
        ),
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Build manifest from file presence and sizes only; do not parse JSON "
            "payloads. Useful for large external drives."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print progress to stderr every N clip records while building manifests.",
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
        recursive=args.recursive,
        layout=args.layout,
    )
    clip_records = build_combined_clip_records(
        parsed_by_dir=parsed_by_dir,
        source_subset=subset_name,
        repo_root=repo_root,
        hash_json=not args.no_json_hash,
        hash_video=args.hash_video,
        fast=args.fast,
        progress_every=args.progress_every,
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
    dataset_manifest["scan_mode"] = (
        args.layout if args.layout != "auto" else "recursive" if args.recursive else "flat"
    )
    dataset_manifest["fast_manifest_mode"] = bool(args.fast)
    dataset_manifest["data_quality"] = build_data_quality_summary(clip_records)
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
    print(
        "zero_byte_video_clips="
        f"{dataset_manifest['data_quality']['zero_byte_video_clip_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
