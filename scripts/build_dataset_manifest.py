#!/usr/bin/env python3
"""Build internal PriVTE dataset manifests from flat clip directories.

Expected input layout:

    data_root/
      210040_VID_2025_09_12_10_45_29_000/
        clip_000.mp4
        questionnaire.json
        questionnaire_list.png
        usage.json
        heartRate.json
        heartRate.png

The script writes:

    dataset_manifest.<subset>.<version>.json
    internal_person_manifest.<subset>.<version>.jsonl
    internal_clip_manifest.<subset>.<version>.jsonl

These manifests are internal indexes. They keep raw paths, raw subject ids, and
exact collection timestamps, so they are not suitable for public release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SEGMENT_DIR_RE = re.compile(
    r"^(?P<person_raw_id>[^_]+)_VID_"
    r"(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})_"
    r"(?P<hour>\d{2})_(?P<minute>\d{2})_(?P<second>\d{2})_"
    r"(?P<clip_index>\d+)$"
)

SENSITIVE_QUESTIONNAIRE_KEYWORDS = (
    "学校",
    "班级",
    "姓名",
    "编号",
    "民族",
    "ip",
    "提交",
    "来源",
    "来源详情",
    "地址",
    "电话",
)


def rel_path(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return slug or "dataset"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_segment_dir_name(name: str) -> dict[str, Any] | None:
    match = SEGMENT_DIR_RE.match(name)
    if not match:
        return None
    groups = match.groupdict()
    session_raw_key = "_".join(name.split("_")[:-1])
    captured_at_raw = (
        f"{groups['year']}-{groups['month']}-{groups['day']} "
        f"{groups['hour']}:{groups['minute']}:{groups['second']}"
    )
    return {
        "person_raw_id": groups["person_raw_id"],
        "session_raw_key": session_raw_key,
        "capture_date_raw": f"{groups['year']}-{groups['month']}-{groups['day']}",
        "capture_time_raw": f"{groups['hour']}:{groups['minute']}:{groups['second']}",
        "captured_at_raw": captured_at_raw,
        "clip_index": int(groups["clip_index"]),
        "clip_index_raw": groups["clip_index"],
    }


def json_structure_info(
    path: Path,
    repo_root: Path,
    *,
    hash_json: bool,
    public_policy: str,
) -> dict[str, Any]:
    info: dict[str, Any] = {
        "available": path.exists(),
        "path": rel_path(path, repo_root) if path.exists() else None,
        "public_policy": public_policy,
    }
    if not path.exists():
        return info

    info["size_bytes"] = path.stat().st_size
    if hash_json:
        info["sha256"] = sha256_file(path)

    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except Exception as exc:
        info.update(
            {
                "json_status": "invalid",
                "json_error": f"{type(exc).__name__}: {exc}",
            }
        )
        return info

    info["json_status"] = "ok"
    info["json_type"] = type(payload).__name__
    if isinstance(payload, list):
        info["record_count"] = len(payload)
        info["is_empty"] = len(payload) == 0
        first_object = next((item for item in payload if isinstance(item, dict)), None)
        info["item_keys"] = sorted(first_object.keys()) if first_object else []
    elif isinstance(payload, dict):
        keys = sorted(payload.keys())
        info["field_count"] = len(keys)
        info["is_empty"] = len(keys) == 0
        info["keys"] = keys
        lower_keys = [key.lower() for key in keys]
        info["sensitive_key_hits"] = sorted(
            {
                keyword
                for keyword in SENSITIVE_QUESTIONNAIRE_KEYWORDS
                if any(keyword in key for key in lower_keys)
            }
        )
        info["contains_sensitive_keys"] = bool(info["sensitive_key_hits"])
    else:
        info["is_empty"] = payload in ("", None)

    return info


def image_info(path: Path, repo_root: Path, public_policy: str) -> dict[str, Any]:
    return {
        "available": path.exists(),
        "path": rel_path(path, repo_root) if path.exists() else None,
        "size_bytes": path.stat().st_size if path.exists() else None,
        "public_policy": public_policy,
    }


def video_info(
    segment_dir: Path,
    repo_root: Path,
    *,
    hash_video: bool,
) -> dict[str, Any]:
    files = sorted(segment_dir.glob("*.mp4"))
    return {
        "available": bool(files),
        "count": len(files),
        "files": [
            {
                "path": rel_path(path, repo_root),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path) if hash_video else None,
                "public_policy": "exclude",
            }
            for path in files
        ],
        "public_policy": "exclude_raw_video",
    }


def assign_uids(person_raw_ids: list[str]) -> dict[str, str]:
    return {
        raw_id: f"internal_p{index:06d}"
        for index, raw_id in enumerate(sorted(set(person_raw_ids)), start=1)
    }


def build_clip_record(
    *,
    segment_dir: Path,
    parsed: dict[str, Any],
    source_subset: str,
    person_uid: str,
    session_uid: str,
    clip_uid: str,
    repo_root: Path,
    hash_json: bool,
    hash_video: bool,
) -> dict[str, Any]:
    questionnaire_path = segment_dir / "questionnaire.json"
    usage_path = segment_dir / "usage.json"
    heart_rate_path = segment_dir / "heartRate.json"

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
        "video": video_info(segment_dir, repo_root, hash_video=hash_video),
        "usage": json_structure_info(
            usage_path,
            repo_root,
            hash_json=hash_json,
            public_policy="coarse_category_and_duration_bin_only",
        ),
        "heart_rate": json_structure_info(
            heart_rate_path,
            repo_root,
            hash_json=hash_json,
            public_policy="exclude_exact_values_use_quality_or_trend_only",
        ),
        "questionnaire_ref": json_structure_info(
            questionnaire_path,
            repo_root,
            hash_json=hash_json,
            public_policy="exclude_raw_values_person_level_only",
        ),
        "images": {
            "questionnaire_list": image_info(
                segment_dir / "questionnaire_list.png",
                repo_root,
                "exclude",
            ),
            "heart_rate_plot": image_info(segment_dir / "heartRate.png", repo_root, "exclude"),
        },
        "privacy_flags": {
            "contains_raw_paths": True,
            "contains_raw_person_id": True,
            "contains_exact_timestamp": True,
            "public_release_allowed": False,
        },
    }


def summarize_person_questionnaire(clips: list[dict[str, Any]]) -> dict[str, Any]:
    questionnaire_infos = [clip["questionnaire_ref"] for clip in clips]
    available = [info for info in questionnaire_infos if info.get("available")]
    hashes = [info.get("sha256") for info in available if info.get("sha256")]
    nonempty = [info for info in available if not info.get("is_empty", True)]
    representative = nonempty[0] if nonempty else available[0] if available else None

    return {
        "available": bool(available),
        "level": "person",
        "scope": "person_level_repeated_in_clip_dirs" if available else "missing",
        "same_within_person": len(set(hashes)) <= 1 if hashes else None,
        "unique_payload_count": len(set(hashes)) if hashes else None,
        "nonempty_clip_count": len(nonempty),
        "empty_clip_count": len(available) - len(nonempty),
        "representative_path": representative.get("path") if representative else None,
        "representative_sha256": representative.get("sha256") if representative else None,
        "field_count": representative.get("field_count") if representative else None,
        "keys": representative.get("keys") if representative else [],
        "contains_sensitive_keys": representative.get("contains_sensitive_keys")
        if representative
        else None,
        "sensitive_key_hits": representative.get("sensitive_key_hits") if representative else [],
        "public_policy": "exclude_raw_values",
    }


def modality_completeness(clips: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_clips": len(clips),
        "num_video_clips": sum(1 for clip in clips if clip["video"]["available"]),
        "num_usage_records": sum(
            clip["usage"].get("record_count") or 0
            for clip in clips
            if clip["usage"].get("available")
        ),
        "num_nonempty_usage": sum(
            1
            for clip in clips
            if clip["usage"].get("available") and not clip["usage"].get("is_empty", True)
        ),
        "num_nonempty_heart_rate": sum(
            1
            for clip in clips
            if clip["heart_rate"].get("available") and not clip["heart_rate"].get("is_empty", True)
        ),
        "num_heart_rate_plots": sum(
            1 for clip in clips if clip["images"]["heart_rate_plot"]["available"]
        ),
        "has_questionnaire": any(
            clip["questionnaire_ref"].get("available")
            and not clip["questionnaire_ref"].get("is_empty", True)
            for clip in clips
        ),
    }


def build_person_records(
    clip_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_person: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for clip in clip_records:
        by_person[clip["person_uid"]].append(clip)

    records: list[dict[str, Any]] = []
    for person_uid, clips in sorted(by_person.items()):
        clips = sorted(clips, key=lambda item: (item["session_raw_key"], item["clip_index"]))
        person_raw_id = clips[0]["person_raw_id"]

        sessions: list[dict[str, Any]] = []
        for session_raw_key, session_clips_iter in group_by_session(clips):
            session_clips = list(session_clips_iter)
            sessions.append(
                {
                    "session_uid": session_clips[0]["session_uid"],
                    "session_raw_key": session_raw_key,
                    "capture_date_raw": session_clips[0]["capture_date_raw"],
                    "capture_time_raw": session_clips[0]["capture_time_raw"],
                    "captured_at_raw": session_clips[0]["captured_at_raw"],
                    "clip_count": len(session_clips),
                    "clips": [
                        {
                            "clip_uid": clip["clip_uid"],
                            "clip_index": clip["clip_index"],
                            "segment_dir_name": clip["segment_dir_name"],
                            "segment_dir_path": clip["segment_dir_path"],
                            "video": clip["video"],
                            "usage": clip["usage"],
                            "heart_rate": clip["heart_rate"],
                            "questionnaire_ref": {
                                "path": clip["questionnaire_ref"]["path"],
                                "sha256": clip["questionnaire_ref"].get("sha256"),
                                "available": clip["questionnaire_ref"]["available"],
                                "is_empty": clip["questionnaire_ref"].get("is_empty"),
                                "public_policy": clip["questionnaire_ref"]["public_policy"],
                            },
                            "images": clip["images"],
                        }
                        for clip in session_clips
                    ],
                }
            )

        records.append(
            {
                "schema_version": "internal_person_manifest.v0",
                "source_subset": clips[0]["source_subset"],
                "person_uid": person_uid,
                "person_raw_id": person_raw_id,
                "source_type": "raw_internal",
                "release_tier": "internal_raw",
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
                "questionnaire": summarize_person_questionnaire(clips),
                "sessions": sessions,
                "modality_completeness": modality_completeness(clips),
                "derived_outputs": {
                    "privte_evidence_json": None,
                    "privte_evidence_text": None,
                    "quality_summary": None,
                },
                "privacy_flags": {
                    "contains_raw_paths": True,
                    "contains_raw_person_id": True,
                    "contains_exact_timestamps": True,
                    "public_release_allowed": False,
                },
                "notes": [],
            }
        )

    return records


def group_by_session(clips: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for clip in clips:
        grouped[clip["session_raw_key"]].append(clip)
    return [
        (session_key, sorted(items, key=lambda item: item["clip_index"]))
        for session_key, items in sorted(grouped.items())
    ]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            file.write("\n")


def build_dataset_manifest(
    *,
    data_root: Path,
    repo_root: Path,
    subset_name: str,
    manifest_version: str,
    person_records: list[dict[str, Any]],
    clip_records: list[dict[str, Any]],
    invalid_dirs: list[str],
    output_paths: dict[str, Path],
) -> dict[str, Any]:
    file_counts = Counter()
    total_video_size = 0
    for clip in clip_records:
        for file_info in clip["video"]["files"]:
            file_counts["mp4"] += 1
            total_video_size += file_info["size_bytes"]
        for key in ("usage", "heart_rate", "questionnaire_ref"):
            if clip[key]["available"]:
                file_counts["json"] += 1
        for image in clip["images"].values():
            if image["available"]:
                file_counts["png"] += 1

    return {
        "schema_version": "dataset_manifest.v0",
        "manifest_version": manifest_version,
        "dataset_name": "MDU-RiskText",
        "source_dataset": "internal_raw_mdu",
        "source_subset": subset_name,
        "data_root": rel_path(data_root, repo_root),
        "release_tier": "internal_raw",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "text-only minor digital use risk screening",
        "unit_of_analysis": "person",
        "split_policy": "subject_wise",
        "directory_pattern": "{person_raw_id}_VID_{yyyy}_{mm}_{dd}_{hh}_{mm}_{ss}_{clip_index}",
        "counts": {
            "persons": len(person_records),
            "sessions": len({clip["session_raw_key"] for clip in clip_records}),
            "clips": len(clip_records),
            "invalid_directories": len(invalid_dirs),
            "files": dict(file_counts),
            "total_video_size_bytes": total_video_size,
        },
        "output_files": {
            key: rel_path(path, repo_root) for key, path in output_paths.items()
        },
        "modalities": {
            "video": "clip_level",
            "heart_rate": "clip_level_or_missing",
            "questionnaire": "person_level_repeated_in_clip_dirs",
            "app_usage": "clip_or_session_level",
        },
        "label_schema": {
            "primary_label": [
                "no_observed_risk",
                "mild_risk",
                "moderate_risk",
                "high_risk",
                "insufficient_evidence",
            ],
            "label_basis": ["video", "heart_rate", "questionnaire", "app_usage"],
        },
        "privacy_policy": {
            "raw_video_public": False,
            "raw_image_public": False,
            "raw_audio_public": False,
            "raw_questionnaire_public": False,
            "exact_heart_rate_public": False,
            "exact_timestamp_public": False,
            "raw_person_id_public": False,
        },
        "invalid_directory_names": invalid_dirs,
    }


def scan_data_root(
    *,
    data_root: Path,
    source_subset: str,
    repo_root: Path,
    hash_json: bool,
    hash_video: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    segment_dirs = sorted([path for path in data_root.iterdir() if path.is_dir()], key=lambda p: p.name)
    parsed_by_dir: list[tuple[Path, dict[str, Any]]] = []
    invalid_dirs: list[str] = []
    for segment_dir in segment_dirs:
        parsed = parse_segment_dir_name(segment_dir.name)
        if parsed is None:
            invalid_dirs.append(segment_dir.name)
            continue
        parsed_by_dir.append((segment_dir, parsed))

    person_uids = assign_uids([parsed["person_raw_id"] for _, parsed in parsed_by_dir])
    session_uid_by_key: dict[str, str] = {}
    session_counts_by_person: dict[str, int] = defaultdict(int)
    clip_counts_by_person: dict[str, int] = defaultdict(int)
    clip_records: list[dict[str, Any]] = []

    for segment_dir, parsed in parsed_by_dir:
        person_uid = person_uids[parsed["person_raw_id"]]
        if parsed["session_raw_key"] not in session_uid_by_key:
            session_counts_by_person[person_uid] += 1
            session_uid_by_key[parsed["session_raw_key"]] = (
                f"{person_uid}_s{session_counts_by_person[person_uid]:04d}"
            )
        clip_counts_by_person[person_uid] += 1
        clip_uid = f"{person_uid}_c{clip_counts_by_person[person_uid]:06d}"

        clip_records.append(
            build_clip_record(
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
        )

    return clip_records, invalid_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build internal person-level and clip-level manifests from flat PriVTE data."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/6.1data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--subset-name", default=None)
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
    data_root = args.data_root
    if not data_root.exists():
        print(f"Data root not found: {data_root}", file=sys.stderr)
        return 1
    if not data_root.is_dir():
        print(f"Data root is not a directory: {data_root}", file=sys.stderr)
        return 1

    subset_name = args.subset_name or data_root.name
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

    clip_records, invalid_dirs = scan_data_root(
        data_root=data_root,
        source_subset=subset_name,
        repo_root=repo_root,
        hash_json=not args.no_json_hash,
        hash_video=args.hash_video,
    )
    person_records = build_person_records(clip_records)
    dataset_manifest = build_dataset_manifest(
        data_root=data_root,
        repo_root=repo_root,
        subset_name=subset_name,
        manifest_version=version,
        person_records=person_records,
        clip_records=clip_records,
        invalid_dirs=invalid_dirs,
        output_paths=output_paths,
    )

    write_json(dataset_path, dataset_manifest)
    write_jsonl(person_path, person_records)
    write_jsonl(clip_path, clip_records)

    print(f"dataset_manifest={dataset_path}")
    print(f"person_manifest={person_path}")
    print(f"clip_manifest={clip_path}")
    print(f"persons={len(person_records)}")
    print(f"sessions={dataset_manifest['counts']['sessions']}")
    print(f"clips={len(clip_records)}")
    print(f"invalid_directories={len(invalid_dirs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
