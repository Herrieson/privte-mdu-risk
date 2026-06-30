#!/usr/bin/env python3
"""Audit questionnaire, heart-rate, usage, and plot coverage from an internal manifest.

The report is intended for internal data planning. It deliberately avoids
emitting questionnaire answers, app names, exact heart-rate values, exact
timestamps, raw person IDs, or raw file paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


JSON_NAME_BY_MODALITY = {
    "questionnaire": "questionnaire_ref",
    "heart_rate": "heart_rate",
    "usage": "usage",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return records


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def safe_digest(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def parse_json_file(path: Path) -> tuple[Any | None, str | None]:
    try:
        with path.open(encoding="utf-8") as file:
            return json.load(file), None
    except FileNotFoundError:
        return None, "missing_on_disk"
    except json.JSONDecodeError:
        return None, "invalid_json"
    except UnicodeDecodeError:
        return None, "invalid_encoding"
    except OSError:
        return None, "read_error"


def stats(values: list[int | float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    sorted_values = sorted(values)

    def percentile(percent: float) -> int | float:
        if len(sorted_values) == 1:
            return sorted_values[0]
        index = round((len(sorted_values) - 1) * percent)
        return sorted_values[index]

    return {
        "count": len(values),
        "min": sorted_values[0],
        "p25": percentile(0.25),
        "median": statistics.median(sorted_values),
        "p75": percentile(0.75),
        "p90": percentile(0.90),
        "max": sorted_values[-1],
        "mean": round(float(statistics.fmean(sorted_values)), 3),
    }


def bucket_count(value: int) -> str:
    if value == 0:
        return "0"
    if value == 1:
        return "1"
    if value <= 3:
        return "2-3"
    if value <= 5:
        return "4-5"
    if value <= 10:
        return "6-10"
    if value <= 30:
        return "11-30"
    if value <= 60:
        return "31-60"
    return "61+"


def bucket_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    if seconds <= 0:
        return "0"
    if seconds < 30:
        return "<30s"
    if seconds < 60:
        return "30-60s"
    if seconds < 180:
        return "1-3min"
    if seconds < 300:
        return "3-5min"
    if seconds < 600:
        return "5-10min"
    if seconds < 1800:
        return "10-30min"
    return "30min+"


def bucket_ratio(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 0.25:
        return "<25%"
    if value < 0.50:
        return "25-50%"
    if value < 0.75:
        return "50-75%"
    if value < 0.90:
        return "75-90%"
    return "90-100%"


def bucket_bpm(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 60:
        return "<60"
    if value < 80:
        return "60-79"
    if value < 100:
        return "80-99"
    if value < 120:
        return "100-119"
    return "120+"


def infer_seconds_span(timestamps: list[int | float]) -> float | None:
    if len(timestamps) < 2:
        return None
    span = max(timestamps) - min(timestamps)
    if span <= 0:
        return 0.0
    ordered = sorted(timestamps)
    deltas = [later - earlier for earlier, later in zip(ordered, ordered[1:]) if later > earlier]
    median_delta = statistics.median(deltas) if deltas else span
    if median_delta > 100:
        return float(span) / 1000.0
    return float(span)


def key_categories(keys: list[str]) -> list[str]:
    categories: set[str] = set()
    for key in keys:
        lowered = key.lower()
        if any(token in lowered for token in ("ip", "来源", "提交答卷时间", "所用时间")):
            categories.add("collection_metadata")
        if any(token in lowered for token in ("编号", "序号")):
            categories.add("respondent_identifier")
        if any(token in lowered for token in ("学校", "校区", "班", "年级")):
            categories.add("school_or_class_context")
        if any(token in lowered for token in ("民族", "性别", "年龄")):
            categories.add("demographic_fields")
        if any(token in lowered for token in ("在哪里", "地址", "位置")):
            categories.add("location_or_use_context")
    return sorted(categories)


def modality_file_path(record: dict[str, Any], manifest_root: Path, modality_key: str) -> Path | None:
    info = record.get(modality_key, {})
    raw_path = info.get("path")
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return manifest_root / path


def init_modality_counter(total_clips: int) -> dict[str, Any]:
    return {
        "clip_files_available": 0,
        "clip_files_missing_in_manifest": 0,
        "clip_files_missing_on_disk": 0,
        "clip_files_empty": 0,
        "clip_files_parse_success": 0,
        "clip_files_parse_failed": 0,
        "parse_errors": {},
        "size_bytes": {"count": 0},
        "coverage_ratio": 0.0,
        "total_clips": total_clips,
    }


def finalize_modality_counter(counter: dict[str, Any], sizes: list[int], total_clips: int) -> None:
    counter["coverage_ratio"] = round(counter["clip_files_available"] / total_clips, 4) if total_clips else 0
    counter["parse_success_ratio_among_available"] = (
        round(counter["clip_files_parse_success"] / counter["clip_files_available"], 4)
        if counter["clip_files_available"]
        else 0
    )
    counter["size_bytes"] = stats(sizes)
    counter["parse_errors"] = dict(sorted(counter["parse_errors"].items()))


def build_report(
    *,
    clip_records: list[dict[str, Any]],
    clip_manifest: Path,
    output_path: Path,
) -> dict[str, Any]:
    manifest_root = Path.cwd()
    total_clips = len(clip_records)
    person_uids = {str(record.get("person_uid")) for record in clip_records}
    session_uids = {str(record.get("session_uid")) for record in clip_records}
    source_roots = Counter(str(record.get("source_data_root", "<unknown>")) for record in clip_records)

    modality_counters = {
        name: init_modality_counter(total_clips) for name in JSON_NAME_BY_MODALITY
    }
    modality_sizes: dict[str, list[int]] = defaultdict(list)

    by_person: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "clips": 0,
            "questionnaire_payload_hashes": set(),
            "questionnaire_parse_success": 0,
            "heart_rate_valid_clips": 0,
            "heart_rate_usable_clips": 0,
            "usage_valid_clips": 0,
        }
    )

    questionnaire_field_counts: list[int] = []
    questionnaire_field_count_buckets: Counter[str] = Counter()
    questionnaire_key_signature_counts: Counter[str] = Counter()
    questionnaire_sensitive_categories: Counter[str] = Counter()
    questionnaire_payload_reuse_counts: Counter[str] = Counter()
    questionnaire_same_hash_by_clip: Counter[str] = Counter()

    heart_record_counts: list[int] = []
    heart_record_count_buckets: Counter[str] = Counter()
    heart_numeric_bpm_ratio_buckets: Counter[str] = Counter()
    heart_numeric_timestamp_ratio_buckets: Counter[str] = Counter()
    heart_span_buckets: Counter[str] = Counter()
    heart_mean_bpm_bins: Counter[str] = Counter()
    heart_usable_clips = 0
    heart_empty_sequences = 0
    heart_schema_signatures: Counter[str] = Counter()

    usage_record_counts: list[int] = []
    usage_record_count_buckets: Counter[str] = Counter()
    usage_total_time_buckets: Counter[str] = Counter()
    usage_top_item_share_buckets: Counter[str] = Counter()
    usage_schema_signatures: Counter[str] = Counter()
    usage_valid_total_time_clips = 0

    image_counts = {
        "questionnaire_list_png_available": 0,
        "questionnaire_list_png_missing": 0,
        "heart_rate_plot_png_available": 0,
        "heart_rate_plot_png_missing": 0,
    }

    clips_with_all_three_json = 0
    clips_with_all_three_json_parse_success = 0
    clips_with_video_nonzero_and_all_three_json = 0

    for index, record in enumerate(clip_records, start=1):
        person_uid = str(record.get("person_uid"))
        by_person[person_uid]["clips"] += 1

        video_files = record.get("video", {}).get("files", [])
        has_nonzero_video = any(int(file_info.get("size_bytes") or 0) > 0 for file_info in video_files)

        all_three_available = True
        all_three_parsed = True

        for public_name, manifest_key in JSON_NAME_BY_MODALITY.items():
            counter = modality_counters[public_name]
            info = record.get(manifest_key, {})
            available_in_manifest = bool(info.get("available"))
            if not available_in_manifest:
                counter["clip_files_missing_in_manifest"] += 1
                all_three_available = False
                all_three_parsed = False
                continue

            counter["clip_files_available"] += 1
            path = modality_file_path(record, manifest_root, manifest_key)
            if path is None or not path.exists():
                counter["clip_files_missing_on_disk"] += 1
                counter["parse_errors"]["missing_on_disk"] = counter["parse_errors"].get(
                    "missing_on_disk", 0
                ) + 1
                all_three_parsed = False
                continue

            size = path.stat().st_size
            modality_sizes[public_name].append(size)
            if size == 0:
                counter["clip_files_empty"] += 1

            payload, error = parse_json_file(path)
            if error:
                counter["clip_files_parse_failed"] += 1
                counter["parse_errors"][error] = counter["parse_errors"].get(error, 0) + 1
                all_three_parsed = False
                continue

            counter["clip_files_parse_success"] += 1

            if public_name == "questionnaire":
                process_questionnaire(
                    payload=payload,
                    person_bucket=by_person[person_uid],
                    field_counts=questionnaire_field_counts,
                    field_count_buckets=questionnaire_field_count_buckets,
                    key_signature_counts=questionnaire_key_signature_counts,
                    sensitive_categories=questionnaire_sensitive_categories,
                    payload_reuse_counts=questionnaire_payload_reuse_counts,
                    same_hash_by_clip=questionnaire_same_hash_by_clip,
                )
            elif public_name == "heart_rate":
                usable = process_heart_rate(
                    payload=payload,
                    record_counts=heart_record_counts,
                    record_count_buckets=heart_record_count_buckets,
                    numeric_bpm_ratio_buckets=heart_numeric_bpm_ratio_buckets,
                    numeric_timestamp_ratio_buckets=heart_numeric_timestamp_ratio_buckets,
                    span_buckets=heart_span_buckets,
                    mean_bpm_bins=heart_mean_bpm_bins,
                    schema_signatures=heart_schema_signatures,
                )
                if isinstance(payload, list) and len(payload) == 0:
                    heart_empty_sequences += 1
                if isinstance(payload, list) and len(payload) > 0:
                    by_person[person_uid]["heart_rate_valid_clips"] += 1
                if usable:
                    heart_usable_clips += 1
                    by_person[person_uid]["heart_rate_usable_clips"] += 1
            elif public_name == "usage":
                valid_total = process_usage(
                    payload=payload,
                    record_counts=usage_record_counts,
                    record_count_buckets=usage_record_count_buckets,
                    total_time_buckets=usage_total_time_buckets,
                    top_item_share_buckets=usage_top_item_share_buckets,
                    schema_signatures=usage_schema_signatures,
                )
                if isinstance(payload, list) and len(payload) > 0:
                    by_person[person_uid]["usage_valid_clips"] += 1
                if valid_total:
                    usage_valid_total_time_clips += 1

        if all_three_available:
            clips_with_all_three_json += 1
        if all_three_parsed:
            clips_with_all_three_json_parse_success += 1
        if has_nonzero_video and all_three_available:
            clips_with_video_nonzero_and_all_three_json += 1

        images = record.get("images", {})
        questionnaire_png = images.get("questionnaire_list", {})
        heart_png = images.get("heart_rate_plot", {})
        if questionnaire_png.get("available"):
            image_counts["questionnaire_list_png_available"] += 1
        else:
            image_counts["questionnaire_list_png_missing"] += 1
        if heart_png.get("available"):
            image_counts["heart_rate_plot_png_available"] += 1
        else:
            image_counts["heart_rate_plot_png_missing"] += 1

        if index % 5000 == 0:
            print(f"audited_clip_records={index}/{total_clips}", file=sys.stderr, flush=True)

    for name, counter in modality_counters.items():
        finalize_modality_counter(counter, modality_sizes[name], total_clips)

    questionnaire_person_summary = summarize_questionnaire_by_person(by_person)
    person_modality_summary = summarize_person_modalities(by_person)

    return {
        "schema_version": "privte_auxiliary_modality_audit.v0",
        "clip_manifest": str(clip_manifest),
        "output_path": str(output_path),
        "privacy_policy": {
            "raw_questionnaire_answers_emitted": False,
            "raw_app_names_emitted": False,
            "exact_heart_rate_values_emitted": False,
            "exact_heart_rate_timestamps_emitted": False,
            "raw_person_ids_emitted": False,
            "raw_paths_emitted": False,
            "note": "This report keeps only aggregate counts, bins, and structural signatures.",
        },
        "dataset_scope": {
            "persons": len(person_uids),
            "sessions": len(session_uids),
            "clips": total_clips,
            "source_root_clip_counts": dict(sorted(source_roots.items())),
        },
        "clip_level_coverage": {
            "json_modalities": modality_counters,
            "images": image_counts,
            "co_presence": {
                "clips_with_questionnaire_usage_heart_rate_json": clips_with_all_three_json,
                "clips_with_questionnaire_usage_heart_rate_json_parse_success": (
                    clips_with_all_three_json_parse_success
                ),
                "clips_with_nonzero_video_and_all_three_json": (
                    clips_with_video_nonzero_and_all_three_json
                ),
            },
        },
        "person_level_coverage": person_modality_summary,
        "questionnaire": {
            "interpretation": (
                "Questionnaire files are repeated in clip directories but should be treated "
                "as person-level/context-or-label data, not public text-only evidence input."
            ),
            "field_count_stats": stats(questionnaire_field_counts),
            "field_count_buckets": dict(sorted(questionnaire_field_count_buckets.items())),
            "key_signature_count": len(questionnaire_key_signature_counts),
            "top_key_signature_clip_counts": questionnaire_key_signature_counts.most_common(10),
            "sensitive_key_category_hits": dict(sorted(questionnaire_sensitive_categories.items())),
            "payload_hash_reuse_clip_count_buckets": dict(
                sorted(
                    Counter(bucket_count(count) for count in questionnaire_payload_reuse_counts.values()).items()
                )
            ),
            "person_payload_repetition": questionnaire_person_summary,
        },
        "heart_rate": {
            "interpretation": (
                "Heart-rate values are available internally as sequences. Public evidence should "
                "use only quality/trend/binned summaries, never exact bpm or exact timestamps."
            ),
            "record_count_stats": stats(heart_record_counts),
            "record_count_buckets": dict(sorted(heart_record_count_buckets.items())),
            "empty_sequence_clips": heart_empty_sequences,
            "usable_sequence_clips": heart_usable_clips,
            "usable_sequence_definition": "list payload with >=10 records and >=80% numeric bpm values",
            "numeric_bpm_ratio_buckets": dict(sorted(heart_numeric_bpm_ratio_buckets.items())),
            "numeric_timestamp_ratio_buckets": dict(sorted(heart_numeric_timestamp_ratio_buckets.items())),
            "duration_span_buckets": dict(sorted(heart_span_buckets.items())),
            "mean_bpm_bins": dict(sorted(heart_mean_bpm_bins.items())),
            "schema_signatures": heart_schema_signatures.most_common(10),
        },
        "usage": {
            "interpretation": (
                "Usage records contain app names and durations. App names should be mapped to "
                "coarse categories or excluded before any public/control-side evidence."
            ),
            "record_count_stats": stats(usage_record_counts),
            "record_count_buckets": dict(sorted(usage_record_count_buckets.items())),
            "clips_with_numeric_total_time": usage_valid_total_time_clips,
            "total_time_buckets": dict(sorted(usage_total_time_buckets.items())),
            "top_item_share_buckets": dict(sorted(usage_top_item_share_buckets.items())),
            "schema_signatures": usage_schema_signatures.most_common(10),
        },
        "recommended_next_steps": [
            "Keep questionnaire and exact heart-rate as internal label/context sources, not LLM input.",
            "Derive privacy-safe heart-rate trend/quality bins only after deciding whether physiological context is part of the controlled tier.",
            "Create an app-name-to-coarse-category map before using usage.json; do not expose raw app names.",
            "Use nonzero-video plus parsed auxiliary-modality coverage when selecting the first V2 tuning subset.",
        ],
    }


def process_questionnaire(
    *,
    payload: Any,
    person_bucket: dict[str, Any],
    field_counts: list[int],
    field_count_buckets: Counter[str],
    key_signature_counts: Counter[str],
    sensitive_categories: Counter[str],
    payload_reuse_counts: Counter[str],
    same_hash_by_clip: Counter[str],
) -> None:
    if not isinstance(payload, dict):
        return
    person_bucket["questionnaire_parse_success"] += 1
    field_count = len(payload)
    field_counts.append(field_count)
    field_count_buckets[bucket_count(field_count)] += 1
    keys = [str(key) for key in payload.keys()]
    key_signature = safe_digest(sorted(keys))[:16]
    key_signature_counts[key_signature] += 1
    payload_hash = safe_digest(payload)[:16]
    payload_reuse_counts[payload_hash] += 1
    same_hash_by_clip[payload_hash] += 1
    person_bucket["questionnaire_payload_hashes"].add(payload_hash)
    for category in key_categories(keys):
        sensitive_categories[category] += 1


def process_heart_rate(
    *,
    payload: Any,
    record_counts: list[int],
    record_count_buckets: Counter[str],
    numeric_bpm_ratio_buckets: Counter[str],
    numeric_timestamp_ratio_buckets: Counter[str],
    span_buckets: Counter[str],
    mean_bpm_bins: Counter[str],
    schema_signatures: Counter[str],
) -> bool:
    if not isinstance(payload, list):
        return False
    record_count = len(payload)
    record_counts.append(record_count)
    record_count_buckets[bucket_count(record_count)] += 1

    bpm_values: list[float] = []
    timestamps: list[float] = []
    key_sets: Counter[str] = Counter()
    for item in payload:
        if not isinstance(item, dict):
            continue
        key_sets[safe_digest(sorted(str(key) for key in item.keys()))[:16]] += 1
        bpm = item.get("bpm")
        timestamp = item.get("timestamp")
        if isinstance(bpm, int | float):
            bpm_values.append(float(bpm))
        if isinstance(timestamp, int | float):
            timestamps.append(float(timestamp))

    if key_sets:
        schema_signatures[key_sets.most_common(1)[0][0]] += 1
    numeric_bpm_ratio = len(bpm_values) / record_count if record_count else None
    numeric_timestamp_ratio = len(timestamps) / record_count if record_count else None
    numeric_bpm_ratio_buckets[bucket_ratio(numeric_bpm_ratio)] += 1
    numeric_timestamp_ratio_buckets[bucket_ratio(numeric_timestamp_ratio)] += 1

    span_buckets[bucket_seconds(infer_seconds_span(timestamps))] += 1
    mean_bpm_bins[bucket_bpm(statistics.fmean(bpm_values) if bpm_values else None)] += 1
    return record_count >= 10 and (numeric_bpm_ratio or 0) >= 0.8


def process_usage(
    *,
    payload: Any,
    record_counts: list[int],
    record_count_buckets: Counter[str],
    total_time_buckets: Counter[str],
    top_item_share_buckets: Counter[str],
    schema_signatures: Counter[str],
) -> bool:
    if not isinstance(payload, list):
        return False
    record_count = len(payload)
    record_counts.append(record_count)
    record_count_buckets[bucket_count(record_count)] += 1

    total_times: list[float] = []
    key_sets: Counter[str] = Counter()
    for item in payload:
        if not isinstance(item, dict):
            continue
        key_sets[safe_digest(sorted(str(key) for key in item.keys()))[:16]] += 1
        total_time = item.get("totalTime")
        if isinstance(total_time, int | float):
            total_times.append(float(total_time))

    if key_sets:
        schema_signatures[key_sets.most_common(1)[0][0]] += 1
    if not total_times:
        total_time_buckets["unknown"] += 1
        top_item_share_buckets["unknown"] += 1
        return False

    total = sum(total_times)
    # Most exports use milliseconds; the fallback keeps small second-scale values usable.
    seconds = total / 1000.0 if total > 24 * 60 * 60 else total
    total_time_buckets[bucket_seconds(seconds)] += 1
    top_item_share_buckets[bucket_ratio(max(total_times) / total if total > 0 else None)] += 1
    return True


def summarize_questionnaire_by_person(by_person: dict[str, dict[str, Any]]) -> dict[str, Any]:
    unique_payload_counts: list[int] = []
    same_payload_persons = 0
    persons_with_questionnaire = 0
    persons_with_repeated_payloads = 0
    parsed_clip_counts: list[int] = []

    for bucket in by_person.values():
        hashes = bucket["questionnaire_payload_hashes"]
        parsed_count = int(bucket["questionnaire_parse_success"])
        if parsed_count:
            persons_with_questionnaire += 1
            parsed_clip_counts.append(parsed_count)
        unique_count = len(hashes)
        unique_payload_counts.append(unique_count)
        if unique_count == 1:
            same_payload_persons += 1
        if parsed_count > 1:
            persons_with_repeated_payloads += 1

    return {
        "persons_with_parse_success": persons_with_questionnaire,
        "persons_without_parse_success": len(by_person) - persons_with_questionnaire,
        "persons_with_single_unique_payload": same_payload_persons,
        "persons_with_multiple_questionnaire_files": persons_with_repeated_payloads,
        "unique_payload_count_per_person_stats": stats(unique_payload_counts),
        "parsed_questionnaire_file_count_per_person_stats": stats(parsed_clip_counts),
    }


def summarize_person_modalities(by_person: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total_persons = len(by_person)
    persons_with_questionnaire = 0
    persons_with_heart_rate = 0
    persons_with_usable_heart_rate = 0
    persons_with_usage = 0
    persons_with_all_aux = 0
    persons_with_all_aux_and_usable_heart = 0

    for bucket in by_person.values():
        has_questionnaire = int(bucket["questionnaire_parse_success"]) > 0
        has_heart_rate = int(bucket["heart_rate_valid_clips"]) > 0
        has_usable_heart_rate = int(bucket["heart_rate_usable_clips"]) > 0
        has_usage = int(bucket["usage_valid_clips"]) > 0
        persons_with_questionnaire += int(has_questionnaire)
        persons_with_heart_rate += int(has_heart_rate)
        persons_with_usable_heart_rate += int(has_usable_heart_rate)
        persons_with_usage += int(has_usage)
        persons_with_all_aux += int(has_questionnaire and has_heart_rate and has_usage)
        persons_with_all_aux_and_usable_heart += int(
            has_questionnaire and has_usable_heart_rate and has_usage
        )

    return {
        "persons": total_persons,
        "persons_with_parsed_questionnaire": persons_with_questionnaire,
        "persons_with_nonempty_heart_rate_sequence": persons_with_heart_rate,
        "persons_with_usable_heart_rate_sequence": persons_with_usable_heart_rate,
        "persons_with_nonempty_usage": persons_with_usage,
        "persons_with_questionnaire_usage_heart_rate": persons_with_all_aux,
        "persons_with_questionnaire_usage_usable_heart_rate": persons_with_all_aux_and_usable_heart,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit internal auxiliary PriVTE modalities without exposing raw values."
    )
    parser.add_argument(
        "--clip-manifest",
        type=Path,
        required=True,
        help="Internal clip manifest JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/data_audit/auxiliary_modality_audit_report.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    clip_records = load_jsonl(args.clip_manifest)
    report = build_report(
        clip_records=clip_records,
        clip_manifest=args.clip_manifest,
        output_path=args.output,
    )
    write_json(args.output, report)
    print(f"auxiliary_modality_audit={args.output}")
    print(f"persons={report['dataset_scope']['persons']}")
    print(f"clips={report['dataset_scope']['clips']}")
    print(
        "questionnaire_parse_success="
        f"{report['clip_level_coverage']['json_modalities']['questionnaire']['clip_files_parse_success']}"
    )
    print(
        "heart_rate_usable_clips="
        f"{report['heart_rate']['usable_sequence_clips']}"
    )
    print(f"usage_parse_success={report['clip_level_coverage']['json_modalities']['usage']['clip_files_parse_success']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
