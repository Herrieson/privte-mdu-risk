#!/usr/bin/env python3
"""Audit alignment between PriVTE evidence features and target labels."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


LABEL_ORDER = {
    "no_observed_risk": 0,
    "mild_risk": 1,
    "moderate_risk": 2,
    "high_risk": 3,
    "insufficient_evidence": 4,
}


NUMERIC_FEATURES = [
    ("sampled_frame_count", ("sampled_frame_count",)),
    ("device_visible_ratio", ("behavior_v1_features", "device_visible_ratio")),
    ("hand_visible_ratio", ("behavior_v1_features", "hand_visible_ratio")),
    (
        "hand_device_proximity_ratio",
        ("behavior_v1_features", "hand_device_proximity_ratio"),
    ),
    (
        "stable_screen_engagement_ratio",
        ("behavior_v1_features", "stable_screen_engagement_proxy_ratio"),
    ),
    (
        "active_hand_device_interaction_ratio",
        ("behavior_v1_features", "active_hand_device_interaction_proxy_ratio"),
    ),
    (
        "device_region_activity_ratio",
        ("behavior_v1_features", "device_region_activity_proxy_ratio"),
    ),
    (
        "repetitive_operation_count",
        ("behavior_v1_features", "repetitive_operation_proxy_count"),
    ),
    (
        "temporal_engagement_episode_count",
        ("temporal_sequence_summary", "engagement_episode_count"),
    ),
    (
        "temporal_active_operation_point_count",
        ("temporal_sequence_summary", "active_operation_point_count"),
    ),
    (
        "temporal_direct_operation_point_count",
        ("temporal_sequence_summary", "direct_operation_point_count"),
    ),
    (
        "temporal_repetitive_operation_point_count",
        ("temporal_sequence_summary", "repetitive_operation_point_count"),
    ),
    (
        "temporal_stable_screen_point_count",
        ("temporal_sequence_summary", "stable_screen_point_count"),
    ),
    (
        "temporal_visible_without_engagement_point_count",
        ("temporal_sequence_summary", "visible_without_engagement_point_count"),
    ),
    (
        "temporal_confounded_activity_point_count",
        ("temporal_sequence_summary", "confounded_activity_point_count"),
    ),
    ("trace_risk_pattern_score", ("trace_risk_summary", "risk_pattern_score")),
    (
        "trace_stable_engagement_share",
        ("trace_risk_summary", "coarse_rates", "stable_engagement_share"),
    ),
    (
        "trace_active_operation_share",
        ("trace_risk_summary", "coarse_rates", "active_operation_share"),
    ),
    (
        "trace_direct_operation_share",
        ("trace_risk_summary", "coarse_rates", "direct_operation_share"),
    ),
    (
        "trace_repetitive_operation_share",
        ("trace_risk_summary", "coarse_rates", "repetitive_operation_share"),
    ),
    (
        "trace_visible_without_engagement_share",
        ("trace_risk_summary", "coarse_rates", "visible_without_engagement_share"),
    ),
    (
        "trace_confounded_activity_share",
        ("trace_risk_summary", "coarse_rates", "confounded_activity_share"),
    ),
]


CATEGORICAL_FEATURES = [
    ("extractor", ("extractor", "name")),
    (
        "temporal_main_observation_pattern",
        ("temporal_sequence_summary", "main_observation_pattern"),
    ),
    (
        "temporal_signal_strength",
        ("temporal_sequence_summary", "temporal_video_proxy_signal_strength"),
    ),
    ("trace_risk_pattern_strength", ("trace_risk_summary", "risk_pattern_strength")),
    ("trace_ordinary_use_pattern", ("trace_risk_summary", "ordinary_use_pattern")),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-jsonl", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis/evidence_label_audit"),
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Output filename prefix. Defaults to the evidence JSONL stem.",
    )
    parser.add_argument("--top-samples", type=int, default=30)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
    return records


def get_path(values: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = values
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def label_from_record(record: dict[str, Any]) -> str:
    label = record.get("target_label", {}).get("risk_level")
    return str(label or "<missing>")


def video_summary(record: dict[str, Any]) -> dict[str, Any]:
    return record.get("evidence", {}).get("feature_blocks", {}).get(
        "video_proxy_summary",
        {},
    )


def extract_row(record: dict[str, Any]) -> dict[str, Any]:
    video = video_summary(record)
    row: dict[str, Any] = {
        "sample_id": record.get("sample_id"),
        "target_label": label_from_record(record),
    }
    for name, path in NUMERIC_FEATURES:
        row[name] = to_float(get_path(video, path))
    for name, path in CATEGORICAL_FEATURES:
        row[name] = get_path(video, path)
    return row


def summarize_numeric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = sorted(
        {row["target_label"] for row in rows},
        key=lambda value: LABEL_ORDER.get(value, 99),
    )
    summaries: dict[str, Any] = {}
    for feature, _path in NUMERIC_FEATURES:
        by_label = {}
        means_for_direction = []
        for label in labels:
            values = [
                row[feature]
                for row in rows
                if row["target_label"] == label and row.get(feature) is not None
            ]
            if values:
                stats = {
                    "count": len(values),
                    "mean": round(statistics.mean(values), 4),
                    "min": round(min(values), 4),
                    "max": round(max(values), 4),
                }
                means_for_direction.append((label, stats["mean"]))
            else:
                stats = {"count": 0, "mean": None, "min": None, "max": None}
            by_label[label] = stats
        summaries[feature] = {
            "by_label": by_label,
            "direction": direction_from_means(means_for_direction),
        }
    return summaries


def direction_from_means(label_means: list[tuple[str, float]]) -> str:
    ordered = [
        mean
        for label, mean in label_means
        if label in {"no_observed_risk", "mild_risk", "moderate_risk", "high_risk"}
    ]
    if len(ordered) < 2:
        return "insufficient_labels"
    increasing = all(left <= right for left, right in zip(ordered, ordered[1:]))
    decreasing = all(left >= right for left, right in zip(ordered, ordered[1:]))
    if increasing and not decreasing:
        return "monotonic_increasing_with_label"
    if decreasing and not increasing:
        return "monotonic_decreasing_with_label"
    if increasing and decreasing:
        return "flat_or_tied"
    return "mixed_or_non_monotonic"


def summarize_categorical(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for feature, _path in CATEGORICAL_FEATURES:
        by_label: dict[str, dict[str, int]] = {}
        for label in sorted(
            {row["target_label"] for row in rows},
            key=lambda value: LABEL_ORDER.get(value, 99),
        ):
            counter = Counter(
                str(row.get(feature))
                for row in rows
                if row["target_label"] == label and row.get(feature) is not None
            )
            by_label[label] = dict(counter.most_common())
        summaries[feature] = {"by_label": by_label}
    return summaries


def discordance_flags(row: dict[str, Any]) -> list[str]:
    label = row["target_label"]
    flags: list[str] = []
    trace_score = row.get("trace_risk_pattern_score")
    active_points = row.get("temporal_active_operation_point_count") or 0
    direct_points = row.get("temporal_direct_operation_point_count") or 0
    repetitive_points = row.get("temporal_repetitive_operation_point_count") or 0
    stable_points = row.get("temporal_stable_screen_point_count") or 0
    if label == "no_observed_risk":
        if trace_score is not None and trace_score >= 4:
            flags.append("no_label_but_trace_score_moderate_or_high")
        if active_points >= 4 or direct_points >= 3 or repetitive_points >= 3:
            flags.append("no_label_but_active_or_repetitive_video_signal")
    if label in {"moderate_risk", "high_risk"}:
        if trace_score is not None and trace_score <= 1:
            flags.append("risk_label_but_trace_score_low")
        if active_points == 0 and repetitive_points == 0 and stable_points < 20:
            flags.append("risk_label_but_low_video_behavior_signal")
    return flags


def build_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [extract_row(record) for record in records]
    flagged = []
    for row in rows:
        flags = discordance_flags(row)
        if flags:
            flagged.append(
                {
                    "sample_id": row["sample_id"],
                    "target_label": row["target_label"],
                    "flags": flags,
                    "trace_risk_pattern_score": row.get("trace_risk_pattern_score"),
                    "temporal_active_operation_point_count": row.get(
                        "temporal_active_operation_point_count"
                    ),
                    "temporal_direct_operation_point_count": row.get(
                        "temporal_direct_operation_point_count"
                    ),
                    "temporal_repetitive_operation_point_count": row.get(
                        "temporal_repetitive_operation_point_count"
                    ),
                    "temporal_stable_screen_point_count": row.get(
                        "temporal_stable_screen_point_count"
                    ),
                }
            )
    return {
        "schema_version": "privte_evidence_label_audit.v0",
        "records": len(records),
        "label_counts": dict(Counter(row["target_label"] for row in rows)),
        "numeric_feature_summaries": summarize_numeric(rows),
        "categorical_feature_summaries": summarize_categorical(rows),
        "discordant_samples": flagged,
        "per_sample_rows": rows,
    }


def markdown_table(report: dict[str, Any], top_samples: int) -> str:
    labels = [
        label
        for label in ("no_observed_risk", "mild_risk", "moderate_risk", "high_risk")
        if label in report["label_counts"]
    ]
    lines = [
        "# PriVTE Evidence-Label Alignment Audit",
        "",
        f"Records: {report['records']}",
        "",
        "## Label Counts",
        "",
        "| label | count |",
        "|---|---:|",
    ]
    for label, count in report["label_counts"].items():
        lines.append(f"| {label} | {count} |")

    lines.extend(
        [
            "",
            "## Numeric Feature Direction",
            "",
            "| feature | direction | "
            + " | ".join(f"{label} mean" for label in labels)
            + " |",
            "|---|---|"
            + "|".join("---:" for _label in labels)
            + "|",
        ]
    )
    for feature, summary in report["numeric_feature_summaries"].items():
        means = []
        for label in labels:
            mean = summary["by_label"].get(label, {}).get("mean")
            means.append("NA" if mean is None else str(mean))
        lines.append(
            f"| {feature} | {summary['direction']} | " + " | ".join(means) + " |"
        )

    lines.extend(
        [
            "",
            "## Discordant Samples",
            "",
            "| sample_id | target_label | flags | trace_score | active | direct | repetitive | stable |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in report["discordant_samples"][:top_samples]:
        lines.append(
            "| {sample_id} | {target_label} | {flags} | {score} | {active} | {direct} | {rep} | {stable} |".format(
                sample_id=item["sample_id"],
                target_label=item["target_label"],
                flags=", ".join(item["flags"]),
                score=item.get("trace_risk_pattern_score"),
                active=item.get("temporal_active_operation_point_count"),
                direct=item.get("temporal_direct_operation_point_count"),
                rep=item.get("temporal_repetitive_operation_point_count"),
                stable=item.get("temporal_stable_screen_point_count"),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if not args.evidence_jsonl.exists():
        raise SystemExit(f"Evidence JSONL not found: {args.evidence_jsonl}")
    records = read_jsonl(args.evidence_jsonl)
    report = build_report(records)
    run_name = args.run_name or args.evidence_jsonl.stem
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{run_name}.audit.json"
    md_path = args.output_dir / f"{run_name}.audit.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    md_path.write_text(markdown_table(report, args.top_samples), encoding="utf-8")
    print(f"records={report['records']}")
    print(f"json={json_path}")
    print(f"markdown={md_path}")
    print(f"discordant_samples={len(report['discordant_samples'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
