#!/usr/bin/env python3
"""Diagnose PriVTE evidence and text-only model predictions.

This script is intentionally lightweight and schema-aware rather than tied to a
final algorithm. It helps inspect whether current PriVTE evidence fields are
discriminative enough, whether a model beats simple baselines, and which fields
are common in false positives / false negatives.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RISK_LEVELS = [
    "no_observed_risk",
    "mild_risk",
    "moderate_risk",
    "high_risk",
    "insufficient_evidence",
]

DEFAULT_DIAGNOSTIC_FIELDS = [
    "session.duration_bin",
    "session.event_window_count",
    "global.overall_posture_trend",
    "global.interaction_intensity",
    "global.repetitive_operation_level",
    "global.motion_confounding_level",
    "global.screen_gaze_ratio_bin",
    "global.blink_rate_trend",
    "global.device_visibility_level",
    "global.hand_visibility_level",
    "global.face_mesh_visibility_level",
    "global.stable_screen_context_level",
    "global.eye_closure_proxy_level",
    "quality.overall_data_sufficiency",
    "quality.face_observability",
    "quality.hand_observability",
    "quality.device_observability",
    "quality.gaze_estimation_quality",
    "quality.motion_confounding_level",
    "state.dominant_session_state",
    "state.engagement_trajectory",
    "state.temporal_interaction_pattern",
    "state.passive_screen_window_count",
    "state.stable_screen_context_window_count",
    "state.longest_passive_screen_streak",
    "state.longest_confirmed_interaction_streak",
    "state.unique_window_state_count",
    "state.behavior_episode_count",
    "policy.behavior_event_count",
    "policy.auxiliary_observation_count",
    "policy.quality_only_observation_count",
    "policy.weak_proxy_observation_count",
    "policy.episode_detector_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose PriVTE evidence distributions and optional LLM prediction "
            "errors."
        )
    )
    parser.add_argument("--evidence-jsonl", type=Path, required=True)
    parser.add_argument(
        "--predictions-jsonl",
        type=Path,
        default=None,
        help="Optional predictions JSONL from experiments/llm/run_llm_baseline.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for diagnosis_report.json and CSV tables.",
    )
    parser.add_argument(
        "--max-error-examples",
        type=int,
        default=30,
        help="Maximum error examples embedded in the JSON report.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_json(path: Path, item: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(item, file, ensure_ascii=False, indent=2)
        file.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def latest_predictions_by_sample(
    prediction_records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    latest = {}
    for item in prediction_records:
        sample_id = item.get("sample_id")
        if sample_id:
            latest[str(sample_id)] = item
    return latest


def prediction_risk_level(prediction_record: dict[str, Any] | None) -> str | None:
    if not prediction_record:
        return None
    prediction = prediction_record.get("prediction")
    if not isinstance(prediction, dict):
        return None
    risk_level = prediction.get("risk_level")
    return risk_level if risk_level in RISK_LEVELS else None


def target_label(record: dict[str, Any]) -> str | None:
    label = record.get("target_label", {})
    if not label.get("available"):
        return None
    risk_level = label.get("risk_level")
    return risk_level if risk_level in RISK_LEVELS else None


def preprocessor_evidence(record: dict[str, Any]) -> dict[str, Any]:
    return (
        record.get("evidence", {})
        .get("feature_blocks", {})
        .get("preprocessor_evidence", {})
    )


def stateful_behavior_summary(record: dict[str, Any]) -> dict[str, Any]:
    global_features = preprocessor_evidence(record).get("global_features", {})
    stateful = global_features.get("stateful_behavior_summary", {})
    return stateful if isinstance(stateful, dict) else {}


def dominant_window_state_counter(stateful: dict[str, Any]) -> Counter[str]:
    counts = (
        stateful.get("state_space_summary", {}).get("dominant_window_state_counts", {})
    )
    if not isinstance(counts, dict):
        return Counter()
    return Counter({str(key): int(value) for key, value in counts.items()})


def detector_policy_summary(global_features: dict[str, Any]) -> dict[str, Any]:
    summary = global_features.get("detector_policy_summary", {})
    if isinstance(summary, dict):
        return summary
    stateful = global_features.get("stateful_behavior_summary", {})
    if isinstance(stateful, dict) and isinstance(
        stateful.get("detector_policy_summary"),
        dict,
    ):
        return stateful["detector_policy_summary"]
    return {}


def episode_detector_counter(stateful: dict[str, Any]) -> Counter[str]:
    detectors = stateful.get("episode_detectors", [])
    counts: Counter[str] = Counter()
    if not isinstance(detectors, list):
        return counts
    for detector in detectors:
        if isinstance(detector, dict) and detector.get("detector_name"):
            counts[str(detector["detector_name"])] += 1
    return counts


def event_counter(events: list[dict[str, Any]], key: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for event in events:
        value = event.get(key)
        if value is not None:
            counts[str(value)] += 1
    return counts


def event_quality_counter(events: list[dict[str, Any]], key: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for event in events:
        value = event.get("quality_metrics", {}).get(key)
        if value is not None:
            counts[str(value)] += 1
    return counts


def compact_counter(counter: Counter[str]) -> str:
    if not counter:
        return ""
    return ";".join(f"{key}={value}" for key, value in sorted(counter.items()))


def list_preview(values: Any, limit: int = 3) -> str:
    if not isinstance(values, list):
        return ""
    return " | ".join(str(value).replace("\n", " ") for value in values[:limit])


def value_from_field(
    *,
    session: dict[str, Any],
    global_features: dict[str, Any],
    quality: dict[str, Any],
    events: list[dict[str, Any]],
    field: str,
) -> Any:
    if field.startswith("session."):
        return session.get(field.split(".", 1)[1])
    if field.startswith("global."):
        return global_features.get(field.split(".", 1)[1])
    if field.startswith("quality."):
        return quality.get(field.split(".", 1)[1])
    if field.startswith("event_trigger_count."):
        trigger = field.split(".", 1)[1]
        return event_counter(events, "trigger_type").get(trigger, 0)
    if field.startswith("event_confidence_count."):
        confidence = field.split(".", 1)[1]
        return event_quality_counter(events, "event_confidence").get(confidence, 0)
    if field.startswith("state."):
        stateful = global_features.get("stateful_behavior_summary", {})
        if not isinstance(stateful, dict):
            return None
        state_space = stateful.get("state_space_summary", {})
        temporal = stateful.get("temporal_behavior_summary", {})
        state_key = field.split(".", 1)[1]
        if isinstance(state_space, dict) and state_key in state_space:
            return state_space.get(state_key)
        if isinstance(temporal, dict) and state_key in temporal:
            return temporal.get(state_key)
    if field.startswith("policy."):
        policy = detector_policy_summary(global_features)
        return policy.get(field.split(".", 1)[1])
    return None


def build_sample_rows(
    evidence_records: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for record in evidence_records:
        sample_id = str(record.get("sample_id", ""))
        prediction_record = predictions.get(sample_id)
        raw_prediction = (
            prediction_record.get("prediction", {})
            if isinstance(prediction_record, dict)
            else {}
        )
        prediction = raw_prediction if isinstance(raw_prediction, dict) else {}
        preprocessor = preprocessor_evidence(record)
        session = preprocessor.get("session_metadata", {})
        global_features = preprocessor.get("global_features", {})
        quality = preprocessor.get("quality_summary", {})
        events = preprocessor.get("event_windows", [])
        stateful = global_features.get("stateful_behavior_summary", {})
        if not isinstance(stateful, dict):
            stateful = {}
        state_space = stateful.get("state_space_summary", {})
        state_space = state_space if isinstance(state_space, dict) else {}
        temporal = stateful.get("temporal_behavior_summary", {})
        temporal = temporal if isinstance(temporal, dict) else {}
        evidence_graph = stateful.get("evidence_graph", {})
        evidence_graph = evidence_graph if isinstance(evidence_graph, dict) else {}
        detector_policy = detector_policy_summary(global_features)
        evidence_role_counts = Counter(
            {
                str(key): int(value)
                for key, value in detector_policy.get(
                    "evidence_role_counts",
                    {},
                ).items()
            }
        )
        suppressed_event_triggers = Counter(
            {
                str(key): int(value)
                for key, value in detector_policy.get(
                    "suppressed_event_triggers",
                    {},
                ).items()
            }
        )
        trigger_counts = event_counter(events, "trigger_type")
        confidence_counts = event_quality_counter(events, "event_confidence")
        dominant_state_counts = dominant_window_state_counter(stateful)
        episode_detector_counts = episode_detector_counter(stateful)
        gold = target_label(record)
        pred = prediction_risk_level(prediction_record)
        row = {
            "sample_id": sample_id,
            "target_label": gold or "",
            "predicted_label": pred or "",
            "is_valid_prediction": bool(pred),
            "is_correct": bool(gold and pred and gold == pred),
            "prediction_confidence": prediction.get("confidence", ""),
            "prediction_error": (
                prediction_record.get("error", "") if prediction_record else ""
            ),
            "needs_human_review": prediction.get("needs_human_review", ""),
            "duration_bin": session.get("duration_bin", ""),
            "total_valid_duration_minutes": session.get(
                "total_valid_duration_minutes", ""
            ),
            "analyzed_window_count": session.get("analyzed_window_count", ""),
            "event_window_count": session.get("event_window_count", len(events)),
            "trigger_counts": compact_counter(trigger_counts),
            "event_confidence_counts": compact_counter(confidence_counts),
            "behavior_event_count": detector_policy.get("behavior_event_count", ""),
            "auxiliary_observation_count": detector_policy.get(
                "auxiliary_observation_count",
                "",
            ),
            "quality_only_observation_count": detector_policy.get(
                "quality_only_observation_count",
                "",
            ),
            "weak_proxy_observation_count": detector_policy.get(
                "weak_proxy_observation_count",
                "",
            ),
            "episode_detector_count": detector_policy.get(
                "episode_detector_count",
                "",
            ),
            "evidence_role_counts": compact_counter(evidence_role_counts),
            "suppressed_event_triggers": compact_counter(suppressed_event_triggers),
            "episode_detector_counts": compact_counter(episode_detector_counts),
            "overall_posture_trend": global_features.get("overall_posture_trend", ""),
            "interaction_intensity": global_features.get("interaction_intensity", ""),
            "repetitive_operation_level": global_features.get(
                "repetitive_operation_level", ""
            ),
            "motion_confounding_level": global_features.get(
                "motion_confounding_level", ""
            ),
            "screen_gaze_ratio_bin": global_features.get("screen_gaze_ratio_bin", ""),
            "stable_screen_context_level": global_features.get(
                "stable_screen_context_level", ""
            ),
            "blink_rate_trend": global_features.get("blink_rate_trend", ""),
            "eye_closure_proxy_level": global_features.get(
                "eye_closure_proxy_level", ""
            ),
            "face_observability": quality.get("face_observability", ""),
            "hand_observability": quality.get("hand_observability", ""),
            "device_observability": quality.get("device_observability", ""),
            "gaze_estimation_quality": quality.get("gaze_estimation_quality", ""),
            "quality_motion_confounding_level": quality.get(
                "motion_confounding_level", ""
            ),
            "negative_evidence_summary": compact_counter(
                Counter(global_features.get("negative_evidence_summary", []))
            ),
            "evidence_balance": global_features.get("evidence_balance", ""),
            "confirmed_interaction_window_count": global_features.get(
                "confirmed_interaction_window_count", ""
            ),
            "possible_interaction_window_count": global_features.get(
                "possible_interaction_window_count", ""
            ),
            "device_region_motion_window_count": global_features.get(
                "device_region_motion_window_count", ""
            ),
            "strong_repetitive_proxy_window_count": global_features.get(
                "strong_repetitive_proxy_window_count", ""
            ),
            "dominant_session_state": temporal.get("dominant_session_state", ""),
            "engagement_trajectory": temporal.get("engagement_trajectory", ""),
            "temporal_interaction_pattern": temporal.get(
                "temporal_interaction_pattern", ""
            ),
            "passive_screen_window_count": temporal.get(
                "passive_screen_window_count", ""
            ),
            "stable_screen_context_window_count": temporal.get(
                "stable_screen_context_window_count", ""
            ),
            "longest_passive_screen_streak": temporal.get(
                "longest_passive_screen_streak", ""
            ),
            "longest_confirmed_interaction_streak": temporal.get(
                "longest_confirmed_interaction_streak", ""
            ),
            "unique_window_state_count": state_space.get(
                "unique_window_state_count", ""
            ),
            "behavior_episode_count": state_space.get("behavior_episode_count", ""),
            "longest_repeated_state_streak": state_space.get(
                "longest_repeated_state_streak", ""
            ),
            "positive_evidence_fact_count": state_space.get(
                "positive_evidence_fact_count", ""
            ),
            "weak_proxy_evidence_fact_count": state_space.get(
                "weak_proxy_evidence_fact_count", ""
            ),
            "counter_evidence_fact_count": state_space.get(
                "counter_evidence_fact_count", ""
            ),
            "quality_limitation_fact_count": state_space.get(
                "quality_limitation_fact_count", ""
            ),
            "dominant_window_state_counts": compact_counter(dominant_state_counts),
            "positive_evidence_preview": list_preview(
                evidence_graph.get("positive_evidence_facts")
            ),
            "counter_evidence_preview": list_preview(
                evidence_graph.get("counter_evidence_facts")
            ),
            "supporting_evidence_preview": list_preview(
                prediction.get("supporting_evidence")
            ),
            "uncertainty_preview": list_preview(
                prediction.get("uncertainty_or_counter_evidence")
            ),
        }
        for trigger, count in trigger_counts.items():
            row[f"event_trigger_count.{trigger}"] = count
        for confidence, count in confidence_counts.items():
            row[f"event_confidence_count.{confidence}"] = count
        for state_name, count in dominant_state_counts.items():
            row[f"window_state_count.{state_name}"] = count
        for role, count in evidence_role_counts.items():
            row[f"evidence_role_count.{role}"] = count
        for trigger, count in suppressed_event_triggers.items():
            row[f"suppressed_event_trigger_count.{trigger}"] = count
        for detector, count in episode_detector_counts.items():
            row[f"episode_detector_count.{detector}"] = count
        rows.append(row)
    return rows


def metric_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [
        row
        for row in rows
        if row.get("target_label") in RISK_LEVELS
        and row.get("predicted_label") in RISK_LEVELS
    ]
    confusion = {gold: {pred: 0 for pred in RISK_LEVELS} for gold in RISK_LEVELS}
    for row in valid_rows:
        confusion[row["target_label"]][row["predicted_label"]] += 1

    total = len(valid_rows)
    correct = sum(confusion[label][label] for label in RISK_LEVELS)
    per_label = {}
    observed_f1 = []
    for label in RISK_LEVELS:
        tp = confusion[label][label]
        fp = sum(confusion[gold][label] for gold in RISK_LEVELS if gold != label)
        fn = sum(confusion[label][pred] for pred in RISK_LEVELS if pred != label)
        support = sum(confusion[label].values())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        if support:
            observed_f1.append(f1)
        per_label[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    target_counts = Counter(row.get("target_label") for row in rows)
    predicted_counts = Counter(row.get("predicted_label") or "<invalid>" for row in rows)
    valid_target_counts = Counter(row.get("target_label") for row in valid_rows)
    majority_count = max(valid_target_counts.values()) if valid_target_counts else 0
    majority_label = (
        valid_target_counts.most_common(1)[0][0] if valid_target_counts else None
    )
    majority_accuracy = majority_count / total if total else 0.0
    accuracy = correct / total if total else 0.0
    return {
        "records": len(rows),
        "valid_predictions": total,
        "invalid_predictions": len(rows) - total,
        "accuracy": round(accuracy, 4),
        "macro_f1_observed_labels": (
            round(sum(observed_f1) / len(observed_f1), 4) if observed_f1 else 0.0
        ),
        "majority_baseline": {
            "label": majority_label,
            "accuracy": round(majority_accuracy, 4),
            "model_minus_majority_accuracy": round(accuracy - majority_accuracy, 4),
        },
        "target_label_counts": dict(target_counts),
        "predicted_label_counts": dict(predicted_counts),
        "error_counts": dict(
            Counter(row.get("prediction_error") for row in rows if row.get("prediction_error"))
        ),
        "per_label": per_label,
        "confusion": confusion,
    }


def sorted_json_counter(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): value for key, value in counter.most_common()}


def feature_distributions(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    fields = [
        "duration_bin",
        "event_window_count",
        "overall_posture_trend",
        "interaction_intensity",
        "repetitive_operation_level",
        "motion_confounding_level",
        "screen_gaze_ratio_bin",
        "stable_screen_context_level",
        "blink_rate_trend",
        "eye_closure_proxy_level",
        "face_observability",
        "hand_observability",
        "device_observability",
        "gaze_estimation_quality",
        "quality_motion_confounding_level",
        "negative_evidence_summary",
        "evidence_balance",
        "confirmed_interaction_window_count",
        "possible_interaction_window_count",
        "device_region_motion_window_count",
        "strong_repetitive_proxy_window_count",
        "trigger_counts",
        "event_confidence_counts",
        "behavior_event_count",
        "auxiliary_observation_count",
        "quality_only_observation_count",
        "weak_proxy_observation_count",
        "episode_detector_count",
        "evidence_role_counts",
        "suppressed_event_triggers",
        "episode_detector_counts",
        "dominant_session_state",
        "engagement_trajectory",
        "temporal_interaction_pattern",
        "passive_screen_window_count",
        "stable_screen_context_window_count",
        "longest_passive_screen_streak",
        "longest_confirmed_interaction_streak",
        "unique_window_state_count",
        "behavior_episode_count",
        "longest_repeated_state_streak",
        "positive_evidence_fact_count",
        "weak_proxy_evidence_fact_count",
        "counter_evidence_fact_count",
        "quality_limitation_fact_count",
        "dominant_window_state_counts",
    ]
    return {field: sorted_json_counter(Counter(row.get(field) for row in rows)) for field in fields}


def state_space_report(evidence_records: list[dict[str, Any]]) -> dict[str, Any]:
    full_signatures = Counter()
    dominant_state_counts = Counter()
    component_value_counts: dict[str, Counter[str]] = defaultdict(Counter)
    session_fields: dict[str, Counter[str]] = defaultdict(Counter)

    for record in evidence_records:
        stateful = stateful_behavior_summary(record)
        state_space = stateful.get("state_space_summary", {})
        temporal = stateful.get("temporal_behavior_summary", {})
        if isinstance(state_space, dict):
            for key in (
                "unique_window_state_count",
                "behavior_episode_count",
                "longest_repeated_state_streak",
            ):
                session_fields[key][str(state_space.get(key))] += 1
        if isinstance(temporal, dict):
            for key in (
                "dominant_session_state",
                "engagement_trajectory",
                "temporal_interaction_pattern",
                "passive_screen_window_count",
                "stable_screen_context_window_count",
                "longest_passive_screen_streak",
                "longest_confirmed_interaction_streak",
            ):
                session_fields[key][str(temporal.get(key))] += 1
        for state in stateful.get("per_window_states", []):
            if not isinstance(state, dict):
                continue
            dominant = str(state.get("dominant_behavior_state"))
            dominant_state_counts[dominant] += 1
            components = state.get("component_states", {})
            if isinstance(components, dict):
                signature = [dominant]
                for key in sorted(components):
                    value = str(components.get(key))
                    component_value_counts[key][value] += 1
                    signature.append(f"{key}={value}")
                full_signatures[tuple(signature)] += 1

    return {
        "total_window_states": sum(dominant_state_counts.values()),
        "unique_window_signatures": len(full_signatures),
        "dominant_window_state_counts": sorted_json_counter(dominant_state_counts),
        "component_value_counts": {
            key: sorted_json_counter(counter)
            for key, counter in sorted(component_value_counts.items())
        },
        "session_state_field_distributions": {
            key: sorted_json_counter(counter)
            for key, counter in sorted(session_fields.items())
        },
        "most_common_window_signatures": [
            {
                "count": count,
                "signature": dict(
                    item.split("=", 1)
                    for item in signature[1:]
                    if "=" in item
                )
                | {"dominant_behavior_state": signature[0]},
            }
            for signature, count in full_signatures.most_common(20)
        ],
    }


def detector_policy_report(evidence_records: list[dict[str, Any]]) -> dict[str, Any]:
    role_counts = Counter()
    suppressed_trigger_counts = Counter()
    episode_detector_counts = Counter()
    policy_fields: dict[str, Counter[str]] = defaultdict(Counter)

    for record in evidence_records:
        global_features = preprocessor_evidence(record).get("global_features", {})
        policy = detector_policy_summary(global_features)
        stateful = stateful_behavior_summary(record)
        role_counts.update(policy.get("evidence_role_counts", {}))
        suppressed_trigger_counts.update(policy.get("suppressed_event_triggers", {}))
        episode_detector_counts.update(episode_detector_counter(stateful))
        for key in (
            "behavior_event_count",
            "auxiliary_observation_count",
            "quality_only_observation_count",
            "weak_proxy_observation_count",
            "episode_detector_count",
        ):
            policy_fields[key][str(policy.get(key))] += 1

    return {
        "policy_field_distributions": {
            key: sorted_json_counter(counter)
            for key, counter in sorted(policy_fields.items())
        },
        "evidence_role_counts": sorted_json_counter(role_counts),
        "suppressed_event_triggers": sorted_json_counter(suppressed_trigger_counts),
        "episode_detector_counts": sorted_json_counter(episode_detector_counts),
    }


def crosstab_rows(
    evidence_records: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    diagnostic_fields: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for field in diagnostic_fields:
        buckets: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "total": 0,
                "target_label_counts": Counter(),
                "predicted_label_counts": Counter(),
                "valid_predictions": 0,
                "correct_predictions": 0,
                "sample_ids": [],
            }
        )
        for record in evidence_records:
            sample_id = str(record.get("sample_id", ""))
            preprocessor = preprocessor_evidence(record)
            session = preprocessor.get("session_metadata", {})
            global_features = preprocessor.get("global_features", {})
            quality = preprocessor.get("quality_summary", {})
            events = preprocessor.get("event_windows", [])
            value = value_from_field(
                session=session,
                global_features=global_features,
                quality=quality,
                events=events,
                field=field,
            )
            value_key = str(value)
            bucket = buckets[value_key]
            bucket["total"] += 1
            bucket["sample_ids"].append(sample_id)
            gold = target_label(record) or "<missing>"
            pred = prediction_risk_level(predictions.get(sample_id)) or "<invalid>"
            bucket["target_label_counts"][gold] += 1
            bucket["predicted_label_counts"][pred] += 1
            if gold in RISK_LEVELS and pred in RISK_LEVELS:
                bucket["valid_predictions"] += 1
                if gold == pred:
                    bucket["correct_predictions"] += 1
        for value, bucket in sorted(
            buckets.items(), key=lambda item: (-item[1]["total"], item[0])
        ):
            valid = bucket["valid_predictions"]
            correct = bucket["correct_predictions"]
            rows.append(
                {
                    "field": field,
                    "value": value,
                    "total": bucket["total"],
                    "target_label_counts": json.dumps(
                        dict(bucket["target_label_counts"]),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "predicted_label_counts": json.dumps(
                        dict(bucket["predicted_label_counts"]),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "valid_predictions": valid,
                    "accuracy": round(correct / valid, 4) if valid else "",
                    "sample_ids_preview": ";".join(bucket["sample_ids"][:10]),
                }
            )
    return rows


def error_pattern_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [
        row
        for row in rows
        if row.get("target_label") in RISK_LEVELS
        and row.get("predicted_label") in RISK_LEVELS
        and row.get("target_label") != row.get("predicted_label")
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in errors:
        grouped[f"{row['target_label']} -> {row['predicted_label']}"].append(row)

    summary = {}
    for key, items in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        summary[key] = {
            "count": len(items),
            "duration_bin": dict(Counter(row.get("duration_bin") for row in items)),
            "event_window_count": dict(
                Counter(str(row.get("event_window_count")) for row in items)
            ),
            "overall_posture_trend": dict(
                Counter(row.get("overall_posture_trend") for row in items)
            ),
            "interaction_intensity": dict(
                Counter(row.get("interaction_intensity") for row in items)
            ),
            "repetitive_operation_level": dict(
                Counter(row.get("repetitive_operation_level") for row in items)
            ),
            "motion_confounding_level": dict(
                Counter(row.get("motion_confounding_level") for row in items)
            ),
            "hand_observability": dict(
                Counter(row.get("hand_observability") for row in items)
            ),
            "device_observability": dict(
                Counter(row.get("device_observability") for row in items)
            ),
            "sample_ids": [row.get("sample_id") for row in items],
        }
    return summary


def output_directory(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return args.output_dir
    if args.predictions_jsonl:
        return args.predictions_jsonl.parent / "diagnosis"
    return args.evidence_jsonl.parent / "diagnosis"


def main() -> int:
    args = parse_args()
    evidence_records = read_jsonl(args.evidence_jsonl)
    prediction_records = (
        read_jsonl(args.predictions_jsonl) if args.predictions_jsonl else []
    )
    predictions = latest_predictions_by_sample(prediction_records)
    rows = build_sample_rows(evidence_records, predictions)
    dynamic_fields = sorted(
        {
            key
            for row in rows
            for key in row
            if key.startswith("event_trigger_count.")
            or key.startswith("event_confidence_count.")
            or key.startswith("window_state_count.")
            or key.startswith("evidence_role_count.")
            or key.startswith("suppressed_event_trigger_count.")
            or key.startswith("episode_detector_count.")
        }
    )
    diagnostic_fields = DEFAULT_DIAGNOSTIC_FIELDS + dynamic_fields
    crosstabs = crosstab_rows(evidence_records, predictions, diagnostic_fields)
    metrics = metric_report(rows) if prediction_records else {}
    errors = [
        row
        for row in rows
        if row.get("target_label") in RISK_LEVELS
        and row.get("predicted_label") in RISK_LEVELS
        and row.get("target_label") != row.get("predicted_label")
    ]

    output_dir = output_directory(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_sample_fields = [
        "sample_id",
        "target_label",
        "predicted_label",
        "is_valid_prediction",
        "is_correct",
        "prediction_confidence",
        "prediction_error",
        "needs_human_review",
        "duration_bin",
        "total_valid_duration_minutes",
        "analyzed_window_count",
        "event_window_count",
        "trigger_counts",
        "event_confidence_counts",
        "behavior_event_count",
        "auxiliary_observation_count",
        "quality_only_observation_count",
        "weak_proxy_observation_count",
        "episode_detector_count",
        "evidence_role_counts",
        "suppressed_event_triggers",
        "episode_detector_counts",
        "overall_posture_trend",
        "interaction_intensity",
        "repetitive_operation_level",
        "motion_confounding_level",
        "screen_gaze_ratio_bin",
        "stable_screen_context_level",
        "blink_rate_trend",
        "eye_closure_proxy_level",
        "face_observability",
        "hand_observability",
        "device_observability",
        "gaze_estimation_quality",
        "quality_motion_confounding_level",
        "negative_evidence_summary",
        "evidence_balance",
        "confirmed_interaction_window_count",
        "possible_interaction_window_count",
        "device_region_motion_window_count",
        "strong_repetitive_proxy_window_count",
        "dominant_session_state",
        "engagement_trajectory",
        "temporal_interaction_pattern",
        "passive_screen_window_count",
        "stable_screen_context_window_count",
        "longest_passive_screen_streak",
        "longest_confirmed_interaction_streak",
        "unique_window_state_count",
        "behavior_episode_count",
        "longest_repeated_state_streak",
        "positive_evidence_fact_count",
        "weak_proxy_evidence_fact_count",
        "counter_evidence_fact_count",
        "quality_limitation_fact_count",
        "dominant_window_state_counts",
        "positive_evidence_preview",
        "counter_evidence_preview",
        "supporting_evidence_preview",
        "uncertainty_preview",
    ]
    sample_fields = base_sample_fields + dynamic_fields
    write_csv(output_dir / "sample_diagnosis.csv", rows, sample_fields)
    write_csv(output_dir / "error_cases.csv", errors, sample_fields)
    write_csv(
        output_dir / "feature_crosstabs.csv",
        crosstabs,
        [
            "field",
            "value",
            "total",
            "target_label_counts",
            "predicted_label_counts",
            "valid_predictions",
            "accuracy",
            "sample_ids_preview",
        ],
    )

    report = {
        "schema_version": "privte_result_diagnosis.v0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "evidence_jsonl": args.evidence_jsonl.as_posix(),
            "predictions_jsonl": (
                args.predictions_jsonl.as_posix() if args.predictions_jsonl else None
            ),
        },
        "outputs": {
            "sample_diagnosis_csv": (output_dir / "sample_diagnosis.csv").as_posix(),
            "error_cases_csv": (output_dir / "error_cases.csv").as_posix(),
            "feature_crosstabs_csv": (
                output_dir / "feature_crosstabs.csv"
            ).as_posix(),
        },
        "metrics": metrics,
        "feature_distributions": feature_distributions(rows),
        "state_space_report": state_space_report(evidence_records),
        "detector_policy_report": detector_policy_report(evidence_records),
        "error_pattern_summary": error_pattern_summary(rows),
        "error_examples": errors[: args.max_error_examples],
    }
    write_json(output_dir / "diagnosis_report.json", report)

    print(
        json.dumps(
            {
                "records": len(rows),
                "prediction_records": len(prediction_records),
                "metrics": metrics,
                "output_dir": output_dir.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
