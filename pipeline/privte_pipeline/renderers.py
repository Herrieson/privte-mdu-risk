"""Text renderers for PriVTE evidence records."""

from __future__ import annotations

import json
from typing import Any


def _render_value(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, dict):
        return "{" + _render_dict(value) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(_render_value(item) for item in value) + "]"
    return str(value)


def _render_dict(values: dict[str, Any]) -> str:
    if not values:
        return "none"
    return ", ".join(f"{key}={_render_value(value)}" for key, value in values.items())


def _render_event(event: Any, index: int) -> str:
    if isinstance(event, dict):
        event_id = event.get("event_id") or event.get("window_id") or f"event_{index:02d}"
        proxy = event.get("proxy_evidence", {})
        quality = event.get("quality_metrics", {})
        fields = {
            "order": event.get("window_order"),
            "position": event.get("relative_position"),
            "period": event.get("relative_time_period"),
            "duration": event.get("duration_bin"),
            "trigger": event.get("trigger_type"),
            "motion": proxy.get("motion_pattern"),
            "device_interaction": proxy.get("device_interaction_status"),
            "basis": proxy.get("interaction_evidence_basis"),
            "posture": proxy.get("posture_state"),
            "gaze": proxy.get("gaze_state"),
            "eye_openness": proxy.get("eye_openness_level"),
            "blink_change": proxy.get("blink_rate_change"),
            "facial_actions": proxy.get("facial_action_proxies"),
            "quality": {
                "face": quality.get("face_visibility_level"),
                "hand": quality.get("hand_visibility_level"),
                "device": quality.get("device_visibility_level"),
                "face_mesh": quality.get("face_mesh_visibility_level"),
                "lighting": quality.get("lighting_quality"),
                "confidence": quality.get("event_confidence"),
            },
        }
        fields = {
            key: value
            for key, value in fields.items()
            if value not in (None, [], {}, "not_computed")
        }
        return f"{event_id}: {_render_dict(fields)}"
    return f"event_{index:02d}: {_render_value(event)}"


def _level_phrase(value: Any) -> str:
    phrases = {
        "very_high": "observed in nearly all sampled windows",
        "high": "observed in most sampled windows",
        "medium": "observed in some sampled windows",
        "low": "observed in a small minority of sampled windows",
        "none": "not observed in the sampled windows",
        "not_computed": "not reliably computed",
        "motion_confounded": "confounded by broad frame motion",
        "not_reliably_estimable": "not reliably estimable",
    }
    return phrases.get(str(value), str(value))


def _compact_counts(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return "none"
    return ", ".join(f"{key}={count}" for key, count in sorted(value.items()))


def _global_behavior_summary(global_features: dict[str, Any]) -> dict[str, Any]:
    screen_status = global_features.get("screen_orientation_proxy_status")
    gaze_ratio = global_features.get("screen_gaze_ratio")
    if gaze_ratio is None:
        screen_orientation = (
            "Screen-oriented posture ratio was not reliably estimated"
            f" ({screen_status or 'unknown status'})."
        )
    else:
        screen_orientation = (
            "Screen-oriented posture proxy was estimated from visible face-device "
            f"context; coarse ratio={gaze_ratio}."
        )

    posture_trend = global_features.get("overall_posture_trend", "not_computed")
    posture = {
        "frequent_close_to_screen": (
            "Close-to-screen face-size proxy appeared in multiple sampled windows."
        ),
        "some_close_to_screen": (
            "Close-to-screen face-size proxy appeared in some sampled windows."
        ),
        "mostly_not_close": (
            "Face was visible without a dominant close-to-screen proxy."
        ),
    }.get(str(posture_trend), "Posture trend was not reliably computed.")

    interaction = (
        "Device or screen-like region was "
        f"{_level_phrase(global_features.get('device_visibility_level', 'unknown'))}; "
        "hands were "
        f"{_level_phrase(global_features.get('hand_visibility_level', 'unknown'))}; "
        "selected behavior windows="
        f"{global_features.get('event_window_count', 'unknown')}."
    )

    facial = (
        "FaceMesh was "
        f"{_level_phrase(global_features.get('face_mesh_visibility_level', 'unknown'))}; "
        "eye-closure proxy was "
        f"{_level_phrase(global_features.get('eye_closure_proxy_level', 'unknown'))}; "
        "facial geometry proxy counts="
        f"{_compact_counts(global_features.get('facial_action_proxy_counts', {}))}."
    )

    return {
        "screen_orientation": screen_orientation,
        "posture": posture,
        "interaction_observability": interaction,
        "facial_and_eye_proxy": facial,
        "motion_confounding": (
            "Overall motion confounding was "
            f"{_level_phrase(global_features.get('motion_confounding_level', 'unknown'))}."
        ),
    }


def build_text_evidence(record: dict[str, Any]) -> str:
    package = record["llm_evidence_package"]
    task = package["task"]
    global_features = package.get("global_features", {})
    text_package = {
        "schema_version": "privte_llm_text_evidence.v2",
        "sample_id": record["sample_id"],
        "task": {
            "name": task["name"],
            "instruction": task["instruction"],
            "allowed_risk_levels": task["allowed_risk_levels"],
            "boundary": (
                "Risk screening from privacy-filtered video-derived text only; "
                "not a medical diagnosis, emotion diagnosis, or addiction conclusion."
            ),
        },
        "observation_scope": package.get("observation_scope", {}),
        "session_metadata": package.get("session_metadata", {}),
        "global_behavior_summary": _global_behavior_summary(global_features),
        "global_features": {
            "screen_gaze_ratio": global_features.get("screen_gaze_ratio"),
            "screen_orientation_proxy_status": global_features.get(
                "screen_orientation_proxy_status"
            ),
            "max_continuous_gaze_duration_minutes": global_features.get(
                "max_continuous_gaze_duration_minutes"
            ),
            "overall_posture_trend": global_features.get("overall_posture_trend"),
            "blink_rate_trend": global_features.get("blink_rate_trend"),
            "visual_observability": {
                "device_or_screen_like_region": _level_phrase(
                    global_features.get("device_visibility_level")
                ),
                "hands": _level_phrase(global_features.get("hand_visibility_level")),
                "face_mesh": _level_phrase(
                    global_features.get("face_mesh_visibility_level")
                ),
            },
            "stable_screen_context": _level_phrase(
                global_features.get("stable_screen_context_level")
            ),
            "motion_confounding": _level_phrase(
                global_features.get("motion_confounding_level")
            ),
            "event_window_count": global_features.get("event_window_count"),
        },
        "event_windows": package.get("event_windows", []),
        "quality_summary": package.get("quality_summary", {}),
        "limitations": package.get("limitations", []),
        "missing_information": package.get("missing_information", []),
        "privacy_processing_summary": package.get("privacy_processing_summary", {}),
        "requested_model_output": package.get("requested_model_output", {}),
    }
    return json.dumps(text_package, ensure_ascii=False, indent=2)
