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
        "not_computed": "unavailable",
        "motion_confounded": "confounded by broad frame motion",
        "not_reliably_estimable": "unavailable",
    }
    return phrases.get(str(value), str(value))


def _compact_counts(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return "none"
    return ", ".join(f"{key}={count}" for key, count in sorted(value.items()))


def _limited_list(values: Any, limit: int = 12) -> list[Any]:
    if not isinstance(values, list):
        return []
    return values[:limit]


def _compact_window_state(state: Any) -> dict[str, Any]:
    if not isinstance(state, dict):
        return {}
    components = state.get("component_states", {})
    quality = state.get("quality_gates", {})
    counts = state.get("local_proxy_counts", {})
    components = components if isinstance(components, dict) else {}
    quality = quality if isinstance(quality, dict) else {}
    counts = counts if isinstance(counts, dict) else {}
    return {
        "order": state.get("window_order"),
        "position": state.get("relative_position"),
        "dominant_state": state.get("dominant_behavior_state"),
        "engagement": components.get("engagement_state"),
        "rhythm": components.get("interaction_rhythm"),
        "screen": components.get("screen_orientation"),
        "posture": components.get("posture_state"),
        "motion": components.get("motion_dominance")
        or components.get("motion_confounding"),
        "confidence": quality.get("state_confidence"),
        "localized_bursts": counts.get("local_motion_burst_count"),
    }


def _important_window_states(stateful: dict[str, Any], limit: int = 6) -> list[dict[str, Any]]:
    states = stateful.get("per_window_states", [])
    if not isinstance(states, list):
        return []
    priority = {
        "confirmed_repetitive_interaction_proxy": 0,
        "confirmed_interaction_proxy": 1,
        "possible_repeated_interaction_proxy": 2,
        "possible_interaction_proxy": 3,
        "close_posture_passive_screen_proxy": 4,
        "passive_screen_oriented_proxy": 5,
        "weak_device_region_motion_proxy": 6,
        "quality_limited_unobservable": 7,
        "device_visible_no_confirmed_engagement": 8,
        "no_relevant_device_behavior_observed": 9,
    }
    ranked = sorted(
        [item for item in states if isinstance(item, dict)],
        key=lambda item: (
            priority.get(str(item.get("dominant_behavior_state")), 99),
            item.get("window_order", 999),
        ),
    )
    selected = []
    seen_orders = set()
    for item in ranked:
        order = item.get("window_order")
        if order in seen_orders:
            continue
        selected.append(_compact_window_state(item))
        seen_orders.add(order)
        if len(selected) >= limit:
            break
    return selected


def _compact_episode(episode: Any) -> dict[str, Any]:
    if not isinstance(episode, dict):
        return {}
    return {
        "state": episode.get("dominant_behavior_state"),
        "window_count": episode.get("window_count"),
        "start_order": episode.get("window_order_start"),
        "end_order": episode.get("window_order_end"),
        "position_counts": episode.get("relative_position_counts", {}),
        "confidence": episode.get("episode_confidence"),
    }


def _important_episodes(stateful: dict[str, Any], limit: int = 6) -> list[dict[str, Any]]:
    episodes = stateful.get("behavior_episodes", [])
    if not isinstance(episodes, list):
        return []
    ranked = sorted(
        [item for item in episodes if isinstance(item, dict)],
        key=lambda item: (-int(item.get("window_count", 0) or 0), item.get("window_order_start", 999)),
    )
    return [_compact_episode(item) for item in ranked[:limit]]


def _compact_episode_detector(detector: Any) -> dict[str, Any]:
    if not isinstance(detector, dict):
        return {}
    return {
        "name": detector.get("detector_name"),
        "role": detector.get("evidence_role"),
        "strength": detector.get("detector_strength"),
        "window_count": detector.get("window_count"),
        "supporting_components": detector.get("supporting_components", []),
    }


def _compact_episode_detectors(
    stateful: dict[str, Any],
    limit: int = 8,
) -> list[dict[str, Any]]:
    detectors = stateful.get("episode_detectors", [])
    if not isinstance(detectors, list):
        return []
    role_priority = {
        "strong_behavior_evidence": 0,
        "behavior_context": 1,
        "counter_evidence": 2,
        "weak_proxy_evidence": 3,
        "quality_limitation": 4,
        "auxiliary_context": 5,
    }
    strength_priority = {"high": 0, "medium": 1, "low": 2}
    ranked = sorted(
        [item for item in detectors if isinstance(item, dict)],
        key=lambda item: (
            role_priority.get(str(item.get("evidence_role")), 99),
            strength_priority.get(str(item.get("detector_strength")), 99),
            -int(item.get("window_count", 0) or 0),
            str(item.get("detector_name")),
        ),
    )
    return [_compact_episode_detector(item) for item in ranked[:limit]]


def _detector_policy_context(stateful: dict[str, Any]) -> dict[str, Any]:
    summary = stateful.get("detector_policy_summary", {})
    evidence_graph = stateful.get("evidence_graph", {})
    summary = summary if isinstance(summary, dict) else {}
    evidence_graph = evidence_graph if isinstance(evidence_graph, dict) else {}
    if not summary and not evidence_graph:
        return {}
    return {
        "summary": summary,
        "interpretation": (
            "Only behavior_event_count windows are treated as behavior events; "
            "suppressed triggers are retained as weak proxy, auxiliary context, "
            "or quality limitations."
        ),
        "episode_detectors": _compact_episode_detectors(stateful),
        "auxiliary_observation_facts": _limited_list(
            evidence_graph.get("auxiliary_observation_facts", []), limit=6
        ),
    }


def _stateful_narrative(stateful: dict[str, Any]) -> dict[str, Any]:
    if not stateful:
        return {}
    temporal = stateful.get("temporal_behavior_summary", {})
    state_space = stateful.get("state_space_summary", {})
    evidence_graph = stateful.get("evidence_graph", {})
    return {
        "dominant_session_state": temporal.get("dominant_session_state"),
        "engagement_trajectory": temporal.get("engagement_trajectory"),
        "temporal_interaction_pattern": temporal.get("temporal_interaction_pattern"),
        "passive_screen_window_count": temporal.get("passive_screen_window_count"),
        "stable_screen_context_window_count": temporal.get(
            "stable_screen_context_window_count"
        ),
        "longest_passive_screen_streak": temporal.get(
            "longest_passive_screen_streak"
        ),
        "longest_confirmed_interaction_streak": temporal.get(
            "longest_confirmed_interaction_streak"
        ),
        "window_state_diversity": (
            f"{state_space.get('unique_window_state_count', 0)} unique symbolic "
            f"states across {state_space.get('window_state_count', 0)} sampled windows"
        ),
        "episode_count": state_space.get("behavior_episode_count"),
        "positive_evidence": _limited_list(
            evidence_graph.get("positive_evidence_facts", []), limit=6
        ),
        "weak_proxy_evidence": _limited_list(
            evidence_graph.get("weak_proxy_evidence_facts", []), limit=6
        ),
        "counter_evidence": _limited_list(
            evidence_graph.get("counter_evidence_facts", []), limit=6
        ),
        "quality_limitations": _limited_list(
            evidence_graph.get("quality_limitation_facts", []), limit=6
        ),
        "auxiliary_observations": _limited_list(
            evidence_graph.get("auxiliary_observation_facts", []), limit=6
        ),
    }


def _global_behavior_summary(global_features: dict[str, Any]) -> dict[str, Any]:
    screen_status = global_features.get("screen_orientation_proxy_status")
    gaze_ratio = global_features.get("screen_gaze_ratio")
    if gaze_ratio is None:
        screen_orientation = (
            "Screen-oriented posture ratio unavailable"
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
    }.get(str(posture_trend), "Posture trend unavailable.")

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
    stateful = package.get("stateful_behavior", {})
    compact_mode = bool(stateful)
    text_package = {
        "schema_version": (
            "privte_llm_text_evidence.v3.compact"
            if compact_mode
            else "privte_llm_text_evidence.v3"
        ),
        "sample_id": record["sample_id"],
        "task": {
            "name": task["name"],
            "instruction": task["instruction"],
            "allowed_risk_levels": task["allowed_risk_levels"],
            "boundary": "Risk screening from privacy-filtered video-derived text evidence only.",
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
            "interaction_intensity": global_features.get("interaction_intensity"),
            "repetitive_operation_level": global_features.get(
                "repetitive_operation_level"
            ),
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
        "calibrated_evidence_summary": {
            "event_trigger_counts": global_features.get("event_trigger_counts", {}),
            "confirmed_interaction_window_count": global_features.get(
                "confirmed_interaction_window_count"
            ),
            "possible_interaction_window_count": global_features.get(
                "possible_interaction_window_count"
            ),
            "device_region_motion_window_count": global_features.get(
                "device_region_motion_window_count"
            ),
            "strong_repetitive_proxy_window_count": global_features.get(
                "strong_repetitive_proxy_window_count"
            ),
            "negative_evidence_summary": global_features.get(
                "negative_evidence_summary", []
            ),
            "evidence_balance": global_features.get("evidence_balance"),
        },
        "stateful_behavior_summary": _stateful_narrative(stateful),
        "detector_policy_context": _detector_policy_context(stateful),
        "key_window_states": _important_window_states(stateful, limit=6),
        "key_behavior_episodes": _important_episodes(stateful, limit=6),
        "episode_detectors": _compact_episode_detectors(stateful, limit=8),
        "evidence_graph": stateful.get("evidence_graph", {}),
        "state_space_summary": stateful.get("state_space_summary", {}),
        "event_windows": package.get("event_windows", []),
        "quality_summary": package.get("quality_summary", {}),
        "limitations": package.get("limitations", []),
        "missing_information": package.get("missing_information", []),
        "privacy_processing_summary": package.get("privacy_processing_summary", {}),
        "requested_model_output": package.get("requested_model_output", {}),
    }
    return json.dumps(text_package, ensure_ascii=False, indent=2)
