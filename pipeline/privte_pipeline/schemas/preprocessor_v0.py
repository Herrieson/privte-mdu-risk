"""PriVTE preprocessor evidence schema v0."""

from __future__ import annotations

from typing import Any


PREPROCESSOR_SCHEMA_VERSION = "privte_preprocessor_evidence.v0"

TOP_LEVEL_FIELDS = {
    "schema_version",
    "session_metadata",
    "global_features",
    "event_windows",
    "quality_summary",
    "limitations",
    "privacy_processing_summary",
}

SESSION_METADATA_FIELDS = {
    "session_id",
    "total_valid_duration_minutes",
    "duration_bin",
    "valid_observation_duration_bin",
    "analyzed_window_count",
    "event_window_count",
    "privacy_processing_summary",
}

GLOBAL_FEATURE_FIELDS = {
    "screen_gaze_ratio",
    "screen_gaze_ratio_bin",
    "max_continuous_gaze_duration_minutes",
    "max_continuous_gaze_duration_bin",
    "average_blink_rate_per_minute",
    "blink_rate_level",
    "blink_rate_trend",
    "overall_posture_trend",
    "interaction_intensity",
    "repetitive_operation_level",
    "motion_confounding_level",
}

QUALITY_SUMMARY_FIELDS = {
    "overall_data_sufficiency",
    "face_observability",
    "hand_observability",
    "device_observability",
    "gaze_estimation_quality",
    "multi_person_interference",
    "motion_confounding_level",
    "limitations",
}

PRIVACY_FLAG_FIELDS = {
    "raw_video_included",
    "raw_images_included",
    "raw_audio_included",
    "ocr_text_included",
    "asr_text_included",
    "exact_timestamps_included",
    "high_frequency_coordinates_included",
    "face_embeddings_included",
    "screen_content_included",
    "questionnaire_answers_included",
    "exact_heart_rate_values_included",
    "app_names_included",
    "raw_paths_included",
    "processing_steps",
}

DEFAULT_GLOBAL_FEATURES = {
    "screen_gaze_ratio": None,
    "screen_gaze_ratio_bin": "not_computed",
    "max_continuous_gaze_duration_minutes": None,
    "max_continuous_gaze_duration_bin": "not_computed",
    "average_blink_rate_per_minute": None,
    "blink_rate_level": "not_computed",
    "blink_rate_trend": "not_computed",
    "overall_posture_trend": "not_computed",
    "interaction_intensity": "not_computed",
    "repetitive_operation_level": "not_computed",
    "motion_confounding_level": "not_computed",
}

DEFAULT_QUALITY_SUMMARY = {
    "overall_data_sufficiency": "insufficient_visual_processing_not_run",
    "face_observability": "not_computed",
    "hand_observability": "not_computed",
    "device_observability": "not_computed",
    "gaze_estimation_quality": "not_computed",
    "multi_person_interference": "not_computed",
    "motion_confounding_level": "not_computed",
    "limitations": [],
}

DEFAULT_PRIVACY_PROCESSING_SUMMARY = {
    "raw_video_included": False,
    "raw_images_included": False,
    "raw_audio_included": False,
    "ocr_text_included": False,
    "asr_text_included": False,
    "exact_timestamps_included": False,
    "high_frequency_coordinates_included": False,
    "face_embeddings_included": False,
    "screen_content_included": False,
    "questionnaire_answers_included": False,
    "exact_heart_rate_values_included": False,
    "app_names_included": False,
    "raw_paths_included": False,
    "processing_steps": [
        "raw video remains local to the preprocessing environment",
        "public evidence uses schema-first low-dimensional fields only",
        "raw paths, exact timestamps, OCR, ASR, app names, and raw media are suppressed",
    ],
}


def build_preprocessor_evidence(
    *,
    session_id: str,
    total_valid_duration_minutes: int | None = None,
    duration_bin: str = "unknown",
    valid_observation_duration_bin: str = "unknown",
    analyzed_window_count: int = 0,
    event_window_count: int = 0,
    global_features: dict[str, Any] | None = None,
    event_windows: list[dict[str, Any]] | None = None,
    quality_summary: dict[str, Any] | None = None,
    limitations: list[str] | None = None,
    privacy_processing_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a schema-constrained preprocessor evidence object."""

    privacy_steps = list(
        (privacy_processing_summary or {}).get(
            "processing_steps",
            DEFAULT_PRIVACY_PROCESSING_SUMMARY["processing_steps"],
        )
    )
    evidence = {
        "schema_version": PREPROCESSOR_SCHEMA_VERSION,
        "session_metadata": {
            "session_id": session_id,
            "total_valid_duration_minutes": total_valid_duration_minutes,
            "duration_bin": duration_bin,
            "valid_observation_duration_bin": valid_observation_duration_bin,
            "analyzed_window_count": analyzed_window_count,
            "event_window_count": event_window_count,
            "privacy_processing_summary": privacy_steps,
        },
        "global_features": {
            **DEFAULT_GLOBAL_FEATURES,
            **(global_features or {}),
        },
        "event_windows": event_windows or [],
        "quality_summary": {
            **DEFAULT_QUALITY_SUMMARY,
            **(quality_summary or {}),
        },
        "limitations": limitations or [],
        "privacy_processing_summary": {
            **DEFAULT_PRIVACY_PROCESSING_SUMMARY,
            **(privacy_processing_summary or {}),
        },
    }
    errors = validate_preprocessor_evidence(evidence)
    if errors:
        raise ValueError("; ".join(errors))
    return evidence


def _missing_fields(values: dict[str, Any], required: set[str], prefix: str) -> list[str]:
    return [f"missing {prefix}.{field}" for field in sorted(required - values.keys())]


def validate_preprocessor_evidence(evidence: dict[str, Any]) -> list[str]:
    """Return validation errors for a preprocessor evidence object."""

    errors: list[str] = []
    errors.extend(_missing_fields(evidence, TOP_LEVEL_FIELDS, "evidence"))
    extra = set(evidence) - TOP_LEVEL_FIELDS
    errors.extend(f"unexpected evidence.{field}" for field in sorted(extra))

    if evidence.get("schema_version") != PREPROCESSOR_SCHEMA_VERSION:
        errors.append("invalid evidence.schema_version")

    session_metadata = evidence.get("session_metadata", {})
    if isinstance(session_metadata, dict):
        errors.extend(
            _missing_fields(
                session_metadata,
                SESSION_METADATA_FIELDS,
                "session_metadata",
            )
        )
    else:
        errors.append("session_metadata must be an object")

    global_features = evidence.get("global_features", {})
    if isinstance(global_features, dict):
        errors.extend(
            _missing_fields(global_features, GLOBAL_FEATURE_FIELDS, "global_features")
        )
    else:
        errors.append("global_features must be an object")

    event_windows = evidence.get("event_windows", [])
    if not isinstance(event_windows, list):
        errors.append("event_windows must be a list")

    quality_summary = evidence.get("quality_summary", {})
    if isinstance(quality_summary, dict):
        errors.extend(
            _missing_fields(quality_summary, QUALITY_SUMMARY_FIELDS, "quality_summary")
        )
    else:
        errors.append("quality_summary must be an object")

    if not isinstance(evidence.get("limitations", []), list):
        errors.append("limitations must be a list")

    privacy = evidence.get("privacy_processing_summary", {})
    if isinstance(privacy, dict):
        errors.extend(
            _missing_fields(privacy, PRIVACY_FLAG_FIELDS, "privacy_processing_summary")
        )
    else:
        errors.append("privacy_processing_summary must be an object")

    return errors
