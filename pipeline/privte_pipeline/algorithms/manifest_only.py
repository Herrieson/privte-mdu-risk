"""Schema/data-flow baseline algorithm."""

from __future__ import annotations

from typing import Any

from ..core import EvidenceExtractor
from .common import collect_clips, count_available, count_nonempty_json


class ManifestOnlyExtractor(EvidenceExtractor):
    """Schema/data-flow baseline without visual feature extraction."""

    name = "manifest_only"
    version = "v0"
    feature_schema_version = "manifest_only_feature_schema.v0"

    def extract(self, person_record: dict[str, Any]) -> dict[str, Any]:
        clips = collect_clips(person_record)
        completeness = person_record.get("modality_completeness", {})
        questionnaire = person_record.get("questionnaire", {})

        num_sessions = len(person_record.get("sessions", []))
        num_clips = len(clips)
        num_video_clips = completeness.get(
            "num_video_clips",
            count_available(clips, "video"),
        )
        num_nonempty_heart_rate = completeness.get(
            "num_nonempty_heart_rate",
            count_nonempty_json(clips, "heart_rate"),
        )
        num_usage_records = completeness.get("num_usage_records", 0)
        has_questionnaire = bool(questionnaire.get("available")) and bool(
            questionnaire.get("field_count")
        )

        return {
            "extractor": self.metadata,
            "modality_availability": {
                "has_video": num_video_clips > 0,
                "has_heart_rate": num_nonempty_heart_rate > 0,
                "has_app_usage": num_usage_records > 0,
                "has_questionnaire": has_questionnaire,
            },
            "feature_blocks": {
                "video_proxy_summary": {
                    "status": "not_extracted_in_pipeline_baseline",
                    "num_sessions": num_sessions,
                    "num_clips": num_clips,
                    "num_video_clips": num_video_clips,
                    "visual_proxy_features": [],
                    "replaceable_by": "future PriVTE video feature extractor",
                },
                "heart_rate_summary": {
                    "status": "availability_only",
                    "num_nonempty_heart_rate_clips": num_nonempty_heart_rate,
                    "exact_values_included": False,
                    "public_policy": "trend_or_quality_only_in_future_versions",
                },
                "app_usage_summary": {
                    "status": "availability_only",
                    "num_usage_records": num_usage_records,
                    "app_names_included": False,
                    "public_policy": "coarse_category_and_duration_bin_only_in_future_versions",
                },
                "questionnaire_status": {
                    "available": has_questionnaire,
                    "used_as_input": False,
                    "reason": (
                        "questionnaire is label/context source, not text-only video proxy input"
                    ),
                },
                "quality_summary": {
                    "status": "not_computed_in_pipeline_baseline",
                    "overall": "unknown",
                    "computed_quality_fields": [],
                    "missing_quality_fields": [
                        "face_visible_ratio",
                        "hands_visible_ratio",
                        "device_visible_ratio",
                        "lighting",
                        "occlusion",
                    ],
                },
            },
            "privacy_processing_summary": {
                "raw_video_included": False,
                "raw_images_included": False,
                "raw_audio_included": False,
                "questionnaire_answers_included": False,
                "exact_heart_rate_values_included": False,
                "app_names_included": False,
            },
            "missing_information": [
                "visual_proxy_features",
                "visual_quality_metrics",
                "screen_attention",
                "touch_or_swipe_frequency",
                "posture_features",
                "blink_or_facial_action_features",
            ],
            "limitations": [
                "manifest_only_pipeline_baseline",
                "no_visual_proxy_features_yet",
                "no_questionnaire_input",
                "no_exact_heart_rate_input",
                "no_app_name_input",
                "not_for_diagnosis",
                "requires_future_privte_video_feature_extraction",
            ],
        }
