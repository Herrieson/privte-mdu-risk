"""Simple video-file and container-metadata quality baseline."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ..core import EvidenceExtractor
from .common import (
    collect_clips,
    count_available,
    count_nonempty_json,
    duration_bucket,
    evenly_sample,
    fps_bucket,
    probe_mp4_box_metadata,
    resolution_bucket,
    round_mb,
    safe_ratio,
    size_bucket,
    video_file_entries,
)


class SimpleVideoQualityExtractor(EvidenceExtractor):
    """Privacy-conservative video-file and container-metadata extractor.

    It may open videos to read container metadata, but it does not decode, save,
    or emit frames.
    """

    name = "simple_video_quality"
    version = "v0"
    feature_schema_version = "simple_video_quality_feature_schema.v0"

    def _try_import_cv2(self) -> Any | None:
        if self.config.get("disable_opencv"):
            return None
        try:
            import cv2  # type: ignore[import-not-found]
        except Exception:
            return None
        return cv2

    def _probe_with_cv2(self, path: Path, cv2: Any) -> dict[str, Any]:
        result: dict[str, Any] = {
            "readable": False,
            "duration_sec": None,
            "width": None,
            "height": None,
            "fps": None,
            "backend": "opencv",
        }
        capture = cv2.VideoCapture(path.as_posix())
        try:
            if not capture.isOpened():
                return result

            frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            width = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)
            height = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)
            duration_sec = frame_count / fps if frame_count > 0 and fps > 0 else None
            result.update(
                {
                    "readable": True,
                    "duration_sec": duration_sec,
                    "width": width if width > 0 else None,
                    "height": height if height > 0 else None,
                    "fps": fps if fps > 0 else None,
                }
            )
        finally:
            capture.release()
        return result

    def _probe_video_metadata(
        self,
        file_entries: list[dict[str, Any]],
    ) -> dict[str, Any]:
        cv2 = self._try_import_cv2()
        max_metadata_clips = int(self.config.get("max_metadata_clips", 16))
        sampled_entries = evenly_sample(file_entries, max_metadata_clips)

        path_exists_count = 0
        readable_count = 0
        duration_bins: Counter[str] = Counter()
        resolution_bins: Counter[str] = Counter()
        fps_bins: Counter[str] = Counter()
        backend_counts: Counter[str] = Counter()

        for file_info in sampled_entries:
            path = Path(file_info["path"])
            if path.exists():
                path_exists_count += 1
            if not path.exists():
                duration_bins["unknown"] += 1
                resolution_bins["unknown"] += 1
                fps_bins["unknown"] += 1
                backend_counts["missing_file"] += 1
                continue

            metadata = (
                self._probe_with_cv2(path, cv2)
                if cv2 is not None
                else probe_mp4_box_metadata(path)
            )
            backend_counts[metadata["backend"]] += 1
            if metadata["readable"]:
                readable_count += 1

            duration_bins[duration_bucket(metadata["duration_sec"])] += 1
            resolution_bins[
                resolution_bucket(metadata["width"], metadata["height"])
            ] += 1
            fps_bins[fps_bucket(metadata["fps"])] += 1

        attempted_count = len(sampled_entries)
        return {
            "opencv_available": cv2 is not None,
            "fallback_backend": "mp4_box_parser",
            "max_metadata_clips": max_metadata_clips,
            "attempted_clip_count": attempted_count,
            "path_exists_count": path_exists_count,
            "path_exists_ratio": safe_ratio(path_exists_count, attempted_count),
            "readable_clip_count": readable_count,
            "readable_ratio": safe_ratio(readable_count, attempted_count),
            "backend_counts": dict(sorted(backend_counts.items())),
            "duration_bin_counts": dict(sorted(duration_bins.items())),
            "resolution_bin_counts": dict(sorted(resolution_bins.items())),
            "fps_bin_counts": dict(sorted(fps_bins.items())),
            "privacy_note": "container metadata only; no frame, image, audio, OCR, or ASR output",
        }

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

        file_entries = video_file_entries(clips)
        size_values = [
            int(file_info["size_bytes"])
            for file_info in file_entries
            if isinstance(file_info.get("size_bytes"), int)
        ]
        size_mb_values = [round_mb(size_bytes) for size_bytes in size_values]
        size_bins = Counter(size_bucket(value) for value in size_mb_values)
        total_video_size_mb = round_mb(sum(size_values)) if size_values else None
        average_video_size_mb = (
            round(
                sum(value for value in size_mb_values if value is not None)
                / len(size_mb_values),
                2,
            )
            if size_mb_values
            else None
        )

        metadata_probe = self._probe_video_metadata(file_entries)
        video_presence_ratio = safe_ratio(num_video_clips, num_clips)
        readable_ratio = metadata_probe["readable_ratio"]
        if not file_entries:
            overall_quality = "insufficient_video"
        elif readable_ratio >= 0.9:
            overall_quality = "usable_container_quality"
        elif readable_ratio >= 0.5:
            overall_quality = "partial_container_quality"
        elif metadata_probe["path_exists_ratio"] > 0:
            overall_quality = "file_level_quality_only"
        else:
            overall_quality = "low_container_quality"

        visual_proxy_features = [
            f"视频片段可用率: {video_presence_ratio}",
            f"视频文件数量: {len(file_entries)}",
            "视频文件大小分布: "
            + (
                ", ".join(f"{key}={value}" for key, value in sorted(size_bins.items()))
                or "unknown"
            ),
            f"抽样容器元数据可读率: {readable_ratio}",
            "视频时长分布: "
            + (
                ", ".join(
                    f"{key}={value}"
                    for key, value in sorted(metadata_probe["duration_bin_counts"].items())
                )
                or "unknown"
            ),
            "视频分辨率分布: "
            + (
                ", ".join(
                    f"{key}={value}"
                    for key, value in sorted(metadata_probe["resolution_bin_counts"].items())
                )
                or "unknown"
            ),
            "视频帧率分布: "
            + (
                ", ".join(
                    f"{key}={value}"
                    for key, value in sorted(metadata_probe["fps_bin_counts"].items())
                )
                or "unknown"
            ),
        ]

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
                    "status": "computed_simple_video_quality",
                    "num_sessions": num_sessions,
                    "num_clips": num_clips,
                    "num_video_clips": num_video_clips,
                    "num_video_files": len(file_entries),
                    "video_presence_ratio": video_presence_ratio,
                    "total_video_size_mb_rounded": total_video_size_mb,
                    "average_video_size_mb_rounded": average_video_size_mb,
                    "video_size_bin_counts": dict(sorted(size_bins.items())),
                    "metadata_probe": metadata_probe,
                    "visual_proxy_features": visual_proxy_features,
                    "replaceable_by": "future PriVTE visual proxy feature extractor",
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
                    "status": "computed_by_simple_video_quality",
                    "overall": overall_quality,
                    "computed_quality_fields": [
                        "video_presence_ratio",
                        "clip_size_distribution",
                        "container_readability_ratio",
                        "duration_distribution",
                        "resolution_distribution",
                        "fps_distribution",
                    ],
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
                "face_visible_ratio",
                "hands_visible_ratio",
                "device_visible_ratio",
                "screen_attention",
                "touch_or_swipe_frequency",
                "posture_features",
                "blink_or_facial_action_features",
            ],
            "limitations": [
                "simple_video_quality_only",
                "container_metadata_not_behavior",
                "no_questionnaire_input",
                "no_exact_heart_rate_input",
                "no_app_name_input",
                "not_for_diagnosis",
                "requires_future_privte_behavior_feature_extraction",
            ],
        }
