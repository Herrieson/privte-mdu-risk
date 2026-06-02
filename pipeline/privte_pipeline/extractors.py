"""Replaceable evidence extractors for the PriVTE evidence pipeline.

Future video algorithms should implement the same extractor contract:

    extractor.extract(person_record) -> evidence dict

The writer, text renderer, label handling, and report generation should not need
to change when the extraction algorithm changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any

CONTAINER_BOX_TYPES = {
    "moov",
    "trak",
    "mdia",
    "minf",
    "stbl",
    "edts",
    "dinf",
    "udta",
}


def collect_clips(person_record: dict[str, Any]) -> list[dict[str, Any]]:
    clips: list[dict[str, Any]] = []
    for session in person_record.get("sessions", []):
        clips.extend(session.get("clips", []))
    return clips


def count_available(clips: list[dict[str, Any]], modality: str) -> int:
    return sum(1 for clip in clips if clip.get(modality, {}).get("available"))


def count_nonempty_json(clips: list[dict[str, Any]], modality: str) -> int:
    return sum(
        1
        for clip in clips
        if clip.get(modality, {}).get("available")
        and not clip.get(modality, {}).get("is_empty", True)
    )


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return round(float(numerator) / float(denominator), 3)


def round_mb(size_bytes: int | float | None) -> float | None:
    if size_bytes is None:
        return None
    return round(float(size_bytes) / (1024 * 1024), 2)


def video_file_entries(clips: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for clip in clips:
        video = clip.get("video", {})
        if not video.get("available"):
            continue
        for file_info in video.get("files", []):
            if file_info.get("path"):
                entries.append(file_info)
    return entries


def evenly_sample(items: list[dict[str, Any]], max_count: int) -> list[dict[str, Any]]:
    if max_count <= 0 or len(items) <= max_count:
        return items
    if max_count == 1:
        return items[:1]
    step = (len(items) - 1) / (max_count - 1)
    return [items[round(index * step)] for index in range(max_count)]


def size_bucket(size_mb: float | None) -> str:
    if size_mb is None:
        return "unknown"
    if size_mb < 5:
        return "<5MB"
    if size_mb < 20:
        return "5-20MB"
    if size_mb < 50:
        return "20-50MB"
    return ">=50MB"


def duration_bucket(duration_sec: float | None) -> str:
    if duration_sec is None:
        return "unknown"
    if duration_sec < 10:
        return "<10s"
    if duration_sec < 30:
        return "10-30s"
    if duration_sec < 90:
        return "30-90s"
    return ">=90s"


def resolution_bucket(width: float | int | None, height: float | int | None) -> str:
    if not width or not height:
        return "unknown"
    pixels = int(width) * int(height)
    if pixels < 640 * 480:
        return "below_vga"
    if pixels < 1280 * 720:
        return "vga_to_720p"
    if pixels < 1920 * 1080:
        return "720p_to_1080p"
    return "1080p_or_above"


def fps_bucket(fps: float | None) -> str:
    if not fps or fps <= 0:
        return "unknown"
    if fps < 15:
        return "<15fps"
    if fps < 24:
        return "15-24fps"
    if fps <= 30:
        return "24-30fps"
    return ">30fps"


def parse_mvhd_duration(payload: bytes) -> float | None:
    if len(payload) < 20:
        return None
    version = payload[0]
    if version == 1:
        if len(payload) < 32:
            return None
        timescale = int.from_bytes(payload[20:24], "big")
        duration = int.from_bytes(payload[24:32], "big")
    else:
        timescale = int.from_bytes(payload[12:16], "big")
        duration = int.from_bytes(payload[16:20], "big")
    if timescale <= 0 or duration <= 0:
        return None
    return duration / timescale


def parse_tkhd_dimensions(payload: bytes) -> tuple[float | None, float | None]:
    if len(payload) < 8:
        return None, None
    width_fixed = int.from_bytes(payload[-8:-4], "big")
    height_fixed = int.from_bytes(payload[-4:], "big")
    width = width_fixed / 65536
    height = height_fixed / 65536
    if width <= 0 or height <= 0:
        return None, None
    return width, height


def probe_mp4_box_metadata(path: Path) -> dict[str, Any]:
    """Read coarse MP4 container metadata without decoding frames."""

    result: dict[str, Any] = {
        "readable": False,
        "duration_sec": None,
        "width": None,
        "height": None,
        "fps": None,
        "backend": "mp4_box_parser",
    }

    try:
        file_size = path.stat().st_size
        with path.open("rb") as file:

            def read_box_header(end: int) -> tuple[int, str, int, int] | None:
                start = file.tell()
                if start + 8 > end:
                    return None
                header = file.read(8)
                if len(header) < 8:
                    return None
                size = int.from_bytes(header[:4], "big")
                box_type = header[4:8].decode("latin-1", errors="replace")
                header_size = 8
                if size == 1:
                    large_size_raw = file.read(8)
                    if len(large_size_raw) < 8:
                        return None
                    size = int.from_bytes(large_size_raw, "big")
                    header_size = 16
                elif size == 0:
                    size = end - start
                if size < header_size:
                    return None
                box_end = min(start + size, end)
                return start, box_type, header_size, box_end

            def parse_range(end: int, depth: int = 0) -> None:
                while file.tell() + 8 <= end:
                    header = read_box_header(end)
                    if header is None:
                        return
                    start, box_type, header_size, box_end = header
                    payload_size = max(0, box_end - start - header_size)

                    if box_type == "mvhd":
                        duration = parse_mvhd_duration(file.read(payload_size))
                        if duration is not None:
                            result["duration_sec"] = duration
                            result["readable"] = True
                    elif box_type == "tkhd":
                        width, height = parse_tkhd_dimensions(file.read(payload_size))
                        current_pixels = (result.get("width") or 0) * (
                            result.get("height") or 0
                        )
                        new_pixels = (width or 0) * (height or 0)
                        if width is not None and height is not None and new_pixels > current_pixels:
                            result["width"] = width
                            result["height"] = height
                            result["readable"] = True
                    elif box_type in CONTAINER_BOX_TYPES and depth < 8:
                        parse_range(box_end, depth + 1)

                    file.seek(box_end)

            parse_range(file_size)
    except OSError:
        return result

    return result


class EvidenceExtractor(ABC):
    """Base interface for PriVTE evidence extraction algorithms."""

    name = "base"
    version = "v0"
    feature_schema_version = "feature_schema.v0"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "feature_schema_version": self.feature_schema_version,
            "config": self.config,
        }

    @abstractmethod
    def extract(self, person_record: dict[str, Any]) -> dict[str, Any]:
        """Return privacy-filtered evidence for one person record."""


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
        num_video_clips = completeness.get("num_video_clips", count_available(clips, "video"))
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
                    "reason": "questionnaire is label/context source, not text-only video proxy input",
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


class SimpleVideoQualityExtractor(EvidenceExtractor):
    """Simple video-file and container-metadata quality extractor.

    This extractor is intentionally privacy-conservative. It may open videos to
    read container metadata, but it does not decode, save, or emit frames.
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
        num_video_clips = completeness.get("num_video_clips", count_available(clips, "video"))
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
            round(sum(value for value in size_mb_values if value is not None) / len(size_mb_values), 2)
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
            + (", ".join(f"{key}={value}" for key, value in sorted(size_bins.items())) or "unknown"),
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
                    f"{key}={value}" for key, value in sorted(metadata_probe["fps_bin_counts"].items())
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
                    "reason": "questionnaire is label/context source, not text-only video proxy input",
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


EXTRACTOR_REGISTRY: dict[str, type[EvidenceExtractor]] = {
    ManifestOnlyExtractor.name: ManifestOnlyExtractor,
    SimpleVideoQualityExtractor.name: SimpleVideoQualityExtractor,
}


def available_extractors() -> list[str]:
    return sorted(EXTRACTOR_REGISTRY)


def build_extractor(name: str, config: dict[str, Any] | None = None) -> EvidenceExtractor:
    try:
        extractor_cls = EXTRACTOR_REGISTRY[name]
    except KeyError as exc:
        options = ", ".join(available_extractors())
        raise ValueError(f"Unknown extractor {name!r}. Available: {options}") from exc
    return extractor_cls(config)
