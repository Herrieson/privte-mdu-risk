"""PriVTE-FlowLite v0 frame-level evidence extractor.

This module implements a lightweight, local-only instantiation of the PriVTE
protocol. It decodes sampled frames when OpenCV is available, extracts coarse
proxy features, aggregates them, and emits privacy-filtered evidence. It never
stores or emits raw frames, frame crops, coordinates, OCR text, ASR text, face
embeddings, or high-dimensional landmark sequences.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import median
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
    safe_ratio,
    video_file_entries,
)


def ratio_bin(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value <= 0:
        return "none"
    if value < 0.25:
        return "low"
    if value < 0.6:
        return "medium"
    if value < 0.85:
        return "high"
    return "very_high"


def count_bin(value: int) -> str:
    if value <= 0:
        return "none"
    if value <= 2:
        return "low"
    if value <= 6:
        return "medium"
    return "high"


def level_bin(value: float | None, *, low: float, medium: float, high: float) -> str:
    if value is None:
        return "unknown"
    if value < low:
        return "low"
    if value < medium:
        return "medium"
    if value < high:
        return "high"
    return "elevated"


def relative_position(index: int, total: int) -> str:
    if total <= 1:
        return "single"
    fraction = index / max(total - 1, 1)
    if fraction < 1 / 3:
        return "early"
    if fraction < 2 / 3:
        return "middle"
    return "late"


def mean_value(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def median_value(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(median(values)), 3)


def sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


class PriVTEFlowLiteExtractor(EvidenceExtractor):
    """Frame-level PriVTE extractor using OpenCV and coarse privacy filtering."""

    name = "privte_flowlite"
    version = "v0"
    feature_schema_version = "privte_flowlite_feature_schema.v0"

    def _try_import_cv2(self) -> Any | None:
        try:
            import cv2  # type: ignore[import-not-found]
        except Exception:
            return None
        return cv2

    def _base_context(self, person_record: dict[str, Any]) -> dict[str, Any]:
        clips = collect_clips(person_record)
        completeness = person_record.get("modality_completeness", {})
        questionnaire = person_record.get("questionnaire", {})
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
            "clips": clips,
            "num_sessions": len(person_record.get("sessions", [])),
            "num_clips": len(clips),
            "num_video_clips": num_video_clips,
            "num_nonempty_heart_rate": num_nonempty_heart_rate,
            "num_usage_records": num_usage_records,
            "has_questionnaire": has_questionnaire,
        }

    def extract(self, person_record: dict[str, Any]) -> dict[str, Any]:
        cv2 = self._try_import_cv2()
        require_opencv = bool(self.config.get("require_opencv", False))
        if cv2 is None and require_opencv:
            raise RuntimeError(
                "PriVTE-FlowLite requires opencv-python-headless for frame-level "
                "analysis. Install project dependencies with uv before running."
            )
        if cv2 is None:
            return self._extract_metadata_fallback(person_record)
        return self._extract_with_cv2(person_record, cv2)

    def _extract_with_cv2(self, person_record: dict[str, Any], cv2: Any) -> dict[str, Any]:
        context = self._base_context(person_record)
        file_entries = video_file_entries(context["clips"])
        max_video_clips = int(self.config.get("max_video_clips", 12))
        selected_entries = evenly_sample(file_entries, max_video_clips)
        face_detector = self._build_face_detector(cv2)

        frame_records: list[dict[str, Any]] = []
        readable_video_files = 0
        opened_video_files = 0
        for clip_order, file_info in enumerate(selected_entries):
            path = Path(file_info["path"])
            if not path.exists():
                continue
            video_result = self._analyze_video_file(
                path=path,
                cv2=cv2,
                face_detector=face_detector,
                clip_order=clip_order,
            )
            if video_result["opened"]:
                opened_video_files += 1
            if video_result["readable"]:
                readable_video_files += 1
            frame_records.extend(video_result["frames"])

        for frame_index, frame_record in enumerate(frame_records):
            frame_record["relative_position"] = relative_position(
                frame_index,
                len(frame_records),
            )

        aggregate = self._aggregate_frames(
            frame_records=frame_records,
            readable_video_files=readable_video_files,
            opened_video_files=opened_video_files,
            selected_video_files=len(selected_entries),
            total_video_files=len(file_entries),
            face_detector_available=face_detector is not None,
        )
        visual_proxy_features = self._build_visual_proxy_lines(aggregate)

        return self._build_evidence(
            context=context,
            file_entries=file_entries,
            status="computed_by_privte_flowlite",
            video_summary={
                "analysis_backend": "opencv",
                "num_video_files": len(file_entries),
                "selected_video_files": len(selected_entries),
                "opened_video_files": opened_video_files,
                "readable_video_files": readable_video_files,
                "sampled_frame_count": len(frame_records),
                "frame_sampling": {
                    "max_video_clips": max_video_clips,
                    "frames_per_clip": int(self.config.get("frames_per_clip", 8)),
                    "analysis_width": int(self.config.get("analysis_width", 640)),
                },
                "roi_summary": aggregate["roi_summary"],
                "global_features": aggregate["global_features"],
                "reference_normalization": aggregate["reference_normalization"],
                "key_window_summary": aggregate["key_window_summary"],
                "event_windows": aggregate["event_windows"],
                "visual_proxy_features": visual_proxy_features,
                "replaceable_by": "future stronger PriVTE visual proxy extractor",
            },
            quality_summary=aggregate["quality_summary"],
            missing_information=[
                "direct_hand_landmarks",
                "direct_gaze_estimation",
                "screen_content_ocr",
                "high_dimensional_pose_or_face_mesh",
                "questionnaire_input",
                "exact_heart_rate_input",
                "app_name_input",
            ],
            limitations=[
                "privte_flowlite_v0",
                "opencv_heuristic_roi_only",
                "screen_like_region_is_heuristic_not_ocr",
                "near_device_motion_is_proxy_not_touch_detection",
                "no_questionnaire_input",
                "no_exact_heart_rate_input",
                "no_app_name_input",
                "not_for_diagnosis",
            ],
        )

    def _build_face_detector(self, cv2: Any) -> Any | None:
        haar_root = getattr(getattr(cv2, "data", None), "haarcascades", "")
        cascade_path = Path(haar_root) / "haarcascade_frontalface_default.xml"
        if not cascade_path.exists():
            return None
        detector = cv2.CascadeClassifier(cascade_path.as_posix())
        if detector.empty():
            return None
        return detector

    def _frame_indices(self, frame_count: int, frames_per_clip: int) -> list[int]:
        if frames_per_clip <= 0:
            return []
        if frame_count <= 0:
            return list(range(frames_per_clip))
        if frame_count <= frames_per_clip:
            return list(range(frame_count))
        step = (frame_count - 1) / max(frames_per_clip - 1, 1)
        return sorted({round(index * step) for index in range(frames_per_clip)})

    def _analyze_video_file(
        self,
        *,
        path: Path,
        cv2: Any,
        face_detector: Any | None,
        clip_order: int,
    ) -> dict[str, Any]:
        frames_per_clip = int(self.config.get("frames_per_clip", 8))
        capture = cv2.VideoCapture(path.as_posix())
        result = {"opened": False, "readable": False, "frames": []}
        try:
            if not capture.isOpened():
                return result
            result["opened"] = True
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            frame_indices = self._frame_indices(frame_count, frames_per_clip)
            previous_gray = None

            for frame_order, frame_index in enumerate(frame_indices):
                if frame_count > 0:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame = capture.read()
                if not ok:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = self._resize_gray(gray, cv2)
                record = self._analyze_frame(
                    gray=gray,
                    previous_gray=previous_gray,
                    cv2=cv2,
                    face_detector=face_detector,
                    clip_order=clip_order,
                    frame_order=frame_order,
                )
                result["frames"].append(record)
                previous_gray = gray
            result["readable"] = bool(result["frames"])
        finally:
            capture.release()
        return result

    def _resize_gray(self, gray: Any, cv2: Any) -> Any:
        analysis_width = int(self.config.get("analysis_width", 640))
        height, width = gray.shape[:2]
        if width <= analysis_width:
            return gray
        scale = analysis_width / width
        resized_height = max(1, int(height * scale))
        return cv2.resize(gray, (analysis_width, resized_height))

    def _analyze_frame(
        self,
        *,
        gray: Any,
        previous_gray: Any | None,
        cv2: Any,
        face_detector: Any | None,
        clip_order: int,
        frame_order: int,
    ) -> dict[str, Any]:
        height, width = gray.shape[:2]
        brightness = float(gray.mean())
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        face_visible, face_area_ratio = self._detect_face(
            gray=gray,
            cv2=cv2,
            face_detector=face_detector,
        )
        screen_visible, screen_area_ratio, screen_box = self._detect_screen_like_region(
            gray=gray,
            cv2=cv2,
        )

        global_motion = None
        near_device_motion = None
        if previous_gray is not None and previous_gray.shape == gray.shape:
            diff = cv2.absdiff(gray, previous_gray)
            global_motion = float(diff.mean())
            if screen_box is not None:
                x, y, box_width, box_height = screen_box
                roi = diff[y : y + box_height, x : x + box_width]
            else:
                roi = diff[int(height * 0.45) :, :]
            near_device_motion = float(roi.mean()) if roi.size else None

        min_brightness = float(self.config.get("min_brightness", 35.0))
        max_brightness = float(self.config.get("max_brightness", 225.0))
        min_blur_score = float(self.config.get("min_blur_score", 15.0))
        quality_usable = (
            min_brightness <= brightness <= max_brightness
            and blur_score >= min_blur_score
        )

        return {
            "clip_order": clip_order,
            "frame_order": frame_order,
            "brightness": round(brightness, 3),
            "blur_score": round(blur_score, 3),
            "quality_usable": quality_usable,
            "face_visible": face_visible,
            "face_area_ratio": round(face_area_ratio, 4),
            "screen_like_region_visible": screen_visible,
            "screen_like_area_ratio": round(screen_area_ratio, 4),
            "global_motion": round(global_motion, 3) if global_motion is not None else None,
            "near_device_motion": (
                round(near_device_motion, 3)
                if near_device_motion is not None
                else None
            ),
        }

    def _detect_face(
        self,
        *,
        gray: Any,
        cv2: Any,
        face_detector: Any | None,
    ) -> tuple[bool, float]:
        if face_detector is None:
            return False, 0.0
        height, width = gray.shape[:2]
        equalized = cv2.equalizeHist(gray)
        faces = face_detector.detectMultiScale(
            equalized,
            scaleFactor=float(self.config.get("face_scale_factor", 1.1)),
            minNeighbors=int(self.config.get("face_min_neighbors", 5)),
            minSize=(30, 30),
        )
        if len(faces) == 0:
            return False, 0.0
        max_area = max(int(face[2]) * int(face[3]) for face in faces)
        return True, max_area / max(width * height, 1)

    def _detect_screen_like_region(self, *, gray: Any, cv2: Any) -> tuple[bool, float, Any | None]:
        height, width = gray.shape[:2]
        threshold = int(self.config.get("screen_brightness_threshold", 145))
        min_area_ratio = float(self.config.get("screen_min_area_ratio", 0.015))
        max_area_ratio = float(self.config.get("screen_max_area_ratio", 0.75))
        min_aspect = float(self.config.get("screen_min_aspect", 0.3))
        max_aspect = float(self.config.get("screen_max_aspect", 3.5))

        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        contours_result = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_result[0] if len(contours_result) == 2 else contours_result[1]

        best_area_ratio = 0.0
        best_box = None
        frame_area = max(width * height, 1)
        for contour in contours:
            x, y, box_width, box_height = cv2.boundingRect(contour)
            if box_width <= 0 or box_height <= 0:
                continue
            area_ratio = (box_width * box_height) / frame_area
            aspect = box_width / box_height
            if not (min_area_ratio <= area_ratio <= max_area_ratio):
                continue
            if not (min_aspect <= aspect <= max_aspect):
                continue
            if area_ratio > best_area_ratio:
                best_area_ratio = area_ratio
                best_box = (x, y, box_width, box_height)

        return best_box is not None, best_area_ratio, best_box

    def _aggregate_frames(
        self,
        *,
        frame_records: list[dict[str, Any]],
        readable_video_files: int,
        opened_video_files: int,
        selected_video_files: int,
        total_video_files: int,
        face_detector_available: bool,
    ) -> dict[str, Any]:
        total_frames = len(frame_records)
        valid_frames = sum(1 for frame in frame_records if frame["quality_usable"])
        face_visible_frames = sum(1 for frame in frame_records if frame["face_visible"])
        screen_visible_frames = sum(
            1 for frame in frame_records if frame["screen_like_region_visible"]
        )
        co_visible_frames = sum(
            1
            for frame in frame_records
            if frame["face_visible"] and frame["screen_like_region_visible"]
        )
        global_motion_values = [
            frame["global_motion"]
            for frame in frame_records
            if frame["global_motion"] is not None
        ]
        near_motion_values = [
            frame["near_device_motion"]
            for frame in frame_records
            if frame["near_device_motion"] is not None
        ]
        brightness_values = [frame["brightness"] for frame in frame_records]
        blur_values = [frame["blur_score"] for frame in frame_records]
        global_motion_baseline = median_value(global_motion_values)
        near_motion_baseline = median_value(near_motion_values)

        motion_spike_threshold = self._motion_threshold(global_motion_baseline)
        near_motion_spike_threshold = self._motion_threshold(near_motion_baseline)
        motion_spike_count = sum(
            1
            for value in global_motion_values
            if value >= motion_spike_threshold
        )
        near_motion_spike_count = sum(
            1
            for value in near_motion_values
            if value >= near_motion_spike_threshold
        )
        stable_frames = sum(
            1
            for frame in frame_records
            if frame["screen_like_region_visible"]
            and (
                frame["global_motion"] is None
                or frame["global_motion"] <= motion_spike_threshold
            )
        )

        valid_frame_ratio = safe_ratio(valid_frames, total_frames)
        face_visible_ratio = safe_ratio(face_visible_frames, total_frames)
        screen_visible_ratio = safe_ratio(screen_visible_frames, total_frames)
        co_visible_ratio = safe_ratio(co_visible_frames, total_frames)
        stable_viewing_ratio = safe_ratio(stable_frames, total_frames)
        readable_video_ratio = safe_ratio(readable_video_files, selected_video_files)

        global_features = {
            "valid_frame_ratio": valid_frame_ratio,
            "valid_frame_ratio_bin": ratio_bin(valid_frame_ratio),
            "face_visible_ratio": face_visible_ratio,
            "face_visible_ratio_bin": ratio_bin(face_visible_ratio),
            "screen_like_region_visible_ratio": screen_visible_ratio,
            "screen_like_region_visible_ratio_bin": ratio_bin(screen_visible_ratio),
            "face_screen_cooccurrence_ratio": co_visible_ratio,
            "face_screen_cooccurrence_ratio_bin": ratio_bin(co_visible_ratio),
            "stable_viewing_proxy_ratio": stable_viewing_ratio,
            "stable_viewing_proxy_ratio_bin": ratio_bin(stable_viewing_ratio),
            "global_motion_level": level_bin(
                mean_value(global_motion_values),
                low=4.0,
                medium=12.0,
                high=24.0,
            ),
            "near_device_motion_level": level_bin(
                mean_value(near_motion_values),
                low=4.0,
                medium=12.0,
                high=24.0,
            ),
            "interaction_burst_count": near_motion_spike_count,
            "interaction_burst_count_bin": count_bin(near_motion_spike_count),
            "motion_burst_count": motion_spike_count,
            "motion_burst_count_bin": count_bin(motion_spike_count),
            "brightness_level": level_bin(
                mean_value(brightness_values),
                low=45.0,
                medium=110.0,
                high=190.0,
            ),
            "blur_quality_level": level_bin(
                mean_value(blur_values),
                low=15.0,
                medium=60.0,
                high=160.0,
            ),
        }

        event_windows = self._build_event_windows(
            frame_records=frame_records,
            motion_spike_threshold=motion_spike_threshold,
            near_motion_spike_threshold=near_motion_spike_threshold,
            stable_viewing_ratio=stable_viewing_ratio,
        )
        relative_counts = Counter(
            frame["relative_position"] for frame in frame_records if frame.get("relative_position")
        )
        overall_quality = self._overall_quality(
            total_frames=total_frames,
            valid_frame_ratio=valid_frame_ratio,
            readable_video_ratio=readable_video_ratio,
        )

        return {
            "roi_summary": {
                "face_detector_available": face_detector_available,
                "screen_roi_method": "bright_rectangle_heuristic",
                "motion_roi_method": "frame_difference_near_screen_or_lower_frame",
                "raw_coordinates_included": False,
                "frame_images_included": False,
            },
            "global_features": global_features,
            "reference_normalization": {
                "method": "within_person_median_motion_reference",
                "global_motion_baseline_level": level_bin(
                    global_motion_baseline,
                    low=4.0,
                    medium=12.0,
                    high=24.0,
                ),
                "near_device_motion_baseline_level": level_bin(
                    near_motion_baseline,
                    low=4.0,
                    medium=12.0,
                    high=24.0,
                ),
                "exact_motion_values_in_public_text": False,
            },
            "key_window_summary": {
                "coverage_relative_position_counts": sorted_counter(relative_counts),
                "event_window_count": len(event_windows),
                "event_window_types": sorted_counter(
                    Counter(event["event_type"] for event in event_windows)
                ),
                "quality_window_basis": "valid_frame_ratio_and_readable_video_ratio",
            },
            "event_windows": event_windows,
            "quality_summary": {
                "status": "computed_by_privte_flowlite",
                "overall": overall_quality,
                "computed_quality_fields": [
                    "readable_video_ratio",
                    "valid_frame_ratio",
                    "brightness_level",
                    "blur_quality_level",
                    "face_visible_ratio",
                    "screen_like_region_visible_ratio",
                    "motion_burst_count",
                    "near_device_motion_level",
                    "stable_viewing_proxy_ratio",
                ],
                "missing_quality_fields": [
                    "direct_hand_visibility",
                    "direct_device_classifier_confidence",
                    "gaze_tracking_quality",
                    "pose_landmark_quality",
                ],
                "readable_video_ratio": readable_video_ratio,
                "opened_video_files": opened_video_files,
                "selected_video_files": selected_video_files,
                "total_video_files": total_video_files,
                "sampled_frame_count": total_frames,
            },
        }

    def _motion_threshold(self, baseline: float | None) -> float:
        if baseline is None:
            return float(self.config.get("motion_min_spike_threshold", 18.0))
        multiplier = float(self.config.get("motion_spike_multiplier", 1.8))
        delta = float(self.config.get("motion_spike_delta", 8.0))
        minimum = float(self.config.get("motion_min_spike_threshold", 18.0))
        return max(baseline * multiplier, baseline + delta, minimum)

    def _build_event_windows(
        self,
        *,
        frame_records: list[dict[str, Any]],
        motion_spike_threshold: float,
        near_motion_spike_threshold: float,
        stable_viewing_ratio: float,
    ) -> list[dict[str, Any]]:
        max_event_windows = int(self.config.get("max_event_windows", 5))
        candidates: list[dict[str, Any]] = []
        for frame in frame_records:
            near_motion = frame["near_device_motion"]
            global_motion = frame["global_motion"]
            if near_motion is not None and near_motion >= near_motion_spike_threshold:
                candidates.append(
                    {
                        "relative_position": frame["relative_position"],
                        "event_type": "near_device_motion_burst",
                        "strength": level_bin(
                            near_motion,
                            low=near_motion_spike_threshold,
                            medium=near_motion_spike_threshold * 1.5,
                            high=near_motion_spike_threshold * 2.0,
                        ),
                        "quality": "usable" if frame["quality_usable"] else "partial",
                        "_score": near_motion,
                    }
                )
            elif global_motion is not None and global_motion >= motion_spike_threshold:
                candidates.append(
                    {
                        "relative_position": frame["relative_position"],
                        "event_type": "global_motion_burst",
                        "strength": level_bin(
                            global_motion,
                            low=motion_spike_threshold,
                            medium=motion_spike_threshold * 1.5,
                            high=motion_spike_threshold * 2.0,
                        ),
                        "quality": "usable" if frame["quality_usable"] else "partial",
                        "_score": global_motion,
                    }
                )

        candidates.sort(key=lambda item: item["_score"], reverse=True)
        events = candidates[:max_event_windows]
        if stable_viewing_ratio >= 0.25 and len(events) < max_event_windows:
            events.append(
                {
                    "relative_position": "global",
                    "event_type": "stable_screen_viewing_proxy",
                    "strength": ratio_bin(stable_viewing_ratio),
                    "quality": "aggregate",
                    "_score": stable_viewing_ratio,
                }
            )
        return [{key: value for key, value in event.items() if key != "_score"} for event in events]

    def _overall_quality(
        self,
        *,
        total_frames: int,
        valid_frame_ratio: float,
        readable_video_ratio: float,
    ) -> str:
        if total_frames == 0:
            return "insufficient_video"
        if readable_video_ratio >= 0.8 and valid_frame_ratio >= 0.7:
            return "usable_frame_quality"
        if readable_video_ratio >= 0.5 and valid_frame_ratio >= 0.4:
            return "partial_frame_quality"
        return "low_frame_quality"

    def _build_visual_proxy_lines(self, aggregate: dict[str, Any]) -> list[str]:
        features = aggregate["global_features"]
        quality = aggregate["quality_summary"]
        windows = aggregate["key_window_summary"]
        event_types = windows["event_window_types"]
        event_summary = (
            ", ".join(f"{key}={value}" for key, value in event_types.items())
            if event_types
            else "none"
        )
        return [
            "PriVTE-FlowLite帧级分析状态: computed_by_privte_flowlite",
            f"有效帧质量: {features['valid_frame_ratio_bin']}",
            f"人脸可见性: {features['face_visible_ratio_bin']}",
            f"设备/屏幕样区域可见性: {features['screen_like_region_visible_ratio_bin']}",
            f"人脸与屏幕样区域共现: {features['face_screen_cooccurrence_ratio_bin']}",
            f"近设备运动水平: {features['near_device_motion_level']}",
            f"交互突增窗口数量: {features['interaction_burst_count_bin']}",
            f"稳定观看代理比例: {features['stable_viewing_proxy_ratio_bin']}",
            f"关键事件窗口类型: {event_summary}",
            f"质量总评: {quality['overall']}",
        ]

    def _extract_metadata_fallback(self, person_record: dict[str, Any]) -> dict[str, Any]:
        context = self._base_context(person_record)
        file_entries = video_file_entries(context["clips"])
        max_video_clips = int(self.config.get("max_video_clips", 12))
        selected_entries = evenly_sample(file_entries, max_video_clips)
        duration_bins: Counter[str] = Counter()
        resolution_bins: Counter[str] = Counter()
        fps_bins: Counter[str] = Counter()
        readable = 0
        path_exists = 0
        for file_info in selected_entries:
            path = Path(file_info["path"])
            if not path.exists():
                duration_bins["unknown"] += 1
                resolution_bins["unknown"] += 1
                fps_bins["unknown"] += 1
                continue
            path_exists += 1
            metadata = probe_mp4_box_metadata(path)
            if metadata["readable"]:
                readable += 1
            duration_bins[duration_bucket(metadata["duration_sec"])] += 1
            resolution_bins[
                resolution_bucket(metadata["width"], metadata["height"])
            ] += 1
            fps_bins[fps_bucket(metadata["fps"])] += 1

        visual_proxy_features = [
            "PriVTE-FlowLite帧级分析状态: opencv_unavailable_metadata_fallback",
            "OpenCV不可用，未执行抽帧、ROI聚焦、帧差运动和关键窗口分析。",
            f"视频文件存在比例: {ratio_bin(safe_ratio(path_exists, len(selected_entries)))}",
            f"MP4容器元数据可读比例: {ratio_bin(safe_ratio(readable, len(selected_entries)))}",
            "视频时长分布: "
            + (
                ", ".join(f"{key}={value}" for key, value in sorted(duration_bins.items()))
                or "unknown"
            ),
            "视频分辨率分布: "
            + (
                ", ".join(f"{key}={value}" for key, value in sorted(resolution_bins.items()))
                or "unknown"
            ),
        ]
        quality_summary = {
            "status": "opencv_unavailable_metadata_fallback",
            "overall": "frame_analysis_unavailable",
            "computed_quality_fields": [
                "video_file_presence",
                "container_readability",
                "duration_distribution",
                "resolution_distribution",
            ],
            "missing_quality_fields": [
                "valid_frame_ratio",
                "face_visible_ratio",
                "screen_like_region_visible_ratio",
                "near_device_motion_level",
                "motion_burst_count",
            ],
        }
        return self._build_evidence(
            context=context,
            file_entries=file_entries,
            status="opencv_unavailable_metadata_fallback",
            video_summary={
                "analysis_backend": "mp4_box_parser_fallback",
                "num_video_files": len(file_entries),
                "selected_video_files": len(selected_entries),
                "path_exists_ratio": safe_ratio(path_exists, len(selected_entries)),
                "container_readable_ratio": safe_ratio(readable, len(selected_entries)),
                "duration_bin_counts": sorted_counter(duration_bins),
                "resolution_bin_counts": sorted_counter(resolution_bins),
                "fps_bin_counts": sorted_counter(fps_bins),
                "visual_proxy_features": visual_proxy_features,
                "replaceable_by": "opencv-enabled PriVTE-FlowLite frame analyzer",
            },
            quality_summary=quality_summary,
            missing_information=[
                "frame_level_visual_features",
                "key_window_selection",
                "roi_focusing",
                "near_device_motion_proxy",
                "face_visible_ratio",
                "screen_like_region_visible_ratio",
                "questionnaire_input",
                "exact_heart_rate_input",
                "app_name_input",
            ],
            limitations=[
                "opencv_dependency_missing",
                "metadata_fallback_only",
                "no_frame_level_proxy_features",
                "no_questionnaire_input",
                "no_exact_heart_rate_input",
                "no_app_name_input",
                "not_for_diagnosis",
            ],
        )

    def _build_evidence(
        self,
        *,
        context: dict[str, Any],
        file_entries: list[dict[str, Any]],
        status: str,
        video_summary: dict[str, Any],
        quality_summary: dict[str, Any],
        missing_information: list[str],
        limitations: list[str],
    ) -> dict[str, Any]:
        video_summary = {
            "status": status,
            "num_sessions": context["num_sessions"],
            "num_clips": context["num_clips"],
            "num_video_clips": context["num_video_clips"],
            "video_presence_ratio": safe_ratio(context["num_video_clips"], context["num_clips"]),
            **video_summary,
        }
        return {
            "extractor": self.metadata,
            "modality_availability": {
                "has_video": context["num_video_clips"] > 0,
                "has_heart_rate": context["num_nonempty_heart_rate"] > 0,
                "has_app_usage": context["num_usage_records"] > 0,
                "has_questionnaire": context["has_questionnaire"],
            },
            "feature_blocks": {
                "video_proxy_summary": video_summary,
                "heart_rate_summary": {
                    "status": "availability_only",
                    "num_nonempty_heart_rate_clips": context["num_nonempty_heart_rate"],
                    "exact_values_included": False,
                    "public_policy": "trend_or_quality_only_in_future_versions",
                },
                "app_usage_summary": {
                    "status": "availability_only",
                    "num_usage_records": context["num_usage_records"],
                    "app_names_included": False,
                    "public_policy": "coarse_category_and_duration_bin_only_in_future_versions",
                },
                "questionnaire_status": {
                    "available": context["has_questionnaire"],
                    "used_as_input": False,
                    "reason": "questionnaire is label/context source, not text-only video proxy input",
                },
                "quality_summary": quality_summary,
            },
            "privacy_processing_summary": {
                "raw_video_included": False,
                "raw_images_included": False,
                "raw_audio_included": False,
                "frame_images_included": False,
                "ocr_text_included": False,
                "asr_text_included": False,
                "face_embeddings_included": False,
                "high_dimensional_landmarks_included": False,
                "questionnaire_answers_included": False,
                "exact_heart_rate_values_included": False,
                "app_names_included": False,
                "raw_paths_included": False,
                "exact_timestamps_included": False,
                "video_files_opened_locally": bool(file_entries),
            },
            "missing_information": missing_information,
            "limitations": limitations,
        }
