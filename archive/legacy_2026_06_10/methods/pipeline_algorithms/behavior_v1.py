"""PriVTE-Behavior v1 practical evidence extractor.

This extractor is a stronger, dependency-backed PriVTE algorithm. It uses
off-the-shelf local CV components when available:

- Ultralytics YOLO for device/person-like object detection;
- MediaPipe Hands / Face Mesh / Pose for local landmark-derived proxies;
- OpenCV for frame decoding, image quality, and motion estimates.

The algorithm only emits coarse, privacy-filtered aggregates. It does not emit
raw frames, crops, coordinates, masks, OCR/ASR, face embeddings, or
high-dimensional landmark sequences.
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


def activity_ratio_bin(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value <= 0:
        return "none"
    if value < 0.05:
        return "low"
    if value < 0.15:
        return "medium"
    if value < 0.3:
        return "high"
    return "very_high"


def density_count_bin(value: int, total: int) -> str:
    if value <= 0:
        return "none"
    if total <= 0:
        return count_bin(value)
    return activity_ratio_bin(safe_ratio(value, total))


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


def point_distance(first: tuple[float, float] | None, second: tuple[float, float] | None) -> float | None:
    if first is None or second is None:
        return None
    return ((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5


def point_to_box_distance(point: tuple[float, float], box: tuple[int, int, int, int]) -> float:
    x, y = point
    left, top, width, height = box
    right = left + width
    bottom = top + height
    dx = max(left - x, 0.0, x - right)
    dy = max(top - y, 0.0, y - bottom)
    return (dx * dx + dy * dy) ** 0.5


def max_true_run(values: list[bool]) -> int:
    best = 0
    current = 0
    for value in values:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


class PriVTEBehaviorV1Extractor(EvidenceExtractor):
    """Practical PriVTE behavior-proxy extractor using local CV models."""

    name = "privte_behavior_v1"
    version = "v1"
    feature_schema_version = "privte_behavior_v1_feature_schema.v1"

    def _try_import_cv2(self) -> Any | None:
        try:
            import cv2  # type: ignore[import-not-found]
        except Exception:
            return None
        return cv2

    def _try_import_mediapipe(self) -> Any | None:
        try:
            import mediapipe as mp  # type: ignore[import-not-found]
        except Exception:
            return None
        return mp

    def _task_model_path(self, config_key: str) -> Path | None:
        value = self.config.get(config_key)
        if not value:
            return None
        path = Path(str(value))
        return path if path.exists() else None

    def _mediapipe_backend(self, mp: Any | None) -> tuple[str | None, str]:
        if mp is None:
            return None, "mediapipe_unavailable"
        if hasattr(mp, "solutions"):
            return "solutions", "mediapipe_solutions"
        has_task_models = all(
            self._task_model_path(key) is not None
            for key in (
                "hand_landmarker_model_path",
                "face_landmarker_model_path",
                "pose_landmarker_model_path",
            )
        )
        if hasattr(mp, "tasks") and has_task_models:
            return "tasks", "mediapipe_tasks"
        return None, "mediapipe_task_model_paths_missing"

    def _try_build_yolo(self) -> tuple[Any | None, str]:
        if self.config.get("disable_yolo"):
            return None, "disabled"
        try:
            from ultralytics import YOLO  # type: ignore[import-not-found]
        except Exception:
            return None, "ultralytics_unavailable"
        model_name = str(self.config.get("yolo_model", "yolo11n.pt"))
        try:
            return YOLO(model_name), f"ultralytics:{model_name}"
        except Exception:
            return None, f"ultralytics_model_unavailable:{model_name}"

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
        mp = self._try_import_mediapipe()
        mp_backend, mp_status = self._mediapipe_backend(mp)
        yolo_model, yolo_status = self._try_build_yolo()

        require_dependencies = bool(self.config.get("require_behavior_dependencies", True))
        require_yolo = bool(self.config.get("require_yolo", False))
        missing = []
        if cv2 is None:
            missing.append("opencv-python-headless")
        if mp_backend is None:
            missing.append(mp_status)
        if require_yolo and yolo_model is None:
            missing.append(f"ultralytics_yolo({yolo_status})")
        if missing and require_dependencies:
            raise RuntimeError(
                "PriVTE-Behavior v1 requires local CV dependencies. Missing: "
                + ", ".join(missing)
            )
        if cv2 is None or mp_backend is None:
            return self._extract_metadata_fallback(
                person_record,
                fallback_reason="behavior_dependencies_missing",
                dependency_status={
                    "opencv_available": cv2 is not None,
                    "mediapipe_status": mp_status,
                    "yolo_status": yolo_status,
                },
            )
        return self._extract_with_behavior_models(
            person_record=person_record,
            cv2=cv2,
            mp=mp,
            mp_backend=mp_backend,
            mp_status=mp_status,
            yolo_model=yolo_model,
            yolo_status=yolo_status,
        )

    def _extract_with_behavior_models(
        self,
        *,
        person_record: dict[str, Any],
        cv2: Any,
        mp: Any,
        mp_backend: str,
        mp_status: str,
        yolo_model: Any | None,
        yolo_status: str,
    ) -> dict[str, Any]:
        context = self._base_context(person_record)
        file_entries = video_file_entries(context["clips"])
        max_video_clips = int(self.config.get("max_video_clips", 16))
        selected_entries = evenly_sample(file_entries, max_video_clips)
        try:
            mp_context = self._build_mediapipe_context(mp, mp_backend)
        except Exception as exc:
            mediapipe_status = self._format_mediapipe_context_error(exc)
            if bool(self.config.get("require_behavior_dependencies", True)):
                raise RuntimeError(
                    "PriVTE-Behavior v1 requires MediaPipe runtime dependencies. "
                    f"Missing: {mediapipe_status}"
                ) from exc
            return self._extract_metadata_fallback(
                person_record,
                fallback_reason="behavior_dependencies_missing",
                dependency_status={
                    "opencv_available": cv2 is not None,
                    "mediapipe_status": mediapipe_status,
                    "yolo_status": yolo_status,
                },
            )

        frame_records: list[dict[str, Any]] = []
        readable_video_files = 0
        opened_video_files = 0
        try:
            for clip_order, file_info in enumerate(selected_entries):
                path = Path(file_info["path"])
                if not path.exists():
                    continue
                video_result = self._analyze_video_file(
                    path=path,
                    cv2=cv2,
                    mp_context=mp_context,
                    yolo_model=yolo_model,
                    clip_order=clip_order,
                )
                if video_result["opened"]:
                    opened_video_files += 1
                if video_result["readable"]:
                    readable_video_files += 1
                frame_records.extend(video_result["frames"])
        finally:
            self._close_mediapipe_context(mp_context)

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
            yolo_status=yolo_status,
            mediapipe_status=mp_status,
        )
        visual_proxy_features = self._build_visual_proxy_lines(aggregate)
        video_summary = {
            "analysis_backend": "opencv_mediapipe_yolo_practical",
            "num_video_files": len(file_entries),
            "selected_video_files": len(selected_entries),
            "opened_video_files": opened_video_files,
            "readable_video_files": readable_video_files,
            "sampled_frame_count": len(frame_records),
            "frame_sampling": {
                "max_video_clips": max_video_clips,
                "frames_per_clip": int(self.config.get("frames_per_clip", 12)),
                "analysis_width": int(self.config.get("analysis_width", 768)),
            },
            "detector_summary": aggregate["detector_summary"],
            "behavior_v1_features": aggregate["behavior_v1_features"],
            "reference_normalization": aggregate["reference_normalization"],
            "key_window_summary": aggregate["key_window_summary"],
            "event_windows": aggregate["event_windows"],
            "visual_proxy_features": visual_proxy_features,
            "replaceable_by": "future stronger PriVTE behavior proxy extractor",
        }
        for optional_key in (
            "behavior_v2_features",
            "concrete_behavior_evidence",
            "privacy_preserving_behavior_summary",
            "behavior_v3_temporal_features",
            "temporal_behavior_sequence",
            "temporal_behavior_narrative",
            "temporal_sequence_summary",
            "privte_trace_features",
            "trace_risk_summary",
            "trace_behavior_narrative",
        ):
            if optional_key in aggregate:
                video_summary[optional_key] = aggregate[optional_key]

        return self._build_evidence(
            context=context,
            file_entries=file_entries,
            status=f"computed_by_{self.name}",
            video_summary=video_summary,
            quality_summary=aggregate["quality_summary"],
            missing_information=[
                "exact_touch_events",
                "direct_gaze_estimation",
                "validated_affect_or_fatigue_labels",
                "screen_content_ocr",
                "high_dimensional_pose_or_face_mesh",
                "questionnaire_input",
                "exact_heart_rate_input",
                "app_name_input",
            ],
            limitations=[
                "privte_behavior_v1_practical",
                "device_detection_is_model_or_heuristic_proxy",
                "hand_device_interaction_is_proxy_not_confirmed_touch",
                "head_device_alignment_is_proxy_not_gaze_tracking",
                "pose_change_is_proxy_not_clinical_assessment",
                "no_questionnaire_input",
                "no_exact_heart_rate_input",
                "no_app_name_input",
                "not_for_diagnosis",
            ],
        )

    def _format_mediapipe_context_error(self, exc: Exception) -> str:
        message = str(exc)
        if "libGLESv2.so.2" in message:
            return (
                "mediapipe_native_runtime_unavailable("
                "missing libGLESv2.so.2; on Ubuntu/WSL install libgles2 "
                "or libgles2-mesa)"
            )
        if "libEGL.so" in message:
            return (
                "mediapipe_native_runtime_unavailable("
                "missing libEGL; on Ubuntu/WSL install libegl1)"
            )
        return f"mediapipe_context_initialization_failed({message})"

    def _build_mediapipe_context(self, mp: Any, backend: str) -> dict[str, Any]:
        min_detection_confidence = float(self.config.get("mp_detection_confidence", 0.5))
        max_num_hands = int(self.config.get("max_num_hands", 2))
        if backend == "solutions":
            return {
                "backend": "solutions",
                "mp": mp,
                "hands": mp.solutions.hands.Hands(
                    static_image_mode=True,
                    max_num_hands=max_num_hands,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=float(self.config.get("mp_tracking_confidence", 0.5)),
                ),
                "face_mesh": mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=min_detection_confidence,
                ),
                "pose": mp.solutions.pose.Pose(
                    static_image_mode=True,
                    model_complexity=int(self.config.get("pose_model_complexity", 1)),
                    min_detection_confidence=min_detection_confidence,
                ),
            }

        from mediapipe.tasks.python import BaseOptions  # type: ignore[import-not-found]
        from mediapipe.tasks.python import vision  # type: ignore[import-not-found]

        hand_model = self._task_model_path("hand_landmarker_model_path")
        face_model = self._task_model_path("face_landmarker_model_path")
        pose_model = self._task_model_path("pose_landmarker_model_path")
        if hand_model is None or face_model is None or pose_model is None:
            raise RuntimeError("MediaPipe task model paths are required for tasks backend.")
        return {
            "backend": "tasks",
            "mp": mp,
            "vision": vision,
            "hands": vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=hand_model.as_posix()),
                    running_mode=vision.RunningMode.IMAGE,
                    num_hands=max_num_hands,
                    min_hand_detection_confidence=min_detection_confidence,
                )
            ),
            "face_mesh": vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=face_model.as_posix()),
                    running_mode=vision.RunningMode.IMAGE,
                    num_faces=1,
                    output_face_blendshapes=True,
                    output_facial_transformation_matrixes=True,
                    min_face_detection_confidence=min_detection_confidence,
                )
            ),
            "pose": vision.PoseLandmarker.create_from_options(
                vision.PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=pose_model.as_posix()),
                    running_mode=vision.RunningMode.IMAGE,
                    min_pose_detection_confidence=min_detection_confidence,
                )
            ),
        }

    def _close_mediapipe_context(self, mp_context: dict[str, Any]) -> None:
        for value in mp_context.values():
            close = getattr(value, "close", None)
            if callable(close):
                close()

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
        mp_context: dict[str, Any],
        yolo_model: Any | None,
        clip_order: int,
    ) -> dict[str, Any]:
        frames_per_clip = int(self.config.get("frames_per_clip", 12))
        capture = cv2.VideoCapture(path.as_posix())
        result = {"opened": False, "readable": False, "frames": []}
        previous_hand_center = None
        previous_pose_center = None
        try:
            if not capture.isOpened():
                return result
            result["opened"] = True
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            frame_indices = self._frame_indices(frame_count, frames_per_clip)

            for frame_order, frame_index in enumerate(frame_indices):
                if frame_count > 0:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame = capture.read()
                if not ok:
                    continue
                ok_motion, motion_frame = capture.read()
                if not ok_motion:
                    motion_frame = None
                frame = self._resize_frame(frame, cv2)
                if motion_frame is not None:
                    motion_frame = self._resize_frame(motion_frame, cv2)
                record = self._analyze_frame(
                    frame=frame,
                    motion_frame=motion_frame,
                    previous_hand_center=previous_hand_center,
                    previous_pose_center=previous_pose_center,
                    cv2=cv2,
                    mp_context=mp_context,
                    yolo_model=yolo_model,
                    clip_order=clip_order,
                    frame_order=frame_order,
                )
                result["frames"].append(record)
                previous_hand_center = record.get("_hand_center")
                previous_pose_center = record.get("_pose_center")
            result["readable"] = bool(result["frames"])
        finally:
            capture.release()
        for record in result["frames"]:
            record.pop("_hand_center", None)
            record.pop("_pose_center", None)
        return result

    def _resize_frame(self, frame: Any, cv2: Any) -> Any:
        analysis_width = int(self.config.get("analysis_width", 768))
        height, width = frame.shape[:2]
        if width <= analysis_width:
            return frame
        scale = analysis_width / width
        resized_height = max(1, int(height * scale))
        return cv2.resize(frame, (analysis_width, resized_height))

    def _analyze_frame(
        self,
        *,
        frame: Any,
        motion_frame: Any | None,
        previous_hand_center: tuple[float, float] | None,
        previous_pose_center: tuple[float, float] | None,
        cv2: Any,
        mp_context: dict[str, Any],
        yolo_model: Any | None,
        clip_order: int,
        frame_order: int,
    ) -> dict[str, Any]:
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        brightness = float(gray.mean())
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        quality_usable = self._quality_usable(brightness, blur_score)

        device = self._detect_device_region(frame=frame, gray=gray, cv2=cv2, yolo_model=yolo_model)
        hands = self._detect_hands(
            rgb=rgb,
            mp_context=mp_context,
            device_box=device["box"],
            width=width,
            height=height,
        )
        face = self._detect_face_mesh(
            rgb=rgb,
            mp_context=mp_context,
            device_box=device["box"],
            width=width,
            height=height,
        )
        pose = self._detect_pose(
            rgb=rgb,
            mp_context=mp_context,
            width=width,
            height=height,
        )

        global_motion = None
        near_device_motion = None
        if motion_frame is not None and motion_frame.shape[:2] == frame.shape[:2]:
            motion_gray = cv2.cvtColor(motion_frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray, motion_gray)
            global_motion = float(diff.mean())
            if device["box"] is not None:
                x, y, box_width, box_height = device["box"]
                roi = diff[y : y + box_height, x : x + box_width]
            else:
                roi = diff[int(height * 0.45) :, :]
            near_device_motion = float(roi.mean()) if roi.size else None

        hand_motion = point_distance(hands["center"], previous_hand_center)
        if hand_motion is not None:
            hand_motion = hand_motion / max(width, height)
        pose_motion = point_distance(pose["center"], previous_pose_center)
        if pose_motion is not None:
            pose_motion = pose_motion / max(width, height)

        return {
            "clip_order": clip_order,
            "frame_order": frame_order,
            "brightness": round(brightness, 3),
            "blur_score": round(blur_score, 3),
            "quality_usable": quality_usable,
            "device_visible": device["visible"],
            "device_area_ratio": round(device["area_ratio"], 4),
            "device_backend": device["backend"],
            "hand_visible": hands["visible"],
            "hand_count": hands["count"],
            "hand_device_proximity": hands["device_proximity"],
            "face_visible": face["visible"],
            "face_area_ratio": round(face["area_ratio"], 4),
            "face_device_cooccurrence": face["visible"] and device["visible"],
            "face_device_alignment_proxy": face["device_alignment_proxy"],
            "pose_visible": pose["visible"],
            "global_motion": round(global_motion, 3) if global_motion is not None else None,
            "near_device_motion": (
                round(near_device_motion, 3)
                if near_device_motion is not None
                else None
            ),
            "hand_motion": round(hand_motion, 4) if hand_motion is not None else None,
            "pose_motion": round(pose_motion, 4) if pose_motion is not None else None,
            "_hand_center": hands["center"],
            "_pose_center": pose["center"],
        }

    def _quality_usable(self, brightness: float, blur_score: float) -> bool:
        min_brightness = float(self.config.get("min_brightness", 35.0))
        max_brightness = float(self.config.get("max_brightness", 225.0))
        min_blur_score = float(self.config.get("min_blur_score", 15.0))
        return min_brightness <= brightness <= max_brightness and blur_score >= min_blur_score

    def _detect_device_region(self, *, frame: Any, gray: Any, cv2: Any, yolo_model: Any | None) -> dict[str, Any]:
        if yolo_model is not None:
            yolo_device = self._detect_device_with_yolo(frame, yolo_model)
            if yolo_device["visible"]:
                return yolo_device
        if self.config.get("device_backend", "yolo_or_heuristic") in {
            "heuristic",
            "yolo_or_heuristic",
        }:
            return self._detect_screen_like_region(gray=gray, cv2=cv2)
        return {"visible": False, "area_ratio": 0.0, "box": None, "backend": "none"}

    def _mediapipe_image(self, mp_context: dict[str, Any], rgb: Any) -> Any:
        return mp_context["mp"].Image(
            image_format=mp_context["mp"].ImageFormat.SRGB,
            data=rgb,
        )

    def _detect_device_with_yolo(self, frame: Any, yolo_model: Any) -> dict[str, Any]:
        device_names = {
            str(item).lower()
            for item in self.config.get(
                "device_class_names",
                ["cell phone", "laptop", "tv", "tvmonitor", "remote", "keyboard"],
            )
        }
        confidence_threshold = float(self.config.get("yolo_confidence", 0.25))
        image_size = int(self.config.get("yolo_imgsz", 640))
        height, width = frame.shape[:2]
        frame_area = max(width * height, 1)
        best = {"visible": False, "area_ratio": 0.0, "box": None, "backend": "yolo"}
        try:
            results = yolo_model.predict(
                source=frame,
                imgsz=image_size,
                conf=confidence_threshold,
                verbose=False,
            )
        except Exception:
            return {
                "visible": False,
                "area_ratio": 0.0,
                "box": None,
                "backend": "yolo_predict_failed",
            }
        for result in results:
            names = getattr(result, "names", {})
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = str(names.get(cls_id, cls_id)).lower()
                if cls_name not in device_names:
                    continue
                confidence = float(box.conf[0])
                if confidence < confidence_threshold:
                    continue
                x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                left = max(0, min(width - 1, int(round(x1))))
                top = max(0, min(height - 1, int(round(y1))))
                right = max(left + 1, min(width, int(round(x2))))
                bottom = max(top + 1, min(height, int(round(y2))))
                area_ratio = ((right - left) * (bottom - top)) / frame_area
                score = area_ratio * confidence
                if score > best["area_ratio"]:
                    best = {
                        "visible": True,
                        "area_ratio": area_ratio,
                        "box": (left, top, right - left, bottom - top),
                        "backend": f"yolo:{cls_name}",
                    }
        return best

    def _detect_screen_like_region(self, *, gray: Any, cv2: Any) -> dict[str, Any]:
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
        return {
            "visible": best_box is not None,
            "area_ratio": best_area_ratio,
            "box": best_box,
            "backend": "bright_rectangle_heuristic",
        }

    def _detect_hands(
        self,
        *,
        rgb: Any,
        mp_context: dict[str, Any],
        device_box: tuple[int, int, int, int] | None,
        width: int,
        height: int,
    ) -> dict[str, Any]:
        if mp_context["backend"] == "solutions":
            result = mp_context["hands"].process(rgb)
            landmarks = [
                hand_landmarks.landmark
                for hand_landmarks in (result.multi_hand_landmarks or [])
            ]
        else:
            image = self._mediapipe_image(mp_context, rgb)
            result = mp_context["hands"].detect(image)
            landmarks = result.hand_landmarks or []
        if not landmarks:
            return {
                "visible": False,
                "count": 0,
                "center": None,
                "device_proximity": False,
            }

        points: list[tuple[float, float]] = []
        tip_points: list[tuple[float, float]] = []
        for hand_landmarks in landmarks:
            for index, landmark in enumerate(hand_landmarks):
                point = (float(landmark.x) * width, float(landmark.y) * height)
                points.append(point)
                if index in {4, 8, 12, 16, 20}:
                    tip_points.append(point)
        center = (
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        )
        device_proximity = False
        if device_box is not None:
            threshold = float(self.config.get("hand_device_distance_ratio", 0.08))
            max_distance = threshold * max(width, height)
            device_proximity = any(
                point_to_box_distance(point, device_box) <= max_distance
                for point in tip_points or points
            )
        return {
            "visible": True,
            "count": len(landmarks),
            "center": center,
            "device_proximity": device_proximity,
        }

    def _detect_face_mesh(
        self,
        *,
        rgb: Any,
        mp_context: dict[str, Any],
        device_box: tuple[int, int, int, int] | None,
        width: int,
        height: int,
    ) -> dict[str, Any]:
        if mp_context["backend"] == "solutions":
            result = mp_context["face_mesh"].process(rgb)
            landmarks = [
                face_landmarks.landmark
                for face_landmarks in (result.multi_face_landmarks or [])
            ]
        else:
            image = self._mediapipe_image(mp_context, rgb)
            result = mp_context["face_mesh"].detect(image)
            landmarks = result.face_landmarks or []
        if not landmarks:
            return {
                "visible": False,
                "area_ratio": 0.0,
                "center": None,
                "device_alignment_proxy": False,
            }
        points = [
            (float(landmark.x) * width, float(landmark.y) * height)
            for landmark in landmarks[0]
        ]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        left, right = min(xs), max(xs)
        top, bottom = min(ys), max(ys)
        area_ratio = ((right - left) * (bottom - top)) / max(width * height, 1)
        center = ((left + right) / 2, (top + bottom) / 2)
        device_alignment_proxy = False
        if device_box is not None:
            device_center = (
                device_box[0] + device_box[2] / 2,
                device_box[1] + device_box[3] / 2,
            )
            distance = point_distance(center, device_center)
            if distance is not None:
                device_alignment_proxy = distance <= float(
                    self.config.get("face_device_distance_ratio", 0.65)
                ) * max(width, height)
        return {
            "visible": True,
            "area_ratio": area_ratio,
            "center": center,
            "device_alignment_proxy": device_alignment_proxy,
        }

    def _detect_pose(self, *, rgb: Any, mp_context: dict[str, Any], width: int, height: int) -> dict[str, Any]:
        if mp_context["backend"] == "solutions":
            result = mp_context["pose"].process(rgb)
            pose_landmarks = getattr(result, "pose_landmarks", None)
            landmarks = pose_landmarks.landmark if pose_landmarks is not None else []
        else:
            image = self._mediapipe_image(mp_context, rgb)
            result = mp_context["pose"].detect(image)
            pose_landmark_sets = result.pose_landmarks or []
            landmarks = pose_landmark_sets[0] if pose_landmark_sets else []
        if not landmarks:
            return {"visible": False, "center": None}
        selected = []
        for index in [0, 11, 12, 23, 24]:
            if index >= len(landmarks):
                continue
            landmark = landmarks[index]
            visibility = getattr(landmark, "visibility", 1.0)
            if visibility >= float(self.config.get("pose_visibility_threshold", 0.5)):
                selected.append((float(landmark.x) * width, float(landmark.y) * height))
        if not selected:
            return {"visible": False, "center": None}
        center = (
            sum(point[0] for point in selected) / len(selected),
            sum(point[1] for point in selected) / len(selected),
        )
        return {"visible": True, "center": center}

    def _aggregate_frames(
        self,
        *,
        frame_records: list[dict[str, Any]],
        readable_video_files: int,
        opened_video_files: int,
        selected_video_files: int,
        total_video_files: int,
        yolo_status: str,
        mediapipe_status: str,
    ) -> dict[str, Any]:
        total_frames = len(frame_records)
        valid_frames = sum(1 for frame in frame_records if frame["quality_usable"])
        device_visible_frames = sum(1 for frame in frame_records if frame["device_visible"])
        hand_visible_frames = sum(1 for frame in frame_records if frame["hand_visible"])
        hand_device_frames = sum(
            1 for frame in frame_records if frame["hand_device_proximity"]
        )
        face_visible_frames = sum(1 for frame in frame_records if frame["face_visible"])
        face_device_frames = sum(
            1 for frame in frame_records if frame["face_device_cooccurrence"]
        )
        face_alignment_frames = sum(
            1 for frame in frame_records if frame["face_device_alignment_proxy"]
        )
        pose_visible_frames = sum(1 for frame in frame_records if frame["pose_visible"])

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
        hand_motion_values = [
            frame["hand_motion"]
            for frame in frame_records
            if frame["hand_motion"] is not None
        ]
        pose_motion_values = [
            frame["pose_motion"]
            for frame in frame_records
            if frame["pose_motion"] is not None
        ]
        brightness_values = [frame["brightness"] for frame in frame_records]
        blur_values = [frame["blur_score"] for frame in frame_records]

        global_motion_threshold = self._absolute_motion_threshold(median_value(global_motion_values))
        near_motion_threshold = self._absolute_motion_threshold(median_value(near_motion_values))
        hand_motion_threshold = self._relative_motion_threshold(
            median_value(hand_motion_values),
            minimum=float(self.config.get("hand_motion_min_spike_threshold", 0.06)),
        )
        pose_motion_threshold = self._relative_motion_threshold(
            median_value(pose_motion_values),
            minimum=float(self.config.get("pose_motion_min_spike_threshold", 0.04)),
        )

        stable_flags = [
            frame["device_visible"]
            and frame["quality_usable"]
            and (
                frame["global_motion"] is None
                or frame["global_motion"] <= global_motion_threshold
            )
            and (
                frame["face_device_alignment_proxy"]
                or frame["hand_device_proximity"]
                or frame["face_device_cooccurrence"]
            )
            for frame in frame_records
        ]
        device_region_activity_flags = [
            (
                frame["device_visible"]
                and frame["near_device_motion"] is not None
                and frame["near_device_motion"] >= near_motion_threshold
            )
            for frame in frame_records
        ]
        active_interaction_flags = [
            frame["hand_device_proximity"] or device_region_activity_flags[index]
            for index, frame in enumerate(frame_records)
        ]
        repetitive_operation_count = sum(
            1
            for frame in frame_records
            if frame["hand_device_proximity"]
            and frame["hand_motion"] is not None
            and frame["hand_motion"] >= hand_motion_threshold
        )
        device_motion_burst_count = sum(device_region_activity_flags)
        posture_change_count = sum(
            1
            for frame in frame_records
            if frame["pose_motion"] is not None
            and frame["pose_motion"] >= pose_motion_threshold
            and (
                frame["global_motion"] is not None
                and frame["global_motion"] >= global_motion_threshold
            )
        )

        valid_frame_ratio = safe_ratio(valid_frames, total_frames)
        readable_video_ratio = safe_ratio(readable_video_files, selected_video_files)
        stable_engagement_ratio = safe_ratio(sum(stable_flags), total_frames)
        active_interaction_ratio = safe_ratio(sum(active_interaction_flags), total_frames)
        device_region_activity_ratio = safe_ratio(
            sum(device_region_activity_flags),
            total_frames,
        )
        max_stable_run = max_true_run(stable_flags)

        behavior_features = {
            "valid_frame_ratio": valid_frame_ratio,
            "valid_frame_ratio_bin": ratio_bin(valid_frame_ratio),
            "device_visible_ratio": safe_ratio(device_visible_frames, total_frames),
            "device_visible_ratio_bin": ratio_bin(safe_ratio(device_visible_frames, total_frames)),
            "hand_visible_ratio": safe_ratio(hand_visible_frames, total_frames),
            "hand_visible_ratio_bin": ratio_bin(safe_ratio(hand_visible_frames, total_frames)),
            "hand_device_proximity_ratio": safe_ratio(hand_device_frames, total_frames),
            "hand_device_proximity_ratio_bin": ratio_bin(
                safe_ratio(hand_device_frames, total_frames)
            ),
            "face_visible_ratio": safe_ratio(face_visible_frames, total_frames),
            "face_visible_ratio_bin": ratio_bin(safe_ratio(face_visible_frames, total_frames)),
            "face_device_cooccurrence_ratio": safe_ratio(face_device_frames, total_frames),
            "face_device_cooccurrence_ratio_bin": ratio_bin(
                safe_ratio(face_device_frames, total_frames)
            ),
            "face_device_alignment_proxy_ratio": safe_ratio(
                face_alignment_frames,
                total_frames,
            ),
            "face_device_alignment_proxy_ratio_bin": ratio_bin(
                safe_ratio(face_alignment_frames, total_frames)
            ),
            "pose_visible_ratio": safe_ratio(pose_visible_frames, total_frames),
            "pose_visible_ratio_bin": ratio_bin(safe_ratio(pose_visible_frames, total_frames)),
            "stable_screen_engagement_proxy_ratio": stable_engagement_ratio,
            "stable_screen_engagement_proxy_ratio_bin": ratio_bin(stable_engagement_ratio),
            "active_hand_device_interaction_proxy_ratio": active_interaction_ratio,
            "active_hand_device_interaction_proxy_ratio_bin": ratio_bin(
                active_interaction_ratio
            ),
            "device_region_activity_proxy_ratio": device_region_activity_ratio,
            "device_region_activity_proxy_ratio_bin": activity_ratio_bin(
                device_region_activity_ratio
            ),
            "repetitive_operation_proxy_count": repetitive_operation_count,
            "repetitive_operation_proxy_count_bin": count_bin(repetitive_operation_count),
            "device_region_activity_proxy_count": device_motion_burst_count,
            "device_region_activity_proxy_count_bin": density_count_bin(
                device_motion_burst_count,
                total_frames,
            ),
            "device_motion_burst_count": device_motion_burst_count,
            "device_motion_burst_count_bin": density_count_bin(
                device_motion_burst_count,
                total_frames,
            ),
            "posture_or_context_change_count": posture_change_count,
            "posture_or_context_change_count_bin": count_bin(posture_change_count),
            "max_continuous_stable_engagement_frames": max_stable_run,
            "max_continuous_stable_engagement_bin": count_bin(max_stable_run),
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
            stable_flags=stable_flags,
            active_interaction_flags=active_interaction_flags,
            near_motion_threshold=near_motion_threshold,
            hand_motion_threshold=hand_motion_threshold,
            pose_motion_threshold=pose_motion_threshold,
            global_motion_threshold=global_motion_threshold,
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
            "detector_summary": {
                "device_detector": yolo_status,
                "device_fallback": "bright_rectangle_heuristic",
                "mediapipe_status": mediapipe_status,
                "hands_backend": "mediapipe_hands",
                "face_backend": "mediapipe_face_mesh",
                "pose_backend": "mediapipe_pose",
                "raw_coordinates_included": False,
                "frame_images_included": False,
                "landmark_sequences_included": False,
            },
            "behavior_v1_features": behavior_features,
            "reference_normalization": {
                "method": "within_person_motion_reference",
                "global_motion_baseline_level": level_bin(
                    median_value(global_motion_values),
                    low=4.0,
                    medium=12.0,
                    high=24.0,
                ),
                "near_device_motion_baseline_level": level_bin(
                    median_value(near_motion_values),
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
                "status": "computed_by_privte_behavior_v1",
                "overall": overall_quality,
                "computed_quality_fields": [
                    "readable_video_ratio",
                    "valid_frame_ratio",
                    "device_visible_ratio",
                    "hand_visible_ratio",
                    "hand_device_proximity_ratio",
                    "face_visible_ratio",
                    "face_device_cooccurrence_ratio",
                    "pose_visible_ratio",
                    "stable_screen_engagement_proxy_ratio",
                    "active_hand_device_interaction_proxy_ratio",
                    "device_region_activity_proxy_ratio",
                ],
                "missing_quality_fields": [
                    "validated_touch_labels",
                    "direct_gaze_tracking_quality",
                    "validated_affect_or_fatigue_quality",
                ],
                "readable_video_ratio": readable_video_ratio,
                "opened_video_files": opened_video_files,
                "selected_video_files": selected_video_files,
                "total_video_files": total_video_files,
                "sampled_frame_count": total_frames,
            },
        }

    def _absolute_motion_threshold(self, baseline: float | None) -> float:
        if baseline is None:
            return float(self.config.get("motion_min_spike_threshold", 18.0))
        multiplier = float(self.config.get("motion_spike_multiplier", 1.8))
        delta = float(self.config.get("motion_spike_delta", 8.0))
        minimum = float(self.config.get("motion_min_spike_threshold", 18.0))
        return max(baseline * multiplier, baseline + delta, minimum)

    def _relative_motion_threshold(self, baseline: float | None, *, minimum: float) -> float:
        if baseline is None:
            return minimum
        return max(baseline * 1.8, baseline + minimum / 2, minimum)

    def _build_event_windows(
        self,
        *,
        frame_records: list[dict[str, Any]],
        stable_flags: list[bool],
        active_interaction_flags: list[bool],
        near_motion_threshold: float,
        hand_motion_threshold: float,
        pose_motion_threshold: float,
        global_motion_threshold: float,
    ) -> list[dict[str, Any]]:
        max_event_windows = int(self.config.get("max_event_windows", 8))
        candidates: list[dict[str, Any]] = []
        for index, frame in enumerate(frame_records):
            if (
                frame["hand_device_proximity"]
                and frame["hand_motion"] is not None
                and frame["hand_motion"] >= hand_motion_threshold
            ):
                candidates.append(
                    {
                        "relative_position": frame["relative_position"],
                        "event_type": "hand_device_interaction_burst",
                        "strength": level_bin(
                            frame["hand_motion"],
                            low=hand_motion_threshold,
                            medium=hand_motion_threshold * 1.5,
                            high=hand_motion_threshold * 2.0,
                        ),
                        "quality": "usable" if frame["quality_usable"] else "partial",
                        "_score": frame["hand_motion"] * 100,
                    }
                )
            elif active_interaction_flags[index]:
                score = frame["near_device_motion"] or near_motion_threshold
                candidates.append(
                    {
                        "relative_position": frame["relative_position"],
                        "event_type": "device_region_motion_burst",
                        "strength": level_bin(
                            score,
                            low=near_motion_threshold,
                            medium=near_motion_threshold * 1.5,
                            high=near_motion_threshold * 2.0,
                        ),
                        "quality": "usable" if frame["quality_usable"] else "partial",
                        "_score": score,
                    }
                )
            elif (
                frame["pose_motion"] is not None
                and frame["pose_motion"] >= pose_motion_threshold
                and frame["global_motion"] is not None
                and frame["global_motion"] >= global_motion_threshold
            ):
                score = (frame["pose_motion"] or 0) * 100 + (frame["global_motion"] or 0)
                candidates.append(
                    {
                        "relative_position": frame["relative_position"],
                        "event_type": "posture_or_context_motion_burst",
                        "strength": "medium",
                        "quality": "usable" if frame["quality_usable"] else "partial",
                        "_score": score,
                    }
                )
            elif (
                frame["global_motion"] is not None
                and frame["global_motion"] >= global_motion_threshold
            ):
                candidates.append(
                    {
                        "relative_position": frame["relative_position"],
                        "event_type": "global_motion_burst",
                        "strength": "medium",
                        "quality": "usable" if frame["quality_usable"] else "partial",
                        "_score": frame["global_motion"],
                    }
                )

        if any(stable_flags):
            candidates.append(
                {
                    "relative_position": "global",
                    "event_type": "stable_screen_engagement_proxy",
                    "strength": ratio_bin(safe_ratio(sum(stable_flags), len(stable_flags))),
                    "quality": "aggregate",
                    "_score": sum(stable_flags),
                }
            )

        candidates.sort(key=lambda item: item["_score"], reverse=True)
        max_by_type = {
            "hand_device_interaction_burst": int(
                self.config.get("max_hand_event_windows", 4)
            ),
            "device_region_motion_burst": int(
                self.config.get("max_device_event_windows", 5)
            ),
            "posture_or_context_motion_burst": int(
                self.config.get("max_posture_event_windows", 2)
            ),
            "global_motion_burst": int(self.config.get("max_global_event_windows", 2)),
            "stable_screen_engagement_proxy": int(
                self.config.get("max_stable_event_windows", 1)
            ),
        }
        selected = []
        selected_counts: Counter[str] = Counter()
        for candidate in candidates:
            event_type = candidate["event_type"]
            if selected_counts[event_type] >= max_by_type.get(event_type, max_event_windows):
                continue
            selected.append(candidate)
            selected_counts[event_type] += 1
            if len(selected) >= max_event_windows:
                break
        events = selected
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
            return "usable_behavior_frame_quality"
        if readable_video_ratio >= 0.5 and valid_frame_ratio >= 0.4:
            return "partial_behavior_frame_quality"
        return "low_behavior_frame_quality"

    def _build_visual_proxy_lines(self, aggregate: dict[str, Any]) -> list[str]:
        features = aggregate["behavior_v1_features"]
        event_types = aggregate["key_window_summary"]["event_window_types"]
        event_summary = (
            ", ".join(f"{key}={value}" for key, value in event_types.items())
            if event_types
            else "none"
        )
        quality = aggregate["quality_summary"]
        return [
            "PriVTE-Behavior v1分析状态: computed_by_privte_behavior_v1",
            f"设备/屏幕可观察性: {features['device_visible_ratio_bin']}",
            f"手部可观察性: {features['hand_visible_ratio_bin']}",
            f"手-设备接近代理: {features['hand_device_proximity_ratio_bin']}",
            f"稳定屏幕参与代理: {features['stable_screen_engagement_proxy_ratio_bin']}",
            f"活跃手-设备交互代理: {features['active_hand_device_interaction_proxy_ratio_bin']}",
            f"设备区域活动代理: {features['device_region_activity_proxy_ratio_bin']}",
            f"重复操作代理窗口数量: {features['repetitive_operation_proxy_count_bin']}",
            f"人脸-设备上下文可观察性: {features['face_device_cooccurrence_ratio_bin']}",
            f"姿态/场景变化代理: {features['posture_or_context_change_count_bin']}",
            f"关键事件窗口类型: {event_summary}",
            f"质量总评: {quality['overall']}",
        ]

    def _extract_metadata_fallback(
        self,
        person_record: dict[str, Any],
        *,
        fallback_reason: str,
        dependency_status: dict[str, Any],
    ) -> dict[str, Any]:
        context = self._base_context(person_record)
        file_entries = video_file_entries(context["clips"])
        max_video_clips = int(self.config.get("max_video_clips", 16))
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
            f"PriVTE-Behavior v1分析状态: {fallback_reason}",
            "行为级依赖不可用，未执行设备检测、手部关键点、人脸网格、姿态和交互代理分析。",
            f"视频文件存在比例: {ratio_bin(safe_ratio(path_exists, len(selected_entries)))}",
            f"MP4容器元数据可读比例: {ratio_bin(safe_ratio(readable, len(selected_entries)))}",
        ]
        quality_summary = {
            "status": fallback_reason,
            "overall": "behavior_frame_analysis_unavailable",
            "computed_quality_fields": [
                "video_file_presence",
                "container_readability",
                "duration_distribution",
                "resolution_distribution",
            ],
            "missing_quality_fields": [
                "device_visible_ratio",
                "hand_visible_ratio",
                "hand_device_proximity_ratio",
                "face_device_cooccurrence_ratio",
                "pose_visible_ratio",
            ],
        }
        return self._build_evidence(
            context=context,
            file_entries=file_entries,
            status=fallback_reason,
            video_summary={
                "analysis_backend": "metadata_fallback",
                "dependency_status": dependency_status,
                "num_video_files": len(file_entries),
                "selected_video_files": len(selected_entries),
                "path_exists_ratio": safe_ratio(path_exists, len(selected_entries)),
                "container_readable_ratio": safe_ratio(readable, len(selected_entries)),
                "duration_bin_counts": sorted_counter(duration_bins),
                "resolution_bin_counts": sorted_counter(resolution_bins),
                "fps_bin_counts": sorted_counter(fps_bins),
                "visual_proxy_features": visual_proxy_features,
                "replaceable_by": "dependency-enabled PriVTE-Behavior v1 extractor",
            },
            quality_summary=quality_summary,
            missing_information=[
                "device_detector",
                "hand_landmarks",
                "face_mesh",
                "pose_landmarks",
                "hand_device_interaction_proxy",
                "stable_screen_engagement_proxy",
                "questionnaire_input",
                "exact_heart_rate_input",
                "app_name_input",
            ],
            limitations=[
                "behavior_dependencies_missing",
                "metadata_fallback_only",
                "no_behavior_proxy_features",
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
