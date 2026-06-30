"""PriVTE preprocessor v0."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from pathlib import Path
from typing import Any

from ..core import EvidenceExtractor
from ..schemas import PREPROCESSOR_SCHEMA_VERSION, build_preprocessor_evidence
from .common import (
    collect_clips,
    count_available,
    count_nonempty_json,
    duration_bucket,
    probe_mp4_box_metadata,
    safe_ratio,
    video_file_entries,
)


class PriVTEPreprocessorV0Extractor(EvidenceExtractor):
    """Schema-first local video preprocessor skeleton."""

    name = "privte_preprocessor_v0"
    version = "v0"
    feature_schema_version = PREPROCESSOR_SCHEMA_VERSION

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
        video_entries = video_file_entries(clips)
        internal_timeline = self._build_internal_timeline(video_entries)
        metadata_summary = self._build_metadata_summary(video_entries, internal_timeline)
        coverage_summary = self._select_coverage_windows(video_entries)
        frame_summary = self._analyze_coverage_windows(
            video_entries,
            coverage_summary["selected_indices"],
            internal_timeline,
        )
        event_windows = frame_summary["event_windows"]

        privacy_processing_summary = {
            "processing_steps": [
                "raw video remains local to the preprocessing environment",
                "video frames are sampled locally and immediately compressed into low-dimensional proxy facts",
                "event windows are represented by relative time periods and coarse categories only",
                "raw paths, absolute timestamps, OCR, ASR, app names, screen content, and raw media are suppressed",
            ],
        }
        preprocessor_evidence = build_preprocessor_evidence(
            session_id=self._session_id(person_record),
            total_valid_duration_minutes=metadata_summary[
                "total_valid_duration_minutes"
            ],
            duration_bin=metadata_summary["duration_bin"],
            valid_observation_duration_bin=metadata_summary[
                "valid_observation_duration_bin"
            ],
            analyzed_window_count=coverage_summary["coverage_window_count"],
            event_window_count=len(event_windows),
            global_features=frame_summary["global_features"],
            event_windows=event_windows,
            quality_summary={
                "overall_data_sufficiency": metadata_summary[
                    "overall_data_sufficiency"
                ],
                **frame_summary["quality_summary"],
                "limitations": frame_summary["limitations"],
            },
            privacy_processing_summary=privacy_processing_summary,
            limitations=[
                *frame_summary["limitations"],
                "screen_orientation_uses_visible_face_device_context",
            ],
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
                "preprocessor_evidence": preprocessor_evidence,
                "video_proxy_summary": {
                    "status": frame_summary["status"],
                    "num_sessions": num_sessions,
                    "num_clips": num_clips,
                    "num_video_clips": num_video_clips,
                    "selected_video_clips_analyzed": coverage_summary[
                        "coverage_window_count"
                    ],
                    "sampled_frame_count": frame_summary["sampled_frame_count"],
                    "coverage_relative_position_counts": coverage_summary[
                        "relative_position_counts"
                    ],
                    "metadata_probe_summary": metadata_summary,
                    "key_window_summary": coverage_summary,
                    "frame_analysis_summary": frame_summary["frame_analysis_summary"],
                    "session_metadata": preprocessor_evidence["session_metadata"],
                    "global_features": preprocessor_evidence["global_features"],
                    "event_windows": preprocessor_evidence["event_windows"],
                },
                "quality_summary": preprocessor_evidence["quality_summary"],
                "heart_rate_summary": {
                    "status": "availability_only",
                    "num_nonempty_heart_rate_clips": num_nonempty_heart_rate,
                    "exact_values_included": False,
                    "public_policy": "excluded_from_video_only_model_input",
                },
                "app_usage_summary": {
                    "status": "availability_only",
                    "num_usage_records": num_usage_records,
                    "app_names_included": False,
                    "public_policy": "excluded_from_video_only_model_input",
                },
                "questionnaire_status": {
                    "available": has_questionnaire,
                    "used_as_input": False,
                    "reason": (
                        "questionnaire is label/context source, not text-only video evidence input"
                    ),
                },
            },
            "privacy_processing_summary": preprocessor_evidence[
                "privacy_processing_summary"
            ],
            "missing_information": [
                *frame_summary["missing_information"],
            ],
            "limitations": preprocessor_evidence["limitations"],
        }

    def _session_id(self, person_record: dict[str, Any]) -> str:
        salt = str(self.config.get("session_id_salt", self.name))
        raw = f"{salt}:{person_record.get('person_uid', 'unknown')}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10].upper()
        return f"ANON_{digest}"

    def _build_internal_timeline(
        self,
        video_entries: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build exact local timing context that is never emitted directly."""

        durations: list[float | None] = []
        readable_count = 0
        total_duration_sec = 0.0
        starts: list[float] = []
        for entry in video_entries:
            starts.append(total_duration_sec)
            path_value = entry.get("path")
            metadata = (
                probe_mp4_box_metadata(self._resolve_video_path(path_value))
                if path_value
                else {"readable": False, "duration_sec": None}
            )
            duration = metadata.get("duration_sec")
            if metadata.get("readable") and duration:
                duration_float = float(duration)
                durations.append(duration_float)
                total_duration_sec += duration_float
                readable_count += 1
            else:
                durations.append(None)

        return {
            "durations_sec": durations,
            "starts_sec": starts,
            "total_duration_sec": total_duration_sec,
            "readable_count": readable_count,
        }

    def _build_metadata_summary(
        self,
        video_entries: list[dict[str, Any]],
        timeline: dict[str, Any],
    ) -> dict[str, Any]:
        max_probe = int(self.config.get("max_metadata_video_files", 128))
        indexed_entries = self._select_indexed_entries(video_entries, max_probe)
        probed = [
            {
                "readable": timeline["durations_sec"][index] is not None,
                "duration_sec": timeline["durations_sec"][index],
            }
            for index, _ in indexed_entries
        ]

        readable = [item for item in probed if item.get("readable")]
        total_duration_sec = float(timeline.get("total_duration_sec") or 0.0)
        readable_ratio = safe_ratio(len(readable), len(probed))
        total_minutes = self._coarse_minutes(total_duration_sec)
        return {
            "method": "mp4_container_metadata_probe",
            "total_video_files": len(video_entries),
            "probed_video_files": len(probed),
            "readable_video_files": len(readable),
            "readable_video_ratio_bin": self._ratio_bin(readable_ratio),
            "total_valid_duration_minutes": total_minutes,
            "duration_granularity": "rounded_to_nearest_5_minutes",
            "duration_bin": self._duration_minutes_bin(total_duration_sec),
            "valid_observation_duration_bin": self._duration_minutes_bin(
                total_duration_sec
            ),
            "overall_data_sufficiency": self._data_sufficiency(
                total_video_files=len(video_entries),
                readable_ratio=readable_ratio,
                readable_count=len(readable),
            ),
            "raw_paths_included": False,
            "exact_durations_included": False,
        }

    def _select_coverage_windows(
        self,
        video_entries: list[dict[str, Any]],
    ) -> dict[str, Any]:
        coverage_count = int(self.config.get("coverage_window_count", 12))
        selected = self._select_indexed_entries(video_entries, coverage_count)
        positions = [
            self._relative_position(index, len(video_entries))
            for index, _ in selected
        ]
        return {
            "method": "even_clip_order_coverage_selection",
            "coverage_window_count": len(selected),
            "relative_position_counts": dict(Counter(positions)),
            "selected_indices": [index for index, _ in selected],
            "event_detection_status": "frame_proxy_enabled",
            "raw_paths_included": False,
            "exact_timestamps_included": False,
        }

    def _analyze_coverage_windows(
        self,
        video_entries: list[dict[str, Any]],
        selected_indices: list[int],
        timeline: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            import cv2  # type: ignore[import-not-found]
            import numpy as np  # type: ignore[import-not-found]
        except ImportError:
            return self._empty_frame_summary(
                status="preprocessor_v0_metadata_only_opencv_unavailable",
                limitations=["opencv_unavailable_frame_analysis_not_run"],
            )

        visual_resources = self._build_visual_resources(cv2)
        analyses = []
        try:
            for order, entry_index in enumerate(selected_indices, start=1):
                if entry_index >= len(video_entries):
                    continue
                path_value = video_entries[entry_index].get("path")
                if not path_value:
                    continue
                duration_sec = self._duration_for_index(timeline, entry_index)
                analysis = self._analyze_one_video_window(
                    cv2=cv2,
                    np=np,
                    visual_resources=visual_resources,
                    path=self._resolve_video_path(path_value),
                    window_order=order,
                    relative_position=self._relative_position(
                        entry_index,
                        len(video_entries),
                    ),
                    relative_time_period=self._relative_time_period(
                        timeline,
                        entry_index,
                    ),
                    duration_bin=duration_bucket(duration_sec),
                )
                analyses.append(analysis)
        finally:
            self._close_visual_resources(visual_resources)

        if not analyses:
            return self._empty_frame_summary(
                status="preprocessor_v0_metadata_only_no_readable_frames",
                limitations=["frame_sampling_produced_no_readable_frames"],
            )
        return self._summarize_frame_analyses(analyses)

    def _analyze_one_video_window(
        self,
        *,
        cv2: Any,
        np: Any,
        visual_resources: dict[str, Any],
        path: Path,
        window_order: int,
        relative_position: str,
        relative_time_period: str,
        duration_bin: str,
    ) -> dict[str, Any]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return self._empty_window_analysis(
                window_order,
                relative_position,
                relative_time_period,
                duration_bin,
            )

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        samples_per_window = int(self.config.get("frames_per_coverage_window", 6))
        indices = self._sample_frame_indices(frame_count, samples_per_window)
        frames = []
        for frame_index in indices:
            if frame_index is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(self._resize_frame(cv2, frame))
        cap.release()

        if not frames:
            return self._empty_window_analysis(
                window_order,
                relative_position,
                relative_time_period,
                duration_bin,
            )

        frame_metrics = [
            self._frame_metrics(cv2, np, visual_resources, frame) for frame in frames
        ]
        motion_metrics = [
            self._motion_metrics(cv2, np, previous, current)
            for previous, current in zip(frames, frames[1:])
        ]
        return self._window_summary(
            window_order=window_order,
            relative_position=relative_position,
            relative_time_period=relative_time_period,
            duration_bin=duration_bin,
            frame_metrics=frame_metrics,
            motion_metrics=motion_metrics,
        )

    def _summarize_frame_analyses(self, analyses: list[dict[str, Any]]) -> dict[str, Any]:
        sampled_frame_count = sum(item["sampled_frame_count"] for item in analyses)
        readable_windows = [item for item in analyses if item["sampled_frame_count"] > 0]
        device_visible_windows = [
            item for item in readable_windows if item["device_visibility_level"] != "none"
        ]
        face_visible_windows = [
            item for item in readable_windows if item["face_visibility_level"] != "none"
        ]
        hand_visible_windows = [
            item for item in readable_windows if item["hand_visibility_level"] != "none"
        ]
        face_mesh_windows = [
            item for item in readable_windows if item["face_mesh_visibility_level"] != "none"
        ]
        interaction_levels = [item["interaction_intensity"] for item in readable_windows]
        repetitive_windows = sum(
            1 for item in readable_windows if item["local_motion_burst_count"] >= 2
        )
        global_motion_levels = [
            item["global_motion_level"] for item in readable_windows
        ]
        stable_screen_windows = sum(
            1 for item in readable_windows if item["stable_screen_context"]
        )
        multi_person_windows = sum(
            1 for item in readable_windows if item["multi_person_interference"]
        )
        screen_orientation_summary = self._screen_orientation_proxy_summary(
            readable_windows
        )
        screen_gaze_ratio = screen_orientation_summary["ratio"]
        blink_proxy_summary = self._blink_proxy_summary(readable_windows)
        facial_action_proxy_counts = self._facial_action_proxy_counts(readable_windows)

        event_windows = self._build_event_windows(readable_windows)
        quality_summary = {
            "face_observability": self._ratio_bin(
                safe_ratio(len(face_visible_windows), len(readable_windows))
            ),
            "hand_observability": self._ratio_bin(
                safe_ratio(len(hand_visible_windows), len(readable_windows))
            ),
            "device_observability": self._ratio_bin(
                safe_ratio(len(device_visible_windows), len(readable_windows))
            ),
            "gaze_estimation_quality": self._gaze_estimation_quality(
                screen_orientation_summary,
            ),
            "multi_person_interference": self._multi_person_level(
                multi_person_windows,
                len(readable_windows),
            ),
            "motion_confounding_level": self._dominant_motion_level(
                global_motion_levels
            ),
        }
        global_features = {
            "screen_gaze_ratio": screen_gaze_ratio,
            "screen_gaze_ratio_bin": screen_orientation_summary["ratio_bin"],
            "screen_orientation_proxy_status": screen_orientation_summary["status"],
            "screen_orientation_proxy_quality": screen_orientation_summary["quality"],
            "max_continuous_gaze_duration_minutes": self._max_continuous_proxy_minutes(
                readable_windows,
                screen_orientation_summary["status"],
            ),
            "max_continuous_gaze_duration_bin": self._continuous_duration_bin(
                self._max_continuous_proxy_minutes(
                    readable_windows,
                    screen_orientation_summary["status"],
                )
            ),
            "average_blink_rate_per_minute": None,
            "blink_rate_level": blink_proxy_summary["blink_rate_level"],
            "blink_rate_trend": blink_proxy_summary["blink_rate_trend"],
            "overall_posture_trend": self._posture_trend(readable_windows),
            "interaction_intensity": self._display_level(
                self._dominant_intensity(interaction_levels)
            ),
            "repetitive_operation_level": self._count_level(repetitive_windows),
            "motion_confounding_level": quality_summary["motion_confounding_level"],
            "device_visibility_level": quality_summary["device_observability"],
            "hand_visibility_level": quality_summary["hand_observability"],
            "face_mesh_visibility_level": self._ratio_bin(
                safe_ratio(len(face_mesh_windows), len(readable_windows))
            ),
            "eye_closure_proxy_level": blink_proxy_summary[
                "eye_closure_proxy_level"
            ],
            "facial_action_proxy_counts": facial_action_proxy_counts,
            "stable_screen_context_level": self._ratio_bin(
                safe_ratio(stable_screen_windows, len(readable_windows))
            ),
            "event_window_count": len(event_windows),
        }
        limitations = [
            "preprocessor_v0_uses_lightweight_frame_proxies",
            "screen_or_device_visibility_is_heuristic",
            "hand_device_interaction_uses_hand_landmark_visibility_and_motion_proxy",
            "blink_rate_is_sparse_frame_proxy_not_exact_blink_counter",
            "facial_action_units_are_geometry_proxies_not_emotion_diagnosis",
        ]
        missing_information = ["validated_eye_tracking", "clinical_facial_au_labels"]
        return {
            "status": "preprocessor_v0_frame_proxy_analysis",
            "sampled_frame_count": sampled_frame_count,
            "global_features": global_features,
            "event_windows": event_windows,
            "quality_summary": quality_summary,
            "limitations": limitations,
            "missing_information": missing_information,
            "frame_analysis_summary": {
                "analyzed_window_count": len(readable_windows),
                "sampled_frame_count": sampled_frame_count,
                "device_visible_window_count": len(device_visible_windows),
                "face_visible_window_count": len(face_visible_windows),
                "hand_visible_window_count": len(hand_visible_windows),
                "face_mesh_window_count": len(face_mesh_windows),
                "stable_screen_context_window_count": stable_screen_windows,
                "event_window_count": len(event_windows),
                "raw_paths_included": False,
                "frame_images_included": False,
                "exact_timestamps_included": False,
            },
        }

    def _build_event_windows(
        self,
        window_analyses: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        candidates = []
        for item in window_analyses:
            if item["device_interaction_status"] in {"detected", "possible"}:
                candidates.append(
                    self._event_window(
                        item,
                        trigger_type="device_interaction_evidence",
                        trigger_source="mediapipe_hand_motion_listener",
                        interaction_pattern=item["motion_pattern"],
                    )
                )
            if (
                item["device_interaction_status"] == "not_reliable"
                and item["device_region_motion_level"] in {"elevated", "high"}
            ):
                candidates.append(
                    self._event_window(
                        item,
                        trigger_type="device_region_motion_proxy",
                        trigger_source="device_region_motion_listener",
                        interaction_pattern=item["motion_pattern"],
                    )
                )
            if item["screen_gaze_proxy"]:
                candidates.append(
                    self._event_window(
                        item,
                        trigger_type="sustained_screen_oriented_proxy",
                        trigger_source="screen_orientation_proxy_listener",
                        interaction_pattern="face_and_screen_like_region_visible_with_limited_motion",
                    )
                )
            if item["blink_rate_change"] != "not_computed":
                candidates.append(
                    self._event_window(
                        item,
                        trigger_type="blink_rate_change",
                        trigger_source="facemesh_eye_openness_listener",
                        interaction_pattern="eye_openness_change_proxy",
                    )
                )
            if item["facial_au_codes"]:
                candidates.append(
                    self._event_window(
                        item,
                        trigger_type="facial_au_trend",
                        trigger_source="facemesh_action_proxy_listener",
                        interaction_pattern="facial_geometry_action_proxy",
                    )
                )
            if item["stable_screen_context"]:
                candidates.append(
                    self._event_window(
                        item,
                        trigger_type="stable_screen_context",
                        trigger_source="screen_context_listener",
                        interaction_pattern="low_motion_with_visible_screen_like_region",
                    )
                )
            if item["global_motion_level"] in {"high", "elevated"}:
                candidates.append(
                    self._event_window(
                        item,
                        trigger_type="motion_confounding",
                        trigger_source="motion_confounder_listener",
                        interaction_pattern="global_motion_may_confound_interaction",
                    )
                )
            if item["lighting_quality"] == "low" or item["blur_quality"] == "low":
                candidates.append(
                    self._event_window(
                        item,
                        trigger_type="quality_drop",
                        trigger_source="quality_gateway",
                        interaction_pattern="low_quality_window",
                    )
                )

        max_events = int(self.config.get("max_event_windows", 8))
        candidates = sorted(
            candidates,
            key=lambda event: (
                self._event_priority(event["trigger_type"]),
                event["window_order"],
            ),
        )
        deduped = []
        seen_windows = set()
        seen_events = set()
        for event in candidates:
            key = (event["window_order"], event["trigger_type"])
            if key in seen_events:
                continue
            if event["trigger_type"] != "quality_drop" and event["window_order"] in seen_windows:
                continue
            seen_events.add(key)
            seen_windows.add(event["window_order"])
            deduped.append(event)
        deduped = deduped[:max_events]
        deduped = sorted(deduped, key=lambda event: event["window_order"])
        for index, event in enumerate(deduped, start=1):
            event["window_id"] = f"Event_{index:02d}"
        return deduped

    def _event_window(
        self,
        item: dict[str, Any],
        *,
        trigger_type: str,
        trigger_source: str,
        interaction_pattern: str,
    ) -> dict[str, Any]:
        return {
            "window_id": "event_pending",
            "window_order": item["window_order"],
            "relative_position": item["relative_position"],
            "relative_time_period": item["relative_time_period"],
            "duration_bin": item["duration_bin"],
            "trigger_type": trigger_type,
            "trigger_label": self._display_trigger(trigger_type),
            "trigger_source": trigger_source,
            "proxy_evidence": {
                "interaction": self._interaction_narrative(
                    item,
                    trigger_type=trigger_type,
                    interaction_pattern=interaction_pattern,
                ),
                "posture": self._posture_narrative(item),
                "facial_cues": self._facial_cue_narrative(item),
                "action_units_detected": self._action_unit_narratives(item),
                "screen_orientation": self._screen_orientation_narrative(item),
                "motion_context": self._motion_context_narrative(item),
            },
            "quality_metrics": {
                "face_visibility_ratio": item["face_visibility_ratio"],
                "hand_visibility_ratio": item["hand_visibility_ratio"],
                "occlusion": item["occlusion_level"],
                "face_visibility_level": item["face_visibility_level"],
                "face_mesh_visibility_level": item["face_mesh_visibility_level"],
                "hand_visibility_level": item["hand_visibility_level"],
                "device_visibility_level": item["device_visibility_level"],
                "lighting_quality": self._display_level(item["lighting_quality"]),
                "occlusion_level": item["occlusion_level"],
                "event_confidence": item["event_confidence"],
            },
        }

    def _interaction_narrative(
        self,
        item: dict[str, Any],
        *,
        trigger_type: str,
        interaction_pattern: str,
    ) -> str:
        status = item.get("device_interaction_status")
        device_visible = item.get("device_visibility_level") in {
            "medium",
            "high",
            "very_high",
        }
        rapid_repetitive = interaction_pattern == "rapid_repetitive_local_motion"
        intermittent = interaction_pattern == "intermittent_local_motion"

        if status == "detected":
            if rapid_repetitive:
                return (
                    "Visible hand landmarks co-occurred with a visible device or "
                    "screen-like region, and lower-frame motion showed repeated rapid "
                    "bursts. This supports a repeated touch-input proxy; exact touch "
                    "type, coordinates, and screen content are not retained."
                )
            return (
                "Visible hand landmarks co-occurred with a visible device or "
                "screen-like region and local lower-frame motion. This supports a "
                "hand-device interaction proxy without retaining touch coordinates."
            )
        if status == "possible":
            if rapid_repetitive:
                return (
                    "Limited hand visibility, a visible device or screen-like region, "
                    "and repeated rapid lower-frame motion appeared together. This is "
                    "treated as possible repeated touch input, but hand evidence is "
                    "not strong enough for a direct interaction claim."
                )
            return (
                "Limited hand visibility and local motion near a visible device or "
                "screen-like region suggest possible device interaction, with reduced "
                "certainty because hand visibility is incomplete."
            )
        if status == "not_reliable" and device_visible:
            if rapid_repetitive:
                return (
                    "The device or screen-like region was visible and lower-frame "
                    "motion showed repeated rapid bursts, but hands were not visible "
                    "enough to confirm touch input. Treat this as device-region "
                    "activity, not confirmed hand-device interaction."
                )
            if intermittent:
                return (
                    "The device or screen-like region was visible with intermittent "
                    "lower-frame movement, but hand visibility was too weak to confirm "
                    "touch input."
                )
            return (
                "A device or screen-like region was visible, but the available hand "
                "and motion evidence does not support a reliable interaction claim."
            )
        if trigger_type == "stable_screen_context":
            return (
                "A device or screen-like region remained visible while global motion "
                "was low. This supports a stable viewing-context proxy rather than an "
                "active input proxy."
            )
        if trigger_type == "sustained_screen_oriented_proxy":
            return (
                "Face and device or screen-like region were visible in a relatively "
                "stable context, supporting a screen-oriented posture proxy. This is "
                "not precise eye tracking."
            )
        if trigger_type == "motion_confounding":
            return (
                "Broad frame motion was present, so local lower-frame activity may be "
                "mixed with posture or camera/context movement."
            )
        if trigger_type == "quality_drop":
            return (
                "This window was selected because visual quality dropped; behavioral "
                "interaction evidence from this window should be treated cautiously."
            )
        return (
            "No confirmed device interaction was observed in this window from the "
            "available hand, device, and local-motion proxies."
        )

    @staticmethod
    def _posture_narrative(item: dict[str, Any]) -> str:
        state = item.get("posture_state", "not_computed")
        if state == "close_to_screen":
            if item.get("global_motion_level") in {"elevated", "high"}:
                return (
                    "The face appeared large when visible, consistent with a "
                    "close-to-screen posture, but broader frame motion may affect "
                    "posture stability interpretation."
                )
            return (
                "The face appeared large when visible, consistent with a "
                "close-to-screen posture in this sampled window."
            )
        if state == "visible_not_close":
            return (
                "The face was visible without a strong close-to-screen proxy in this "
                "sampled window."
            )
        return (
            "Posture could not be reliably summarized because face visibility or "
            "face-size evidence was insufficient."
        )

    @staticmethod
    def _facial_cue_narrative(item: dict[str, Any]) -> str:
        face_level = item.get("face_mesh_visibility_level")
        eye_level = item.get("eye_openness_level")
        blink_change = item.get("blink_rate_change")
        if face_level not in {"low", "medium", "high", "very_high"}:
            return (
                "Facial cues were not reliable in this window because FaceMesh was "
                "not consistently available."
            )

        parts = []
        if eye_level == "typical":
            parts.append("sampled eye openness appeared typical")
        elif eye_level == "reduced":
            parts.append("sampled eye openness appeared reduced")
        elif eye_level == "low":
            parts.append("sampled eye openness appeared low")
        else:
            parts.append("eye-openness proxy was not stable enough to summarize")

        if blink_change == "elevated_eye_closure":
            parts.append("sparse samples included more eye-closure frames")
        elif blink_change == "reduced_eye_openness":
            parts.append("sparse samples showed reduced eye openness")
        elif blink_change == "stable":
            parts.append("no clear eye-closure change was selected")
        else:
            parts.append("blink-rate change was not computed")

        if item.get("facial_au_codes"):
            parts.append("facial geometry proxies were selected")
        else:
            parts.append("no dominant facial geometry action proxy was selected")
        return "; ".join(parts) + "."

    @staticmethod
    def _action_unit_narratives(item: dict[str, Any]) -> list[str]:
        label_map = {
            "AU4_proxy": "Brow-lowering geometry proxy (AU4-like)",
            "AU25_proxy": "Lips-part geometry proxy (AU25-like)",
        }
        narratives = []
        for proxy in item.get("facial_au_codes", []):
            code = str(proxy.get("au_code", "unknown_proxy"))
            label = label_map.get(code, str(proxy.get("label", code)))
            intensity = str(proxy.get("intensity", "unknown"))
            narratives.append(f"{label}, {intensity} intensity")
        return narratives

    @staticmethod
    def _screen_orientation_narrative(item: dict[str, Any]) -> str:
        gaze_state = item.get("gaze_state")
        if gaze_state == "sustained_screen_oriented_proxy":
            return (
                "Face and device or screen-like region were jointly visible in a "
                "stable enough context to support a screen-oriented posture proxy."
            )
        if gaze_state == "device_visible_but_face_or_motion_proxy_insufficient":
            return (
                "The device or screen-like region was visible, but face visibility or "
                "motion conditions were not sufficient for a reliable "
                "screen-orientation proxy."
            )
        return (
            "Screen orientation was not computed reliably for this window."
        )

    @staticmethod
    def _motion_context_narrative(item: dict[str, Any]) -> str:
        local = item.get("local_motion_level")
        global_motion = item.get("global_motion_level")
        burst_count = item.get("local_motion_burst_count", 0)
        if local == "high" and burst_count >= 2 and global_motion in {"elevated", "high"}:
            return (
                "Lower-frame motion had repeated bursts, while broad frame motion was "
                "also present; interaction interpretation is confounded by global "
                "movement."
            )
        if local == "high" and burst_count >= 2:
            return (
                "Lower-frame motion had repeated bursts concentrated near the visible "
                "device/screen-like region."
            )
        if local in {"elevated", "high"}:
            return (
                "Lower-frame motion increased near the visible device/screen-like "
                "region, but the sampled pattern was not enough to identify a "
                "specific touch type."
            )
        if global_motion in {"elevated", "high"}:
            return (
                "Broad frame motion was elevated, so local movement cues should be "
                "interpreted cautiously."
            )
        return (
            "No clear local motion burst was selected in this sampled window."
        )

    def _window_summary(
        self,
        *,
        window_order: int,
        relative_position: str,
        relative_time_period: str,
        duration_bin: str,
        frame_metrics: list[dict[str, Any]],
        motion_metrics: list[dict[str, Any]],
    ) -> dict[str, Any]:
        device_visible_frames = sum(1 for item in frame_metrics if item["device_visible"])
        face_visible_frames = sum(1 for item in frame_metrics if item["face_visible"])
        face_mesh_visible_frames = sum(
            1 for item in frame_metrics if item["face_mesh_visible"]
        )
        hand_visible_frames = sum(1 for item in frame_metrics if item["hand_visible"])
        multi_face_frames = sum(1 for item in frame_metrics if item["face_count"] > 1)
        largest_face_levels = [
            item["largest_face_size_level"]
            for item in frame_metrics
            if item["largest_face_size_level"] != "none"
        ]
        brightness_levels = [item["lighting_quality"] for item in frame_metrics]
        blur_levels = [item["blur_quality"] for item in frame_metrics]
        local_motion_values = [item["local_motion"] for item in motion_metrics]
        global_motion_values = [item["global_motion"] for item in motion_metrics]
        eye_openness_values = [
            item["eye_openness"]
            for item in frame_metrics
            if item["eye_openness"] is not None
        ]
        eye_closed_frames = sum(1 for item in frame_metrics if item["eye_closed_proxy"])
        facial_au_codes = self._dominant_facial_action_proxies(frame_metrics)
        local_motion_burst_count = sum(
            1 for value in local_motion_values if value >= self._local_motion_threshold()
        )
        global_motion_burst_count = sum(
            1 for value in global_motion_values if value >= self._global_motion_threshold()
        )
        local_motion_level = self._motion_level(
            max(local_motion_values or [0.0]),
            self._local_motion_threshold(),
        )
        global_motion_level = self._motion_level(
            max(global_motion_values or [0.0]),
            self._global_motion_threshold(),
        )
        device_visibility_level = self._ratio_bin(
            safe_ratio(device_visible_frames, len(frame_metrics))
        )
        face_visibility_ratio = safe_ratio(face_visible_frames, len(frame_metrics))
        face_visibility_level = self._ratio_bin(face_visibility_ratio)
        hand_visibility_ratio = safe_ratio(hand_visible_frames, len(frame_metrics))
        hand_visibility_level = self._ratio_bin(hand_visibility_ratio)
        face_mesh_visibility_level = self._ratio_bin(
            safe_ratio(face_mesh_visible_frames, len(frame_metrics))
        )
        device_region_motion_level = (
            local_motion_level
            if device_visibility_level in {"medium", "high", "very_high"}
            else "not_reliable"
        )
        device_interaction_status = self._device_interaction_status(
            hand_visibility_level=hand_visibility_level,
            device_visibility_level=device_visibility_level,
            local_motion_level=local_motion_level,
        )
        interaction_intensity = self._interaction_intensity(
            device_visibility_level=device_visibility_level,
            hand_visibility_level=hand_visibility_level,
            device_interaction_status=device_interaction_status,
            local_motion_level=local_motion_level,
            local_motion_burst_count=local_motion_burst_count,
        )
        hand_device_interaction_proxy = (
            device_interaction_status in {"detected", "possible"}
        )
        screen_gaze_proxy = (
            device_visibility_level in {"medium", "high", "very_high"}
            and face_visibility_level in {"medium", "high", "very_high"}
            and global_motion_level in {"none", "low", "elevated"}
        )
        stable_screen_context = (
            device_visibility_level in {"high", "very_high"}
            and global_motion_level in {"none", "low"}
        )
        occlusion_level = self._occlusion_level(face_visibility_level)
        posture_state = self._posture_state(largest_face_levels)
        blink_rate_change = self._window_blink_proxy(
            eye_openness_values,
            eye_closed_frames,
            len(frame_metrics),
        )
        return {
            "window_order": window_order,
            "relative_position": relative_position,
            "relative_time_period": relative_time_period,
            "duration_bin": duration_bin,
            "sampled_frame_count": len(frame_metrics),
            "device_visibility_level": device_visibility_level,
            "face_visibility_ratio": face_visibility_ratio,
            "face_visibility_level": face_visibility_level,
            "face_mesh_visibility_level": face_mesh_visibility_level,
            "hand_visibility_ratio": hand_visibility_ratio,
            "hand_visibility_level": hand_visibility_level,
            "lighting_quality": self._dominant_quality(brightness_levels),
            "blur_quality": self._dominant_quality(blur_levels),
            "local_motion_level": local_motion_level,
            "global_motion_level": global_motion_level,
            "motion_activity_level": self._dominant_motion_level(
                [local_motion_level, global_motion_level]
            ),
            "device_region_motion_level": device_region_motion_level,
            "motion_pattern": self._motion_pattern(
                local_motion_level,
                local_motion_burst_count,
            ),
            "local_motion_burst_count": local_motion_burst_count,
            "global_motion_burst_count": global_motion_burst_count,
            "interaction_intensity": interaction_intensity,
            "device_interaction_status": device_interaction_status,
            "interaction_evidence_basis": self._interaction_evidence_basis(
                device_interaction_status
            ),
            "hand_device_interaction_proxy": hand_device_interaction_proxy,
            "screen_gaze_proxy": screen_gaze_proxy,
            "stable_screen_context": stable_screen_context,
            "posture_state": posture_state,
            "gaze_state": self._gaze_state(screen_gaze_proxy, device_visibility_level),
            "blink_rate_change": blink_rate_change,
            "eye_openness_level": self._eye_openness_level(eye_openness_values),
            "facial_au_codes": facial_au_codes,
            "occlusion_level": occlusion_level,
            "multi_person_interference": multi_face_frames > 0,
            "event_confidence": self._event_confidence(
                device_visibility_level,
                hand_visibility_level,
                device_interaction_status,
                self._dominant_quality(brightness_levels),
                self._dominant_quality(blur_levels),
            ),
        }

    def _frame_metrics(
        self,
        cv2: Any,
        np: Any,
        visual_resources: dict[str, Any],
        frame: Any,
    ) -> dict[str, Any]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        face_cascade = visual_resources.get("face_cascade")
        face_count, largest_face_size_level = self._face_metrics(face_cascade, gray)
        screen_visible = self._screen_like_region_visible(cv2, gray)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_metrics = self._mediapipe_hand_metrics(
            visual_resources.get("mp"),
            visual_resources.get("hands"),
            rgb,
        )
        face_mesh_metrics = self._mediapipe_face_mesh_metrics(
            visual_resources.get("mp"),
            visual_resources.get("face_mesh"),
            rgb,
        )
        if face_mesh_metrics["face_count"]:
            face_count = face_mesh_metrics["face_count"]
            largest_face_size_level = face_mesh_metrics["largest_face_size_level"]
        return {
            "screen_visible": screen_visible,
            "device_visible": screen_visible,
            "face_visible": face_count > 0,
            "face_count": face_count,
            "largest_face_size_level": largest_face_size_level,
            "face_mesh_visible": face_mesh_metrics["face_count"] > 0,
            "eye_openness": face_mesh_metrics["eye_openness"],
            "eye_closed_proxy": face_mesh_metrics["eye_closed_proxy"],
            "facial_au_codes": face_mesh_metrics["facial_au_codes"],
            "hand_visible": hand_metrics["hand_count"] > 0,
            "hand_count": hand_metrics["hand_count"],
            "hand_region_level": hand_metrics["hand_region_level"],
            "lighting_quality": self._lighting_quality(brightness),
            "blur_quality": self._blur_quality(blur),
        }

    def _build_visual_resources(self, cv2: Any) -> dict[str, Any]:
        resources: dict[str, Any] = {
            "face_cascade": self._build_face_cascade(cv2),
            "hands": None,
            "face_mesh": None,
            "mp": None,
            "mediapipe_available": False,
        }
        try:
            import mediapipe as mp  # type: ignore[import-not-found]
            from mediapipe.tasks.python.core.base_options import BaseOptions  # type: ignore[import-not-found]
            from mediapipe.tasks.python.vision import (  # type: ignore[import-not-found]
                FaceLandmarker,
                FaceLandmarkerOptions,
                HandLandmarker,
                HandLandmarkerOptions,
                RunningMode,
            )
        except ImportError:
            return resources

        resources["mediapipe_available"] = True
        resources["mp"] = mp
        min_detection = float(self.config.get("mediapipe_min_detection_confidence", 0.5))
        if self.config.get("enable_mediapipe_hands", True):
            hand_model = Path(
                self.config.get(
                    "mediapipe_hand_model",
                    "models/mediapipe/hand_landmarker.task",
                )
            )
            if hand_model.exists():
                resources["hands"] = HandLandmarker.create_from_options(
                    HandLandmarkerOptions(
                        base_options=BaseOptions(model_asset_path=hand_model.as_posix()),
                        running_mode=RunningMode.IMAGE,
                        num_hands=2,
                        min_hand_detection_confidence=min_detection,
                    )
                )
        if self.config.get("enable_mediapipe_face_mesh", True):
            face_model = Path(
                self.config.get(
                    "mediapipe_face_model",
                    "models/mediapipe/face_landmarker.task",
                )
            )
            if face_model.exists():
                resources["face_mesh"] = FaceLandmarker.create_from_options(
                    FaceLandmarkerOptions(
                        base_options=BaseOptions(model_asset_path=face_model.as_posix()),
                        running_mode=RunningMode.IMAGE,
                        num_faces=2,
                        min_face_detection_confidence=min_detection,
                        output_face_blendshapes=False,
                        output_facial_transformation_matrixes=False,
                    )
                )
        return resources

    @staticmethod
    def _close_visual_resources(resources: dict[str, Any]) -> None:
        for key in ("hands", "face_mesh"):
            resource = resources.get(key)
            if resource is not None:
                resource.close()

    @staticmethod
    def _mediapipe_hand_metrics(mp: Any, hands: Any, rgb_frame: Any) -> dict[str, Any]:
        if mp is None or hands is None:
            return {
                "hand_count": 0,
                "hand_region_level": "not_computed",
            }
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = hands.detect(image)
        landmarks = results.hand_landmarks or []
        if not landmarks:
            return {
                "hand_count": 0,
                "hand_region_level": "none",
            }
        mean_y_values = [
            sum(point.y for point in hand) / max(len(hand), 1)
            for hand in landmarks
        ]
        mean_y = sum(mean_y_values) / len(mean_y_values)
        if mean_y >= 0.66:
            region = "lower_frame"
        elif mean_y >= 0.33:
            region = "middle_frame"
        else:
            region = "upper_frame"
        return {
            "hand_count": len(landmarks),
            "hand_region_level": region,
        }

    def _mediapipe_face_mesh_metrics(
        self,
        mp: Any,
        face_mesh: Any,
        rgb_frame: Any,
    ) -> dict[str, Any]:
        if mp is None or face_mesh is None:
            return self._empty_face_mesh_metrics("not_computed")
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = face_mesh.detect(image)
        faces = results.face_landmarks or []
        if not faces:
            return self._empty_face_mesh_metrics("none")
        primary = faces[0]
        eye_openness = self._eye_openness(primary)
        return {
            "face_count": len(faces),
            "largest_face_size_level": self._face_mesh_size_level(primary),
            "eye_openness": eye_openness,
            "eye_closed_proxy": (
                eye_openness is not None
                and eye_openness < float(self.config.get("eye_closed_ear_threshold", 0.18))
            ),
            "facial_au_codes": self._facial_action_proxies(primary),
        }

    @staticmethod
    def _empty_face_mesh_metrics(status: str) -> dict[str, Any]:
        return {
            "face_count": 0,
            "largest_face_size_level": status,
            "eye_openness": None,
            "eye_closed_proxy": False,
            "facial_au_codes": [],
        }

    def _eye_openness(self, landmarks: Any) -> float | None:
        left = self._eye_aspect_ratio(landmarks, (33, 160, 158, 133, 153, 144))
        right = self._eye_aspect_ratio(landmarks, (362, 385, 387, 263, 373, 380))
        values = [value for value in (left, right) if value is not None]
        if not values:
            return None
        return round(sum(values) / len(values), 3)

    @staticmethod
    def _eye_aspect_ratio(landmarks: Any, indices: tuple[int, int, int, int, int, int]) -> float | None:
        try:
            p1, p2, p3, p4, p5, p6 = [landmarks[index] for index in indices]
        except IndexError:
            return None
        horizontal = PriVTEPreprocessorV0Extractor._point_distance(p1, p4)
        if horizontal <= 0:
            return None
        vertical = (
            PriVTEPreprocessorV0Extractor._point_distance(p2, p6)
            + PriVTEPreprocessorV0Extractor._point_distance(p3, p5)
        )
        return vertical / (2.0 * horizontal)

    @staticmethod
    def _point_distance(first: Any, second: Any) -> float:
        return math.sqrt(
            (float(first.x) - float(second.x)) ** 2
            + (float(first.y) - float(second.y)) ** 2
            + (float(getattr(first, "z", 0.0)) - float(getattr(second, "z", 0.0))) ** 2
        )

    def _facial_action_proxies(self, landmarks: Any) -> list[dict[str, str]]:
        face_height = self._face_mesh_height(landmarks)
        if face_height <= 0:
            return []

        proxies = []
        brow_eye_gap = self._brow_eye_gap(landmarks) / face_height
        if brow_eye_gap < 0.045:
            proxies.append(
                {
                    "au_code": "AU4_proxy",
                    "label": "brow_lowerer_geometry_proxy",
                    "intensity": "medium",
                }
            )
        elif brow_eye_gap < 0.06:
            proxies.append(
                {
                    "au_code": "AU4_proxy",
                    "label": "brow_lowerer_geometry_proxy",
                    "intensity": "low",
                }
            )

        mouth_open = self._point_distance(landmarks[13], landmarks[14]) / face_height
        if mouth_open > 0.055:
            proxies.append(
                {
                    "au_code": "AU25_proxy",
                    "label": "lips_part_geometry_proxy",
                    "intensity": "medium",
                }
            )
        elif mouth_open > 0.04:
            proxies.append(
                {
                    "au_code": "AU25_proxy",
                    "label": "lips_part_geometry_proxy",
                    "intensity": "low",
                }
            )
        return proxies

    @staticmethod
    def _face_mesh_height(landmarks: Any) -> float:
        y_values = [float(point.y) for point in landmarks]
        if not y_values:
            return 0.0
        return max(y_values) - min(y_values)

    @staticmethod
    def _brow_eye_gap(landmarks: Any) -> float:
        pairs = ((105, 159), (334, 386))
        gaps = [
            abs(float(landmarks[brow].y) - float(landmarks[eye].y))
            for brow, eye in pairs
        ]
        return sum(gaps) / len(gaps)

    def _face_mesh_size_level(self, landmarks: Any) -> str:
        x_values = [float(point.x) for point in landmarks]
        y_values = [float(point.y) for point in landmarks]
        if not x_values or not y_values:
            return "none"
        area_ratio = (max(x_values) - min(x_values)) * (max(y_values) - min(y_values))
        return self._face_size_level(area_ratio)

    def _motion_metrics(self, cv2: Any, np: Any, previous: Any, current: Any) -> dict[str, Any]:
        previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(previous_gray, current_gray)
        global_motion = float(np.mean(diff)) / 255.0
        height = diff.shape[0]
        lower_half = diff[height // 2 :, :]
        local_motion = float(np.mean(lower_half)) / 255.0
        return {
            "global_motion": global_motion,
            "local_motion": local_motion,
        }

    def _screen_like_region_visible(self, cv2: Any, gray: Any) -> bool:
        _, thresholded = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresholded,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        frame_area = gray.shape[0] * gray.shape[1]
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            area_ratio = (width * height) / frame_area
            aspect = width / max(height, 1)
            if 0.02 <= area_ratio <= 0.65 and 0.4 <= aspect <= 4.0:
                return True
        return False

    @staticmethod
    def _build_face_cascade(cv2: Any) -> Any:
        cascade_path = getattr(cv2, "data", None)
        if cascade_path is None:
            return None
        path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if not path.exists():
            return None
        cascade = cv2.CascadeClassifier(str(path))
        return cascade if not cascade.empty() else None

    def _face_metrics(self, face_cascade: Any, gray: Any) -> tuple[int, str]:
        if face_cascade is None:
            return 0, "none"
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(24, 24),
        )
        face_count = len(faces)
        if face_count == 0:
            return 0, "none"
        frame_area = gray.shape[0] * gray.shape[1]
        largest_area_ratio = max((width * height) / frame_area for _, _, width, height in faces)
        return face_count, self._face_size_level(largest_area_ratio)

    @staticmethod
    def _face_size_level(area_ratio: float) -> str:
        if area_ratio >= 0.18:
            return "very_close"
        if area_ratio >= 0.08:
            return "close"
        if area_ratio >= 0.03:
            return "visible"
        return "small"

    @staticmethod
    def _duration_for_index(timeline: dict[str, Any], index: int) -> float | None:
        durations = timeline.get("durations_sec", [])
        if 0 <= index < len(durations):
            return durations[index]
        return None

    def _resolve_video_path(self, path_value: str | Path) -> Path:
        """Resolve local-only fallback paths without emitting them as evidence."""

        path = Path(path_value)
        if self._path_exists(path):
            return path

        fallbacks = self.config.get("path_fallbacks", {})
        if not isinstance(fallbacks, dict):
            return path

        path_text = path.as_posix()
        for source_prefix, fallback_prefix in fallbacks.items():
            source = str(source_prefix).strip("/")
            fallback = str(fallback_prefix).rstrip("/")
            if path_text == source or path_text.startswith(source + "/"):
                suffix = path_text[len(source) :].lstrip("/")
                candidate = Path(fallback) / suffix
                if self._path_exists(candidate):
                    return candidate
        return path

    @staticmethod
    def _path_exists(path: Path) -> bool:
        try:
            return path.exists()
        except OSError:
            return False

    def _relative_time_period(self, timeline: dict[str, Any], index: int) -> str:
        durations = timeline.get("durations_sec", [])
        starts = timeline.get("starts_sec", [])
        if not (0 <= index < len(durations)) or not (0 <= index < len(starts)):
            return "relative_period_unknown"
        duration = durations[index]
        if duration is None:
            return self._relative_position(index, len(durations))
        start = self._round_relative_seconds(float(starts[index]))
        end = self._round_relative_seconds(float(starts[index]) + float(duration))
        return f"{self._format_mmss(start)} - {self._format_mmss(end)}"

    @staticmethod
    def _round_relative_seconds(value: float) -> int:
        return int(round(value / 30.0) * 30)

    @staticmethod
    def _format_mmss(seconds: int) -> str:
        minutes = max(0, seconds) // 60
        remain = max(0, seconds) % 60
        return f"{minutes:02d}:{remain:02d}"

    @staticmethod
    def _event_priority(trigger_type: str) -> int:
        priority = {
            "device_interaction_evidence": 0,
            "device_region_motion_proxy": 1,
            "high_interaction_intensity": 1,
            "sustained_screen_oriented_proxy": 2,
            "blink_rate_change": 3,
            "facial_au_trend": 4,
            "stable_screen_context": 5,
            "quality_drop": 6,
            "motion_confounding": 7,
        }
        return priority.get(trigger_type, 9)

    @staticmethod
    def _display_trigger(trigger_type: str) -> str:
        return "_".join(part.capitalize() for part in trigger_type.split("_"))

    @staticmethod
    def _display_level(level: str | None) -> str:
        if level is None:
            return "not_computed"
        display = {
            "very_high": "Very high",
            "high": "High",
            "elevated": "Elevated",
            "medium": "Medium",
            "low": "Low",
            "very_low": "Very low",
            "none": "No clear",
            "sufficient": "Sufficient",
            "unknown": "Unknown",
            "not_computed": "not_computed",
        }
        return display.get(str(level), str(level))

    def _interaction_description(
        self,
        item: dict[str, Any],
        interaction_pattern: str,
    ) -> str:
        level = self._display_level(item["interaction_intensity"])
        if interaction_pattern == "device_region_or_lower_frame_motion_proxy":
            return (
                f"{level} lower-frame motion near visible device/screen-like region; "
                "interpreted only as a hand-device interaction proxy"
            )
        if interaction_pattern == "hand_landmark_visibility_near_device_motion_proxy":
            return (
                f"{level} interaction proxy with hand landmark visibility and "
                "near-device motion evidence"
            )
        if interaction_pattern == "low_motion_with_visible_screen_like_region":
            return (
                f"{level} interaction; device/screen-like region remains visible "
                "with low global motion"
            )
        if interaction_pattern == "eye_openness_change_proxy":
            return (
                f"{level} interaction context with sparse-frame eye openness change proxy"
            )
        if interaction_pattern == "facial_geometry_action_proxy":
            return (
                f"{level} interaction context with facial geometry action-unit proxy"
            )
        if interaction_pattern == "global_motion_may_confound_interaction":
            return (
                f"{level} interaction proxy, but high global motion may confound "
                "local interaction interpretation"
            )
        return f"{level} interaction proxy"

    @staticmethod
    def _posture_description(item: dict[str, Any]) -> str:
        state = item.get("posture_state", "not_computed")
        if state == "close_to_screen":
            return "Face appears large when visible; proxy suggests close-to-screen posture"
        if state == "visible_not_close":
            return "Face is visible without strong close-to-screen proxy"
        return "Posture proxy not reliable in this window"

    @staticmethod
    def _facial_cue_description(item: dict[str, Any]) -> str:
        if item.get("facial_au_codes"):
            return "FaceMesh geometry produced facial action-unit proxy codes"
        if item.get("blink_rate_change") != "not_computed":
            return "FaceMesh eye openness proxy is available for this window"
        if item.get("face_visibility_level") in {"medium", "high", "very_high"}:
            return "Face visible, but no stable blink or facial action proxy was selected"
        return "Facial cues not reliable because face visibility is low or unavailable"

    @staticmethod
    def _gaze_state(screen_gaze_proxy: bool, device_visibility_level: str) -> str:
        if screen_gaze_proxy:
            return "sustained_screen_oriented_proxy"
        if device_visibility_level in {"medium", "high", "very_high"}:
            return "device_visible_but_face_or_motion_proxy_insufficient"
        return "not_computed"

    @staticmethod
    def _occlusion_level(face_visibility_level: str) -> str:
        if face_visibility_level in {"high", "very_high"}:
            return "none_detected_by_face_visibility_proxy"
        if face_visibility_level == "medium":
            return "possible_partial_occlusion_or_pose_change"
        if face_visibility_level == "low":
            return "limited_face_visibility"
        return "not_computed"

    @staticmethod
    def _posture_state(face_size_levels: list[str]) -> str:
        if any(level in {"very_close", "close"} for level in face_size_levels):
            return "close_to_screen"
        if face_size_levels:
            return "visible_not_close"
        return "not_computed"

    @staticmethod
    def _eye_openness_level(values: list[float]) -> str:
        if not values:
            return "not_computed"
        mean_value = sum(values) / len(values)
        if mean_value < 0.18:
            return "low"
        if mean_value < 0.23:
            return "reduced"
        return "typical"

    def _window_blink_proxy(
        self,
        eye_openness_values: list[float],
        eye_closed_frames: int,
        sampled_frame_count: int,
    ) -> str:
        if not eye_openness_values:
            return "not_computed"
        closed_ratio = safe_ratio(eye_closed_frames, sampled_frame_count)
        eye_level = self._eye_openness_level(eye_openness_values)
        if closed_ratio >= 0.35:
            return "elevated_eye_closure"
        if eye_level == "low":
            return "reduced_eye_openness"
        return "stable"

    @staticmethod
    def _dominant_facial_action_proxies(
        frame_metrics: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        counts: Counter[tuple[str, str, str]] = Counter()
        for frame in frame_metrics:
            for item in frame.get("facial_au_codes", []):
                counts[
                    (
                        str(item.get("au_code")),
                        str(item.get("label")),
                        str(item.get("intensity")),
                    )
                ] += 1
        if not counts:
            return []
        threshold = max(1, len(frame_metrics) // 3)
        selected = []
        for (au_code, label, intensity), count in counts.most_common(3):
            if count >= threshold:
                selected.append(
                    {
                        "au_code": au_code,
                        "label": label,
                        "intensity": intensity,
                    }
                )
        return selected

    @staticmethod
    def _blink_proxy_summary(windows: list[dict[str, Any]]) -> dict[str, str]:
        values = [
            item.get("blink_rate_change", "not_computed")
            for item in windows
            if item.get("blink_rate_change") != "not_computed"
        ]
        if not values:
            return {
                "blink_rate_level": "not_computed",
                "blink_rate_trend": "not_computed",
                "eye_closure_proxy_level": "not_computed",
            }
        elevated = sum(1 for value in values if value == "elevated_eye_closure")
        reduced = sum(1 for value in values if value == "reduced_eye_openness")
        if elevated >= max(2, len(values) // 3):
            trend = "frequent_eye_closure"
        elif reduced >= max(2, len(values) // 3):
            trend = "frequent_reduced_eye_openness"
        else:
            trend = "mostly_stable"
        return {
            "blink_rate_level": "sparse_frame_proxy_available",
            "blink_rate_trend": trend,
            "eye_closure_proxy_level": PriVTEPreprocessorV0Extractor._count_level(
                elevated + reduced
            ),
        }

    @staticmethod
    def _facial_action_proxy_counts(windows: list[dict[str, Any]]) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for item in windows:
            for proxy in item.get("facial_au_codes", []):
                counts[str(proxy.get("au_code"))] += 1
        return dict(counts)

    @staticmethod
    def _screen_orientation_proxy_summary(
        windows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not windows:
            return {
                "ratio": None,
                "ratio_bin": "not_computed",
                "status": "not_computed",
                "quality": "not_computed",
            }

        observable = [
            item
            for item in windows
            if item.get("device_visibility_level") in {"medium", "high", "very_high"}
            and (
                item.get("face_mesh_visibility_level") in {"medium", "high", "very_high"}
                or item.get("face_visibility_level") in {"medium", "high", "very_high"}
            )
        ]
        if not observable:
            return {
                "ratio": None,
                "ratio_bin": "not_reliably_estimable",
                "status": "insufficient_face_or_device_visibility",
                "quality": "low",
            }

        confounded = sum(
            1 for item in observable if item.get("global_motion_level") in {"elevated", "high"}
        )
        if safe_ratio(confounded, len(observable)) >= 0.5:
            return {
                "ratio": None,
                "ratio_bin": "motion_confounded",
                "status": "motion_confounded",
                "quality": "low",
            }

        proxy_count = sum(1 for item in observable if item.get("screen_gaze_proxy"))
        ratio = safe_ratio(proxy_count, len(observable))
        quality = "medium" if len(observable) >= max(2, len(windows) // 3) else "low"
        return {
            "ratio": ratio,
            "ratio_bin": PriVTEPreprocessorV0Extractor._ratio_bin(ratio),
            "status": "estimated_from_visible_face_and_device",
            "quality": quality,
        }

    @staticmethod
    def _device_interaction_status(
        *,
        hand_visibility_level: str,
        device_visibility_level: str,
        local_motion_level: str,
    ) -> str:
        device_visible = device_visibility_level in {"medium", "high", "very_high"}
        hand_visible = hand_visibility_level in {"low", "medium", "high", "very_high"}
        motion_active = local_motion_level in {"elevated", "high"}
        if not device_visible:
            return "not_observed"
        if hand_visibility_level in {"medium", "high", "very_high"} and motion_active:
            return "detected"
        if hand_visible and motion_active:
            return "possible"
        if motion_active:
            return "not_reliable"
        return "not_observed"

    @staticmethod
    def _interaction_evidence_basis(device_interaction_status: str) -> str:
        if device_interaction_status == "detected":
            return "hand_landmarks_device_visibility_and_local_motion"
        if device_interaction_status == "possible":
            return "limited_hand_visibility_device_visibility_and_local_motion"
        if device_interaction_status == "not_reliable":
            return "device_region_motion_without_reliable_hand_visibility"
        return "no_clear_hand_device_evidence"

    @staticmethod
    def _motion_pattern(local_motion_level: str, local_motion_burst_count: int) -> str:
        if local_motion_level == "high" and local_motion_burst_count >= 2:
            return "rapid_repetitive_local_motion"
        if local_motion_level in {"elevated", "high"}:
            return "intermittent_local_motion"
        if local_motion_level == "low":
            return "low_local_motion"
        return "no_clear_local_motion"

    @staticmethod
    def _max_continuous_proxy_minutes(
        windows: list[dict[str, Any]],
        screen_orientation_status: str,
    ) -> int | None:
        if not windows or screen_orientation_status != "estimated_from_visible_face_and_device":
            return None
        longest = 0
        current = 0
        for item in sorted(windows, key=lambda value: value["window_order"]):
            if item.get("screen_gaze_proxy"):
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        if longest == 0:
            return 0
        # Coverage windows are sparse; report an approximate 5-minute-granularity proxy.
        return longest * 5

    @staticmethod
    def _continuous_duration_bin(minutes: int | None) -> str:
        if minutes is None:
            return "not_computed"
        if minutes <= 0:
            return "none"
        if minutes < 5:
            return "short"
        if minutes < 15:
            return "medium"
        return "long"

    @staticmethod
    def _posture_trend(windows: list[dict[str, Any]]) -> str:
        states = [item.get("posture_state") for item in windows]
        if not states or all(state == "not_computed" for state in states):
            return "not_computed"
        close_count = sum(1 for state in states if state == "close_to_screen")
        if close_count >= max(2, len(states) // 2):
            return "frequent_close_to_screen"
        return "mostly_visible_not_close"

    @staticmethod
    def _gaze_estimation_quality(screen_orientation_summary: dict[str, Any]) -> str:
        status = screen_orientation_summary.get("status")
        quality = screen_orientation_summary.get("quality")
        if status == "estimated_from_visible_face_and_device":
            return f"{quality}_orientation_proxy"
        if status in {"motion_confounded", "insufficient_face_or_device_visibility"}:
            return f"not_reliably_estimable_{status}"
        return "not_computed"

    @staticmethod
    def _multi_person_level(count: int, total: int) -> str:
        ratio = safe_ratio(count, total)
        if ratio >= 0.5:
            return "frequent_possible_interference"
        if ratio > 0:
            return "occasional_possible_interference"
        return "none_detected"

    def _sample_frame_indices(
        self,
        frame_count: int,
        samples_per_window: int,
    ) -> list[int | None]:
        if frame_count <= 0:
            return [None] * max(samples_per_window, 1)
        if samples_per_window <= 1:
            return [frame_count // 2]
        start = max(0, int(frame_count * 0.1))
        end = max(start + 1, int(frame_count * 0.9))
        step = (end - start) / (samples_per_window - 1)
        return [min(frame_count - 1, round(start + index * step)) for index in range(samples_per_window)]

    def _resize_frame(self, cv2: Any, frame: Any) -> Any:
        analysis_width = int(self.config.get("analysis_width", 320))
        height, width = frame.shape[:2]
        if width <= analysis_width:
            return frame
        scale = analysis_width / width
        new_size = (analysis_width, max(1, int(height * scale)))
        return cv2.resize(frame, new_size)

    def _empty_window_analysis(
        self,
        window_order: int,
        relative_position: str,
        relative_time_period: str,
        duration_bin: str,
    ) -> dict[str, Any]:
        return {
            "window_order": window_order,
            "relative_position": relative_position,
            "relative_time_period": relative_time_period,
            "duration_bin": duration_bin,
            "sampled_frame_count": 0,
            "device_visibility_level": "none",
            "face_visibility_ratio": 0.0,
            "face_visibility_level": "none",
            "face_mesh_visibility_level": "none",
            "hand_visibility_ratio": 0.0,
            "hand_visibility_level": "none",
            "lighting_quality": "unknown",
            "blur_quality": "unknown",
            "local_motion_level": "none",
            "global_motion_level": "none",
            "motion_activity_level": "none",
            "device_region_motion_level": "not_reliable",
            "motion_pattern": "no_clear_local_motion",
            "local_motion_burst_count": 0,
            "global_motion_burst_count": 0,
            "interaction_intensity": "none",
            "device_interaction_status": "not_observed",
            "interaction_evidence_basis": "no_clear_hand_device_evidence",
            "hand_device_interaction_proxy": False,
            "screen_gaze_proxy": False,
            "stable_screen_context": False,
            "posture_state": "not_computed",
            "gaze_state": "not_computed",
            "blink_rate_change": "not_computed",
            "eye_openness_level": "not_computed",
            "facial_au_codes": [],
            "occlusion_level": "not_computed",
            "multi_person_interference": False,
            "event_confidence": "low",
        }

    @staticmethod
    def _empty_frame_summary(status: str, limitations: list[str]) -> dict[str, Any]:
        return {
            "status": status,
            "sampled_frame_count": 0,
            "global_features": {},
            "event_windows": [],
            "quality_summary": {},
            "limitations": limitations,
            "missing_information": ["frame_level_visual_proxy_features"],
            "frame_analysis_summary": {
                "analyzed_window_count": 0,
                "sampled_frame_count": 0,
                "device_visible_window_count": 0,
                "stable_screen_context_window_count": 0,
                "event_window_count": 0,
                "raw_paths_included": False,
                "frame_images_included": False,
                "exact_timestamps_included": False,
            },
        }

    @staticmethod
    def _select_indexed_entries(
        entries: list[dict[str, Any]],
        max_count: int,
    ) -> list[tuple[int, dict[str, Any]]]:
        if max_count <= 0 or len(entries) <= max_count:
            return list(enumerate(entries))
        if max_count == 1:
            return [(0, entries[0])]
        step = (len(entries) - 1) / (max_count - 1)
        indices = [round(index * step) for index in range(max_count)]
        return [(index, entries[index]) for index in indices]

    @staticmethod
    def _relative_position(index: int, total_count: int) -> str:
        if total_count <= 1:
            return "global"
        ratio = index / (total_count - 1)
        if ratio < 1 / 3:
            return "early"
        if ratio < 2 / 3:
            return "middle"
        return "late"

    def _local_motion_threshold(self) -> float:
        return float(self.config.get("local_motion_threshold", 0.035))

    def _global_motion_threshold(self) -> float:
        return float(self.config.get("global_motion_threshold", 0.05))

    @staticmethod
    def _motion_level(value: float, threshold: float) -> str:
        if value >= threshold * 2.0:
            return "high"
        if value >= threshold:
            return "elevated"
        if value > 0:
            return "low"
        return "none"

    @staticmethod
    def _interaction_intensity(
        *,
        device_visibility_level: str,
        hand_visibility_level: str,
        device_interaction_status: str,
        local_motion_level: str,
        local_motion_burst_count: int,
    ) -> str:
        if device_interaction_status not in {"detected", "possible"}:
            return "none"
        if hand_visibility_level in {"medium", "high", "very_high"} and (
            local_motion_level == "high" or local_motion_burst_count >= 2
        ):
            return "high"
        if hand_visibility_level in {"low", "medium", "high", "very_high"} and (
            local_motion_level in {"elevated", "high"} or local_motion_burst_count >= 1
        ):
            return "elevated"
        if device_visibility_level in {"high", "very_high"} and (
            local_motion_level == "high" or local_motion_burst_count >= 2
        ):
            return "high"
        if device_visibility_level in {"medium", "high", "very_high"} and (
            local_motion_level in {"elevated", "high"} or local_motion_burst_count >= 1
        ):
            return "elevated"
        if local_motion_level in {"low", "elevated"}:
            return "low"
        return "none"

    @staticmethod
    def _dominant_intensity(levels: list[str]) -> str:
        for level in ("high", "elevated", "medium", "low"):
            if level in levels:
                return level
        return "none"

    @staticmethod
    def _dominant_motion_level(levels: list[str]) -> str:
        for level in ("high", "elevated", "medium", "low"):
            if level in levels:
                return level
        return "none"

    @staticmethod
    def _count_level(count: int) -> str:
        if count >= 5:
            return "high"
        if count >= 2:
            return "medium"
        if count == 1:
            return "low"
        return "none"

    @staticmethod
    def _lighting_quality(brightness: float) -> str:
        if 45 <= brightness <= 220:
            return "sufficient"
        if 25 <= brightness <= 240:
            return "low"
        return "very_low"

    @staticmethod
    def _blur_quality(blur_variance: float) -> str:
        if blur_variance >= 80:
            return "sufficient"
        if blur_variance >= 30:
            return "low"
        return "very_low"

    @staticmethod
    def _dominant_quality(levels: list[str]) -> str:
        if not levels:
            return "unknown"
        counts = Counter(levels)
        return counts.most_common(1)[0][0]

    @staticmethod
    def _event_confidence(
        device_visibility_level: str,
        hand_visibility_level: str,
        device_interaction_status: str,
        lighting_quality: str,
        blur_quality: str,
    ) -> str:
        if (
            device_visibility_level in {"high", "very_high"}
            and lighting_quality == "sufficient"
            and blur_quality == "sufficient"
        ):
            base = "high"
        elif lighting_quality in {"sufficient", "low"} and blur_quality in {
            "sufficient",
            "low",
        }:
            base = "medium"
        else:
            base = "low"
        if device_interaction_status == "not_reliable":
            return "medium" if base == "high" else base
        if device_interaction_status == "possible" and hand_visibility_level == "low":
            return "medium" if base == "high" else base
        return base

    @staticmethod
    def _ratio_bin(value: float) -> str:
        if value >= 0.9:
            return "very_high"
        if value >= 0.75:
            return "high"
        if value >= 0.5:
            return "medium"
        if value > 0:
            return "low"
        return "none"

    @staticmethod
    def _duration_minutes_bin(duration_sec: float) -> str:
        if duration_sec <= 0:
            return "unknown"
        minutes = duration_sec / 60
        if minutes < 5:
            return "<5min"
        if minutes < 10:
            return "5-10min"
        if minutes < 20:
            return "10-20min"
        if minutes < 30:
            return "20-30min"
        if minutes < 45:
            return "30-45min"
        return ">=45min"

    @staticmethod
    def _coarse_minutes(duration_sec: float) -> int | None:
        if duration_sec <= 0:
            return None
        minutes = duration_sec / 60
        return int(round(minutes / 5.0) * 5)

    @staticmethod
    def _data_sufficiency(
        *,
        total_video_files: int,
        readable_ratio: float,
        readable_count: int,
    ) -> str:
        if total_video_files == 0 or readable_count == 0:
            return "insufficient_no_readable_video"
        if readable_ratio >= 0.8:
            return "adequate_for_frame_proxy_analysis"
        if readable_ratio >= 0.5:
            return "partial_for_frame_proxy_analysis"
        return "low_for_frame_proxy_analysis"
