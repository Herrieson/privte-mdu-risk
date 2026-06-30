"""PriVTE preprocessor v2.

V2 keeps the calibrated V1 visual proxies and adds a stateful evidence layer:
per-window symbolic states, short behavior episodes, temporal trend summaries,
and separated positive / weak / counter-evidence facts. The goal is to give the
text-only model richer evidence structure without releasing raw media, raw
coordinates, OCR, ASR, app names, or exact timestamps.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .privte_preprocessor_v1 import PriVTEPreprocessorV1Extractor


class PriVTEPreprocessorV2Extractor(PriVTEPreprocessorV1Extractor):
    """Stateful multi-component evidence layer over V1 frame proxies."""

    name = "privte_preprocessor_v2"
    version = "v2"

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
        summary = super()._window_summary(
            window_order=window_order,
            relative_position=relative_position,
            relative_time_period=relative_time_period,
            duration_bin=duration_bin,
            frame_metrics=frame_metrics,
            motion_metrics=motion_metrics,
        )
        return self._calibrate_window_summary_v2(
            summary=summary,
            frame_metrics=frame_metrics,
            motion_metrics=motion_metrics,
        )

    def _calibrate_window_summary_v2(
        self,
        *,
        summary: dict[str, Any],
        frame_metrics: list[dict[str, Any]],
        motion_metrics: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not frame_metrics:
            return summary

        raw_local_motion_level = summary.get("local_motion_level")
        raw_global_motion_level = summary.get("global_motion_level")
        raw_local_bursts = int(summary.get("local_motion_burst_count", 0) or 0)
        raw_global_bursts = int(summary.get("global_motion_burst_count", 0) or 0)

        local_values = [float(item.get("local_motion", 0.0)) for item in motion_metrics]
        global_values = [float(item.get("global_motion", 0.0)) for item in motion_metrics]
        localized_values = [
            max(local - global_value * 0.6, 0.0)
            for local, global_value in zip(local_values, global_values)
        ]
        localized_peak = max(localized_values or [0.0])
        localized_threshold = float(
            self.config.get("localized_motion_threshold", self._local_motion_threshold() * 0.45)
        )
        localized_bursts = sum(
            1 for value in localized_values if value >= localized_threshold
        )
        localized_motion_level = self._motion_level(localized_peak, localized_threshold)
        motion_dominance = self._motion_dominance(
            raw_local_bursts=raw_local_bursts,
            raw_global_bursts=raw_global_bursts,
            localized_bursts=localized_bursts,
            localized_motion_level=localized_motion_level,
            raw_global_motion_level=str(raw_global_motion_level),
        )
        effective_motion_confounding_level = self._effective_motion_confounding_level(
            motion_dominance
        )

        hand_region_counts = Counter(
            str(item.get("hand_region_level", "none")) for item in frame_metrics
        )
        lower_or_middle_hands = sum(
            1
            for item in frame_metrics
            if item.get("hand_region_level") in {"lower_frame", "middle_frame"}
        )
        lower_hands = sum(
            1 for item in frame_metrics if item.get("hand_region_level") == "lower_frame"
        )
        hand_device_proximity_ratio = lower_or_middle_hands / max(len(frame_metrics), 1)
        hand_device_proximity_level = self._ratio_bin(hand_device_proximity_ratio)
        device_interaction_status = self._device_interaction_status_v2(
            summary=summary,
            localized_motion_level=localized_motion_level,
            localized_bursts=localized_bursts,
            lower_or_middle_hands=lower_or_middle_hands,
            lower_hands=lower_hands,
        )

        summary["raw_local_motion_level"] = raw_local_motion_level
        summary["raw_global_motion_level"] = raw_global_motion_level
        summary["raw_local_motion_burst_count"] = raw_local_bursts
        summary["raw_global_motion_burst_count"] = raw_global_bursts
        summary["v2_localized_motion_level"] = localized_motion_level
        summary["v2_localized_motion_burst_count"] = localized_bursts
        summary["v2_motion_dominance"] = motion_dominance
        summary["v2_effective_motion_confounding_level"] = (
            effective_motion_confounding_level
        )
        summary["hand_region_counts"] = dict(hand_region_counts)
        summary["hand_device_proximity_level"] = hand_device_proximity_level
        summary["hand_device_proximity_ratio_bin"] = hand_device_proximity_level
        summary["device_interaction_status"] = device_interaction_status
        summary["interaction_evidence_basis"] = self._interaction_evidence_basis_v2(
            device_interaction_status
        )
        summary["hand_device_interaction_proxy"] = device_interaction_status in {
            "detected",
            "possible",
        }
        summary["local_motion_level"] = localized_motion_level
        summary["local_motion_burst_count"] = localized_bursts
        summary["motion_pattern"] = self._motion_pattern(
            localized_motion_level,
            localized_bursts,
        )
        summary["global_motion_level"] = effective_motion_confounding_level
        summary["global_motion_burst_count"] = raw_global_bursts
        summary["device_region_motion_level"] = self._device_region_motion_level_v2(
            summary,
            localized_motion_level,
        )
        summary["interaction_intensity"] = self._interaction_intensity(
            device_visibility_level=summary.get("device_visibility_level", "none"),
            hand_visibility_level=summary.get("hand_visibility_level", "none"),
            device_interaction_status=device_interaction_status,
            local_motion_level=localized_motion_level,
            local_motion_burst_count=localized_bursts,
        )
        summary["stable_screen_context"] = self._stable_screen_context_v2(summary)
        summary["screen_gaze_proxy"] = self._screen_gaze_proxy_v2(summary)
        summary["gaze_state"] = self._gaze_state(
            bool(summary["screen_gaze_proxy"]),
            str(summary.get("device_visibility_level", "none")),
        )
        summary["event_confidence"] = self._event_confidence_v2(summary)
        return summary

    def _summarize_frame_analyses(self, analyses: list[dict[str, Any]]) -> dict[str, Any]:
        frame_summary = super()._summarize_frame_analyses(analyses)
        readable_windows = [item for item in analyses if item["sampled_frame_count"] > 0]
        detector_policy = self._detector_policy_summary(
            readable_windows,
            frame_summary.get("event_windows", []),
        )
        behavior_event_windows = detector_policy["behavior_event_windows"]
        stateful = self._stateful_behavior_summary(
            readable_windows,
            behavior_event_windows,
            frame_summary.get("quality_summary", {}),
            detector_policy,
        )

        global_features = frame_summary["global_features"]
        frame_summary["event_windows"] = behavior_event_windows
        global_features["event_window_count"] = len(behavior_event_windows)
        global_features["event_trigger_counts"] = dict(
            Counter(event.get("trigger_type") for event in behavior_event_windows)
        )
        global_features["detector_policy_summary"] = detector_policy[
            "detector_policy_summary"
        ]
        global_features["stateful_behavior_summary"] = stateful
        global_features["state_space_summary"] = stateful["state_space_summary"]
        global_features["temporal_behavior_summary"] = stateful[
            "temporal_behavior_summary"
        ]
        global_features["evidence_graph"] = stateful["evidence_graph"]

        frame_summary["status"] = "preprocessor_v2_stateful_frame_proxy_analysis"
        frame_summary["limitations"] = [
            "stateful_lightweight_frame_features",
            "symbolic_privacy_filtered_window_states",
            "relative_window_order_only",
            "heuristic_screen_device_visibility",
            "hand_device_interaction_uses_hand_visibility_and_local_motion",
            "device_region_motion_without_hand_visibility_is_separated",
            "sparse_frame_eye_openness_features",
            "facial_geometry_action_features",
        ]
        frame_summary["missing_information"] = [
            "validated_eye_tracking",
            "clinical_facial_au_labels",
            "confirmed_touch_coordinates",
            "validated_device_touch_logs",
        ]
        frame_summary["frame_analysis_summary"]["stateful_window_state_count"] = len(
            stateful["per_window_states"]
        )
        frame_summary["frame_analysis_summary"]["unique_window_state_count"] = stateful[
            "state_space_summary"
        ]["unique_window_state_count"]
        frame_summary["frame_analysis_summary"]["behavior_episode_count"] = len(
            stateful["behavior_episodes"]
        )
        frame_summary["frame_analysis_summary"]["event_window_count"] = len(
            behavior_event_windows
        )
        frame_summary["frame_analysis_summary"]["auxiliary_observation_count"] = len(
            detector_policy["auxiliary_observations"]
        )
        frame_summary["frame_analysis_summary"]["quality_only_observation_count"] = len(
            detector_policy["quality_only_observations"]
        )
        return frame_summary

    def _detector_policy_summary(
        self,
        windows: list[dict[str, Any]],
        event_windows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        behavior_events = []
        auxiliary_observations = []
        quality_only_observations = []
        weak_proxy_observations = []
        detector_role_counts: Counter[str] = Counter()

        for event in event_windows:
            role = self._event_evidence_role(event)
            event = {**event, "evidence_role": role}
            detector_role_counts[role] += 1
            if role == "strong_behavior_evidence":
                behavior_events.append(event)
            elif role == "behavior_context":
                behavior_events.append(event)
            elif role == "quality_limitation":
                quality_only_observations.append(event)
            elif role == "auxiliary_context":
                auxiliary_observations.append(event)
            else:
                weak_proxy_observations.append(event)

        episode_detectors = self._episode_level_detectors(windows)
        for detector in episode_detectors:
            detector_role_counts[str(detector.get("evidence_role"))] += 1

        return {
            "behavior_event_windows": behavior_events,
            "auxiliary_observations": auxiliary_observations,
            "quality_only_observations": quality_only_observations,
            "weak_proxy_observations": weak_proxy_observations,
            "episode_detectors": episode_detectors,
            "detector_policy_summary": {
                "schema_version": "privte_detector_policy.v2",
                "behavior_event_count": len(behavior_events),
                "auxiliary_observation_count": len(auxiliary_observations),
                "quality_only_observation_count": len(quality_only_observations),
                "weak_proxy_observation_count": len(weak_proxy_observations),
                "episode_detector_count": len(episode_detectors),
                "evidence_role_counts": dict(detector_role_counts),
                "suppressed_event_triggers": dict(
                    Counter(
                        event.get("trigger_type")
                        for event in [
                            *auxiliary_observations,
                            *quality_only_observations,
                            *weak_proxy_observations,
                        ]
                    )
                ),
            },
        }

    @staticmethod
    def _event_evidence_role(event: dict[str, Any]) -> str:
        trigger = event.get("trigger_type")
        confidence = event.get("quality_metrics", {}).get("event_confidence")
        if trigger == "device_interaction_evidence":
            return (
                "strong_behavior_evidence"
                if confidence in {"medium", "high"}
                else "weak_proxy_evidence"
            )
        if trigger == "sustained_screen_oriented_proxy":
            return "behavior_context"
        if trigger in {"quality_drop"}:
            return "quality_limitation"
        if trigger in {"facial_au_trend", "blink_rate_change"}:
            return "auxiliary_context"
        if trigger in {"device_region_motion_proxy", "stable_screen_context"}:
            return "weak_proxy_evidence"
        return "auxiliary_context"

    def _episode_level_detectors(
        self,
        windows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        states = [self._window_state(item) for item in windows]
        state_counts = Counter(state["dominant_behavior_state"] for state in states)
        temporal = self._temporal_behavior_summary(states)
        detectors = []

        confirmed = (
            state_counts["confirmed_interaction_proxy"]
            + state_counts["confirmed_repetitive_interaction_proxy"]
        )
        possible = (
            state_counts["possible_interaction_proxy"]
            + state_counts["possible_repeated_interaction_proxy"]
        )
        if confirmed >= 2 or (confirmed >= 1 and possible >= 1):
            detectors.append(
                self._episode_detector(
                    name="confirmed_interaction_episode",
                    role="strong_behavior_evidence",
                    strength="high" if confirmed >= 2 else "medium",
                    window_count=confirmed + possible,
                    supporting_components=[
                        "hand_near_device",
                        "localized_repeated_motion",
                        "visible_device_context",
                    ],
                )
            )

        passive = int(temporal.get("passive_screen_window_count", 0) or 0)
        longest_passive = int(temporal.get("longest_passive_screen_streak", 0) or 0)
        stable = int(temporal.get("stable_screen_context_window_count", 0) or 0)
        if passive >= 3 or longest_passive >= 2 or stable >= 3:
            detectors.append(
                self._episode_detector(
                    name="sustained_passive_viewing_episode",
                    role="behavior_context",
                    strength="medium" if longest_passive >= 2 or passive >= 3 else "low",
                    window_count=passive,
                    supporting_components=[
                        "stable_device_context",
                        "screen_oriented_proxy",
                        "privacy_filtered_window_sequence",
                    ],
                )
            )

        close_count = int(temporal.get("close_posture_window_count", 0) or 0)
        if close_count >= max(3, len(states) // 2):
            detectors.append(
                self._episode_detector(
                    name="sustained_close_posture_episode",
                    role="weak_proxy_evidence",
                    strength="medium" if close_count >= 6 else "low",
                    window_count=close_count,
                    supporting_components=[
                        "face_size_close_to_screen_proxy",
                        "face_visibility_quality_gated",
                    ],
                )
            )

        if possible >= 2 and confirmed == 0:
            detectors.append(
                self._episode_detector(
                    name="possible_interaction_episode",
                    role="weak_proxy_evidence",
                    strength="low",
                    window_count=possible,
                    supporting_components=[
                        "limited_hand_visibility",
                        "localized_motion",
                        "visible_device_context",
                    ],
                )
            )

        if self._negative_behavior_evidence_present(temporal, state_counts, states):
            detectors.append(
                self._episode_detector(
                    name="negative_behavior_evidence",
                    role="counter_evidence",
                    strength="medium",
                    window_count=len(states),
                    supporting_components=[
                        "no_confirmed_interaction_episode",
                        "no_sustained_passive_viewing_episode",
                        "adequate_sampled_windows",
                    ],
                )
            )
        return detectors

    @staticmethod
    def _episode_detector(
        *,
        name: str,
        role: str,
        strength: str,
        window_count: int,
        supporting_components: list[str],
    ) -> dict[str, Any]:
        return {
            "detector_name": name,
            "evidence_role": role,
            "detector_strength": strength,
            "window_count": window_count,
            "supporting_components": supporting_components,
        }

    @staticmethod
    def _negative_behavior_evidence_present(
        temporal: dict[str, Any],
        state_counts: Counter[str],
        states: list[dict[str, Any]],
    ) -> bool:
        if len(states) < 4:
            return False
        confirmed = (
            state_counts["confirmed_interaction_proxy"]
            + state_counts["confirmed_repetitive_interaction_proxy"]
        )
        passive = int(temporal.get("passive_screen_window_count", 0) or 0)
        longest_confirmed = int(
            temporal.get("longest_confirmed_interaction_streak", 0) or 0
        )
        longest_passive = int(temporal.get("longest_passive_screen_streak", 0) or 0)
        quality_limited = int(temporal.get("quality_limited_window_count", 0) or 0)
        return (
            confirmed == 0
            and longest_confirmed == 0
            and passive <= 1
            and longest_passive <= 1
            and quality_limited < len(states) // 2
        )

    @staticmethod
    def _motion_dominance(
        *,
        raw_local_bursts: int,
        raw_global_bursts: int,
        localized_bursts: int,
        localized_motion_level: str,
        raw_global_motion_level: str,
    ) -> str:
        if localized_motion_level in {"elevated", "high"} and localized_bursts >= 2:
            if raw_global_motion_level in {"elevated", "high"}:
                return "localized_motion_present_with_broad_motion"
            return "localized_lower_frame_motion"
        if raw_global_motion_level in {"elevated", "high"} and raw_global_bursts >= max(
            2,
            raw_local_bursts,
        ):
            return "broad_frame_motion_dominant"
        if raw_local_bursts > 0 or raw_global_bursts > 0:
            return "low_or_mixed_motion"
        return "no_clear_motion"

    @staticmethod
    def _effective_motion_confounding_level(motion_dominance: str) -> str:
        if motion_dominance == "broad_frame_motion_dominant":
            return "high"
        if motion_dominance == "localized_motion_present_with_broad_motion":
            return "elevated"
        if motion_dominance == "low_or_mixed_motion":
            return "low"
        return "none"

    @staticmethod
    def _device_interaction_status_v2(
        *,
        summary: dict[str, Any],
        localized_motion_level: str,
        localized_bursts: int,
        lower_or_middle_hands: int,
        lower_hands: int,
    ) -> str:
        device_visible = summary.get("device_visibility_level") in {
            "medium",
            "high",
            "very_high",
        }
        hand_level = summary.get("hand_visibility_level")
        hand_visible = hand_level in {"low", "medium", "high", "very_high"}
        hand_near_device = lower_or_middle_hands > 0
        lower_hand_present = lower_hands > 0
        localized_active = localized_motion_level in {"elevated", "high"}
        repeated_localized = localized_bursts >= 2

        if not device_visible:
            return "not_observed"
        if (
            hand_level in {"medium", "high", "very_high"}
            and hand_near_device
            and localized_active
            and repeated_localized
        ):
            return "detected"
        if (
            hand_visible
            and hand_near_device
            and localized_active
            and (repeated_localized or lower_hand_present)
        ):
            return "possible"
        if localized_active:
            return "device_region_motion_only"
        return "not_observed"

    @staticmethod
    def _interaction_evidence_basis_v2(device_interaction_status: str) -> str:
        if device_interaction_status == "detected":
            return "hand_near_device_visibility_and_repeated_localized_motion"
        if device_interaction_status == "possible":
            return "limited_hand_near_device_visibility_and_localized_motion"
        if device_interaction_status == "device_region_motion_only":
            return "localized_device_region_motion_without_reliable_hand_proximity"
        return "no_clear_hand_device_evidence"

    @staticmethod
    def _device_region_motion_level_v2(
        summary: dict[str, Any],
        localized_motion_level: str,
    ) -> str:
        if summary.get("device_visibility_level") not in {"medium", "high", "very_high"}:
            return "not_reliable"
        if localized_motion_level in {"elevated", "high"}:
            return localized_motion_level
        return "none"

    @staticmethod
    def _stable_screen_context_v2(summary: dict[str, Any]) -> bool:
        face_visible = summary.get("face_visibility_level") in {
            "medium",
            "high",
            "very_high",
        }
        device_visible = summary.get("device_visibility_level") in {
            "high",
            "very_high",
        }
        quality_ok = summary.get("lighting_quality") not in {"low", "very_low"} and summary.get(
            "blur_quality"
        ) not in {"low", "very_low"}
        motion_ok = summary.get("v2_effective_motion_confounding_level") in {
            "none",
            "low",
            "elevated",
        }
        return bool(face_visible and device_visible and quality_ok and motion_ok)

    @staticmethod
    def _screen_gaze_proxy_v2(summary: dict[str, Any]) -> bool:
        if not summary.get("stable_screen_context"):
            return False
        if summary.get("v2_effective_motion_confounding_level") == "elevated":
            return summary.get("face_mesh_visibility_level") in {"high", "very_high"}
        return True

    @staticmethod
    def _event_confidence_v2(summary: dict[str, Any]) -> str:
        if summary.get("lighting_quality") in {"low", "very_low"} or summary.get(
            "blur_quality"
        ) in {"low", "very_low"}:
            return "low"
        status = summary.get("device_interaction_status")
        if status == "detected":
            return "high"
        if status == "possible":
            return "medium"
        if summary.get("stable_screen_context"):
            return "medium"
        return str(summary.get("event_confidence", "low"))

    def _stateful_behavior_summary(
        self,
        windows: list[dict[str, Any]],
        event_windows: list[dict[str, Any]],
        quality_summary: dict[str, Any],
        detector_policy: dict[str, Any],
    ) -> dict[str, Any]:
        per_window_states = [self._window_state(item) for item in windows]
        behavior_episodes = self._behavior_episodes(per_window_states)
        temporal_summary = self._temporal_behavior_summary(per_window_states)
        evidence_graph = self._evidence_graph(
            per_window_states=per_window_states,
            behavior_episodes=behavior_episodes,
            event_windows=event_windows,
            quality_summary=quality_summary,
            detector_policy=detector_policy,
        )
        state_counter = Counter(
            state["dominant_behavior_state"] for state in per_window_states
        )
        unique_signatures = {
            self._state_signature(state) for state in per_window_states
        }
        return {
            "schema_version": "privte_stateful_behavior.v2",
            "per_window_states": per_window_states,
            "behavior_episodes": behavior_episodes,
            "temporal_behavior_summary": temporal_summary,
            "evidence_graph": evidence_graph,
            "detector_policy_summary": detector_policy["detector_policy_summary"],
            "episode_detectors": detector_policy["episode_detectors"],
            "auxiliary_observations": detector_policy["auxiliary_observations"],
            "quality_only_observations": detector_policy["quality_only_observations"],
            "weak_proxy_observations": detector_policy["weak_proxy_observations"],
            "state_space_summary": {
                "window_state_count": len(per_window_states),
                "unique_window_state_count": len(unique_signatures),
                "dominant_window_state_counts": dict(state_counter),
                "behavior_episode_count": len(behavior_episodes),
                "longest_repeated_state_streak": self._longest_repeated_state_streak(
                    per_window_states
                ),
                "positive_evidence_fact_count": len(
                    evidence_graph["positive_evidence_facts"]
                ),
                "weak_proxy_evidence_fact_count": len(
                    evidence_graph["weak_proxy_evidence_facts"]
                ),
                "counter_evidence_fact_count": len(
                    evidence_graph["counter_evidence_facts"]
                ),
                "quality_limitation_fact_count": len(
                    evidence_graph["quality_limitation_facts"]
                ),
            },
        }

    def _interaction_summary(self, windows: list[dict[str, Any]]) -> dict[str, Any]:
        summary = super()._interaction_summary(windows)
        confirmed = [
            item
            for item in windows
            if item.get("device_interaction_status") == "detected"
            and item.get("local_motion_burst_count", 0)
            >= int(self.config.get("repetition_min_burst_count", 2))
        ]
        possible = [
            item
            for item in windows
            if item.get("device_interaction_status") in {"detected", "possible"}
            and item.get("local_motion_burst_count", 0)
            >= int(self.config.get("repetition_min_burst_count", 2))
        ]
        device_region_only = [
            item
            for item in windows
            if item.get("device_interaction_status")
            in {"not_reliable", "device_region_motion_only"}
            and item.get("device_region_motion_level") in {"elevated", "high"}
            and item.get("local_motion_burst_count", 0)
            >= int(self.config.get("repetition_min_burst_count", 2))
        ]
        summary["confirmed_interaction_window_count"] = len(confirmed)
        summary["possible_interaction_window_count"] = len(possible)
        summary["device_region_motion_window_count"] = len(device_region_only)
        summary["strong_repetitive_proxy_window_count"] = len(confirmed)
        if len(confirmed) >= 2:
            summary["interaction_intensity"] = "high"
            summary["repetitive_operation_level"] = "high"
        elif len(confirmed) == 1 or len(possible) >= 2:
            summary["interaction_intensity"] = "elevated"
            summary["repetitive_operation_level"] = "medium"
        elif len(possible) == 1 or len(device_region_only) >= 2:
            summary["interaction_intensity"] = "low"
            summary["repetitive_operation_level"] = "low"
        else:
            summary["interaction_intensity"] = "none"
            summary["repetitive_operation_level"] = "none"
        return summary

    def _device_region_event_score(self, item: dict[str, Any]) -> float:
        if item.get("device_interaction_status") not in {
            "not_reliable",
            "device_region_motion_only",
        }:
            return 0.0
        return super()._device_region_event_score(item)

    def _window_state(self, item: dict[str, Any]) -> dict[str, Any]:
        observability = self._window_observability_state(item)
        device_context = self._device_context_state(item)
        engagement_state = self._engagement_state(item, observability, device_context)
        interaction_rhythm = self._interaction_rhythm_state(item)
        posture_state = self._posture_component_state(item)
        screen_orientation = self._screen_orientation_component_state(item)
        eye_fatigue_proxy = self._eye_fatigue_proxy_state(item)
        dominant_behavior_state = self._dominant_behavior_state(
            observability=observability,
            device_context=device_context,
            engagement_state=engagement_state,
            interaction_rhythm=interaction_rhythm,
            posture_state=posture_state,
        )
        return {
            "window_order": item["window_order"],
            "relative_position": item["relative_position"],
            "duration_bin": item["duration_bin"],
            "dominant_behavior_state": dominant_behavior_state,
            "component_states": {
                "observability": observability,
                "device_context": device_context,
                "engagement_state": engagement_state,
                "interaction_rhythm": interaction_rhythm,
                "posture_state": posture_state,
                "screen_orientation": screen_orientation,
                "eye_fatigue_proxy": eye_fatigue_proxy,
                "motion_confounding": item.get("global_motion_level", "not_computed"),
                "motion_dominance": item.get("v2_motion_dominance", "not_computed"),
                "multi_person_context": (
                    "possible_interference"
                    if item.get("multi_person_interference")
                    else "none_detected"
                ),
            },
            "quality_gates": {
                "face_visibility": item.get("face_visibility_level"),
                "hand_visibility": item.get("hand_visibility_level"),
                "device_visibility": item.get("device_visibility_level"),
                "face_mesh_visibility": item.get("face_mesh_visibility_level"),
                "lighting_quality": item.get("lighting_quality"),
                "blur_quality": item.get("blur_quality"),
                "state_confidence": self._state_confidence(item, observability),
            },
            "local_proxy_counts": {
                "local_motion_burst_count": item.get("local_motion_burst_count", 0),
                "global_motion_burst_count": item.get("global_motion_burst_count", 0),
                "raw_local_motion_burst_count": item.get(
                    "raw_local_motion_burst_count", 0
                ),
                "raw_global_motion_burst_count": item.get(
                    "raw_global_motion_burst_count", 0
                ),
                "facial_action_proxy_count": len(item.get("facial_au_codes", [])),
            },
            "privacy_note": (
                "state uses relative window order and categorical proxy facts only"
            ),
        }

    @staticmethod
    def _window_observability_state(item: dict[str, Any]) -> str:
        if item.get("sampled_frame_count", 0) <= 0:
            return "unusable_no_readable_frames"
        poor_quality = item.get("lighting_quality") in {"very_low", "low"} or item.get(
            "blur_quality"
        ) in {"very_low", "low"}
        face_visible = item.get("face_visibility_level") != "none"
        device_visible = item.get("device_visibility_level") != "none"
        if poor_quality and not face_visible and not device_visible:
            return "poor_visual_quality"
        if not face_visible and not device_visible:
            return "limited_no_face_or_device_context"
        if poor_quality:
            return "partial_quality_limited"
        return "usable_proxy_window"

    @staticmethod
    def _device_context_state(item: dict[str, Any]) -> str:
        device_level = item.get("device_visibility_level")
        if device_level == "none":
            return "device_not_observed"
        if item.get("stable_screen_context"):
            return "stable_device_context"
        if device_level in {"high", "very_high"}:
            return "device_visible_unstable_context"
        return "partial_device_context"

    @staticmethod
    def _engagement_state(
        item: dict[str, Any],
        observability: str,
        device_context: str,
    ) -> str:
        if observability in {
            "unusable_no_readable_frames",
            "poor_visual_quality",
            "limited_no_face_or_device_context",
        }:
            return "not_reliably_observable"
        status = item.get("device_interaction_status")
        if status == "detected":
            return "confirmed_hand_device_interaction_proxy"
        if status == "possible":
            return "possible_hand_device_interaction_proxy"
        if item.get("screen_gaze_proxy") or item.get("stable_screen_context"):
            return "passive_screen_oriented_proxy"
        if device_context != "device_not_observed":
            return "device_visible_no_confirmed_interaction"
        return "no_device_engagement_observed"

    @staticmethod
    def _interaction_rhythm_state(item: dict[str, Any]) -> str:
        bursts = int(item.get("local_motion_burst_count", 0) or 0)
        status = item.get("device_interaction_status")
        if bursts >= 4 and status in {"detected", "possible"}:
            return "sustained_repetitive_interaction_proxy"
        if bursts >= 2 and status in {"detected", "possible"}:
            return "repeated_local_bursts_near_device"
        if (
            bursts >= 2
            and status == "device_region_motion_only"
            and item.get("device_region_motion_level") in {"elevated", "high"}
        ):
            return "device_region_repeated_motion_without_confirmed_hand"
        if bursts == 1:
            return "isolated_local_motion"
        return "no_repeated_motion_proxy"

    @staticmethod
    def _posture_component_state(item: dict[str, Any]) -> str:
        posture = item.get("posture_state")
        motion = item.get("v2_effective_motion_confounding_level") or item.get(
            "global_motion_level"
        )
        if posture == "static_close_to_screen":
            return "close_static_proxy"
        if posture == "close_to_screen":
            if motion in {"elevated", "high"}:
                return "close_to_screen_motion_confounded"
            return "close_to_screen_proxy"
        if posture in {"mostly_upright", "visible_not_close"}:
            return "visible_not_close_proxy"
        return "posture_not_reliably_estimable"

    @staticmethod
    def _screen_orientation_component_state(item: dict[str, Any]) -> str:
        if item.get("screen_gaze_proxy"):
            return "sustained_screen_orientation_proxy"
        if item.get("stable_screen_context"):
            return "stable_screen_context_without_gaze_claim"
        if item.get("v2_effective_motion_confounding_level") in {"elevated", "high"}:
            return "screen_orientation_motion_confounded"
        if item.get("device_visibility_level") != "none":
            return "device_visible_orientation_not_estimated"
        return "screen_orientation_not_observable"

    @staticmethod
    def _eye_fatigue_proxy_state(item: dict[str, Any]) -> str:
        if item.get("face_mesh_visibility_level") == "none":
            return "not_estimable_no_facemesh"
        change = item.get("blink_rate_change")
        openness = item.get("eye_openness_level")
        if change in {"elevated_eye_closure", "reduced_eye_openness"}:
            if openness in {"low", "very_low"}:
                return "repeated_low_eye_openness_proxy"
            return "weak_eye_openness_change_proxy"
        return "no_eye_fatigue_proxy_observed"

    @staticmethod
    def _dominant_behavior_state(
        *,
        observability: str,
        device_context: str,
        engagement_state: str,
        interaction_rhythm: str,
        posture_state: str,
    ) -> str:
        if observability in {"unusable_no_readable_frames", "poor_visual_quality"}:
            return "quality_limited_unobservable"
        if engagement_state == "confirmed_hand_device_interaction_proxy":
            if interaction_rhythm in {
                "sustained_repetitive_interaction_proxy",
                "repeated_local_bursts_near_device",
            }:
                return "confirmed_repetitive_interaction_proxy"
            return "confirmed_interaction_proxy"
        if engagement_state == "possible_hand_device_interaction_proxy":
            if interaction_rhythm == "repeated_local_bursts_near_device":
                return "possible_repeated_interaction_proxy"
            return "possible_interaction_proxy"
        if engagement_state == "passive_screen_oriented_proxy":
            if posture_state.startswith("close"):
                return "close_posture_passive_screen_proxy"
            return "passive_screen_oriented_proxy"
        if interaction_rhythm == "device_region_repeated_motion_without_confirmed_hand":
            return "weak_device_region_motion_proxy"
        if device_context != "device_not_observed":
            return "device_visible_no_confirmed_engagement"
        return "no_relevant_device_behavior_observed"

    @staticmethod
    def _state_confidence(item: dict[str, Any], observability: str) -> str:
        if observability in {
            "unusable_no_readable_frames",
            "poor_visual_quality",
            "limited_no_face_or_device_context",
        }:
            return "low"
        if item.get("event_confidence") == "high" and item.get(
            "global_motion_level"
        ) not in {"high"}:
            return "high"
        if item.get("event_confidence") in {"medium", "high"}:
            return "medium"
        return "low"

    @staticmethod
    def _state_signature(state: dict[str, Any]) -> tuple[Any, ...]:
        components = state.get("component_states", {})
        return (
            state.get("dominant_behavior_state"),
            components.get("observability"),
            components.get("device_context"),
            components.get("engagement_state"),
            components.get("interaction_rhythm"),
            components.get("posture_state"),
            components.get("screen_orientation"),
            components.get("eye_fatigue_proxy"),
            components.get("motion_confounding"),
            components.get("motion_dominance"),
        )

    def _behavior_episodes(
        self,
        states: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not states:
            return []
        episodes = []
        start = states[0]
        current_state = start["dominant_behavior_state"]
        members = [start]
        for state in states[1:]:
            if state["dominant_behavior_state"] == current_state:
                members.append(state)
                continue
            episodes.append(self._episode_from_members(current_state, members))
            current_state = state["dominant_behavior_state"]
            members = [state]
        episodes.append(self._episode_from_members(current_state, members))
        return episodes

    @staticmethod
    def _episode_from_members(
        dominant_state: str,
        members: list[dict[str, Any]],
    ) -> dict[str, Any]:
        positions = Counter(member["relative_position"] for member in members)
        confidence = Counter(
            member.get("quality_gates", {}).get("state_confidence", "low")
            for member in members
        )
        component_counts = Counter()
        for member in members:
            for component, value in member.get("component_states", {}).items():
                component_counts[f"{component}:{value}"] += 1
        return {
            "episode_id": f"episode_{len(members)}_{members[0]['window_order']:02d}",
            "dominant_behavior_state": dominant_state,
            "window_count": len(members),
            "window_order_start": members[0]["window_order"],
            "window_order_end": members[-1]["window_order"],
            "relative_position_counts": dict(positions),
            "episode_confidence": confidence.most_common(1)[0][0],
            "component_state_counts": dict(component_counts.most_common(8)),
        }

    @staticmethod
    def _longest_repeated_state_streak(states: list[dict[str, Any]]) -> int:
        longest = 0
        current = 0
        previous = None
        for state in states:
            value = state.get("dominant_behavior_state")
            if value == previous:
                current += 1
            else:
                current = 1
                previous = value
            longest = max(longest, current)
        return longest

    def _temporal_behavior_summary(
        self,
        states: list[dict[str, Any]],
    ) -> dict[str, Any]:
        by_position: dict[str, Counter[str]] = {
            "early": Counter(),
            "middle": Counter(),
            "late": Counter(),
        }
        interaction_scores = []
        close_posture_count = 0
        quality_limited_count = 0
        for state in states:
            dominant = state["dominant_behavior_state"]
            by_position.setdefault(state["relative_position"], Counter())[dominant] += 1
            interaction_scores.append(self._engagement_score(state))
            posture = state.get("component_states", {}).get("posture_state")
            if isinstance(posture, str) and posture.startswith("close"):
                close_posture_count += 1
            if dominant == "quality_limited_unobservable":
                quality_limited_count += 1
        return {
            "relative_position_state_counts": {
                key: dict(value) for key, value in by_position.items()
            },
            "engagement_trajectory": self._trajectory_label(interaction_scores),
            "temporal_interaction_pattern": self._interaction_pattern_label(states),
            "close_posture_window_count": close_posture_count,
            "quality_limited_window_count": quality_limited_count,
            "passive_screen_window_count": self._passive_screen_window_count(states),
            "stable_screen_context_window_count": self._stable_screen_context_window_count(
                states
            ),
            "longest_passive_screen_streak": self._longest_matching_streak(
                states,
                {
                    "passive_screen_oriented_proxy",
                    "close_posture_passive_screen_proxy",
                },
            ),
            "longest_confirmed_interaction_streak": self._longest_matching_streak(
                states,
                {
                    "confirmed_interaction_proxy",
                    "confirmed_repetitive_interaction_proxy",
                },
            ),
            "dominant_session_state": self._dominant_session_state(states),
        }

    @staticmethod
    def _passive_screen_window_count(states: list[dict[str, Any]]) -> int:
        return sum(
            1
            for state in states
            if state.get("dominant_behavior_state")
            in {"passive_screen_oriented_proxy", "close_posture_passive_screen_proxy"}
        )

    @staticmethod
    def _stable_screen_context_window_count(states: list[dict[str, Any]]) -> int:
        return sum(
            1
            for state in states
            if state.get("component_states", {}).get("device_context")
            == "stable_device_context"
        )

    @staticmethod
    def _longest_matching_streak(
        states: list[dict[str, Any]],
        target_states: set[str],
    ) -> int:
        longest = 0
        current = 0
        for state in states:
            if state.get("dominant_behavior_state") in target_states:
                current += 1
            else:
                current = 0
            longest = max(longest, current)
        return longest

    @staticmethod
    def _engagement_score(state: dict[str, Any]) -> int:
        dominant = state.get("dominant_behavior_state")
        return {
            "confirmed_repetitive_interaction_proxy": 4,
            "confirmed_interaction_proxy": 3,
            "possible_repeated_interaction_proxy": 3,
            "possible_interaction_proxy": 2,
            "close_posture_passive_screen_proxy": 2,
            "passive_screen_oriented_proxy": 2,
            "weak_device_region_motion_proxy": 1,
            "device_visible_no_confirmed_engagement": 1,
            "no_relevant_device_behavior_observed": 0,
            "quality_limited_unobservable": 0,
        }.get(str(dominant), 0)

    @staticmethod
    def _trajectory_label(scores: list[int]) -> str:
        if not scores:
            return "not_computed"
        thirds = max(1, len(scores) // 3)
        early = sum(scores[:thirds]) / max(1, len(scores[:thirds]))
        late = sum(scores[-thirds:]) / max(1, len(scores[-thirds:]))
        if max(scores) == 0:
            return "no_observed_engagement_signal"
        if late - early >= 1.0:
            return "increasing_proxy_engagement"
        if early - late >= 1.0:
            return "decreasing_proxy_engagement"
        if sum(1 for score in scores if score >= 2) >= 2:
            return "intermittent_proxy_engagement"
        return "sparse_or_low_proxy_engagement"

    @staticmethod
    def _interaction_pattern_label(states: list[dict[str, Any]]) -> str:
        dominant_values = [state["dominant_behavior_state"] for state in states]
        confirmed = sum(
            1 for value in dominant_values if value.startswith("confirmed_")
        )
        possible = sum(1 for value in dominant_values if value.startswith("possible_"))
        passive = sum("passive_screen" in value for value in dominant_values)
        weak = dominant_values.count("weak_device_region_motion_proxy")
        if confirmed >= 3:
            return "multi_window_confirmed_interaction_pattern"
        if confirmed >= 1 and possible >= 1:
            return "mixed_confirmed_and_possible_interaction_pattern"
        if possible >= 2:
            return "multi_window_possible_interaction_pattern"
        if passive >= 2:
            return "multi_window_passive_screen_orientation_pattern"
        if weak >= 2:
            return "weak_device_region_motion_pattern"
        return "no_consistent_interaction_pattern"

    @staticmethod
    def _dominant_session_state(states: list[dict[str, Any]]) -> str:
        if not states:
            return "not_computed"
        counts = Counter(state["dominant_behavior_state"] for state in states)
        return counts.most_common(1)[0][0]

    def _evidence_graph(
        self,
        *,
        per_window_states: list[dict[str, Any]],
        behavior_episodes: list[dict[str, Any]],
        event_windows: list[dict[str, Any]],
        quality_summary: dict[str, Any],
        detector_policy: dict[str, Any],
    ) -> dict[str, list[str]]:
        state_counts = Counter(
            state["dominant_behavior_state"] for state in per_window_states
        )
        event_counts = Counter(event.get("trigger_type") for event in event_windows)
        episode_detectors = detector_policy["episode_detectors"]
        positive = self._positive_evidence_facts(
            state_counts,
            event_counts,
            episode_detectors,
        )
        weak = self._weak_proxy_evidence_facts(
            state_counts,
            event_counts,
            detector_policy["weak_proxy_observations"],
            detector_policy["auxiliary_observations"],
            episode_detectors,
        )
        counter = self._counter_evidence_facts(
            state_counts,
            event_counts,
            episode_detectors,
        )
        quality = self._quality_limitation_facts(
            quality_summary,
            per_window_states,
            detector_policy["quality_only_observations"],
        )
        auxiliary = self._auxiliary_observation_facts(
            detector_policy["auxiliary_observations"],
        )
        if behavior_episodes:
            longest = max(behavior_episodes, key=lambda item: item["window_count"])
            weak.append(
                "Longest symbolic behavior episode was "
                f"{longest['dominant_behavior_state']} across "
                f"{longest['window_count']} sampled windows."
            )
        return {
            "positive_evidence_facts": positive,
            "weak_proxy_evidence_facts": weak,
            "counter_evidence_facts": counter,
            "quality_limitation_facts": quality,
            "auxiliary_observation_facts": auxiliary,
        }

    @staticmethod
    def _positive_evidence_facts(
        state_counts: Counter[str],
        event_counts: Counter[str],
        episode_detectors: list[dict[str, Any]],
    ) -> list[str]:
        facts = []
        confirmed = (
            state_counts["confirmed_repetitive_interaction_proxy"]
            + state_counts["confirmed_interaction_proxy"]
        )
        possible = (
            state_counts["possible_repeated_interaction_proxy"]
            + state_counts["possible_interaction_proxy"]
        )
        passive = (
            state_counts["close_posture_passive_screen_proxy"]
            + state_counts["passive_screen_oriented_proxy"]
        )
        if confirmed:
            facts.append(
                f"Confirmed hand-device interaction proxies appeared in {confirmed} sampled windows."
            )
        if state_counts["confirmed_repetitive_interaction_proxy"]:
            facts.append(
                "Repeated interaction rhythm was confirmed in "
                f"{state_counts['confirmed_repetitive_interaction_proxy']} sampled windows."
            )
        if possible >= 2:
            facts.append(
                f"Possible hand-device interaction proxies appeared in {possible} sampled windows."
            )
        if passive >= 2:
            facts.append(
                f"Passive screen-oriented or close-posture screen proxies appeared in {passive} sampled windows."
            )
        if passive >= 3:
            facts.append(
                "Passive viewing evidence formed a multi-window pattern rather than a single isolated window."
            )
        if event_counts["sustained_screen_oriented_proxy"]:
            facts.append(
                "Selected event windows included sustained screen-oriented proxy evidence."
            )
        for detector in episode_detectors:
            if detector.get("evidence_role") != "strong_behavior_evidence":
                continue
            facts.append(
                f"{detector['detector_name']} detected with {detector['detector_strength']} strength across {detector['window_count']} sampled windows."
            )
        return facts

    @staticmethod
    def _weak_proxy_evidence_facts(
        state_counts: Counter[str],
        event_counts: Counter[str],
        weak_proxy_observations: list[dict[str, Any]],
        auxiliary_observations: list[dict[str, Any]],
        episode_detectors: list[dict[str, Any]],
    ) -> list[str]:
        facts = []
        weak_motion = state_counts["weak_device_region_motion_proxy"]
        if weak_motion:
            facts.append(
                "Device-region motion with limited hand visibility appeared in "
                f"{weak_motion} sampled windows."
            )
        auxiliary_counts = Counter(
            event.get("trigger_type") for event in auxiliary_observations
        )
        weak_counts = Counter(event.get("trigger_type") for event in weak_proxy_observations)
        if auxiliary_counts["blink_rate_change"]:
            facts.append(
                f"Eye-openness changes appeared in {auxiliary_counts['blink_rate_change']} auxiliary observations."
            )
        if auxiliary_counts["facial_au_trend"]:
            facts.append(
                f"Facial geometry changes appeared in {auxiliary_counts['facial_au_trend']} auxiliary observations."
            )
        if weak_counts:
            facts.append(
                f"Additional weak-observation trigger counts: {dict(weak_counts)}."
            )
        for detector in episode_detectors:
            if detector.get("evidence_role") != "weak_proxy_evidence":
                continue
            facts.append(
                f"{detector['detector_name']} appeared across {detector['window_count']} sampled windows."
            )
        return facts

    @staticmethod
    def _counter_evidence_facts(
        state_counts: Counter[str],
        event_counts: Counter[str],
        episode_detectors: list[dict[str, Any]],
    ) -> list[str]:
        facts = []
        confirmed = (
            state_counts["confirmed_repetitive_interaction_proxy"]
            + state_counts["confirmed_interaction_proxy"]
        )
        strong_events = (
            event_counts["device_interaction_evidence"]
            + event_counts["sustained_screen_oriented_proxy"]
        )
        if confirmed == 0:
            facts.append("Confirmed hand-device interaction windows: 0.")
        if state_counts["confirmed_repetitive_interaction_proxy"] == 0:
            facts.append("Confirmed repetitive interaction episodes: 0.")
        if strong_events == 0:
            facts.append("Selected strong interaction or sustained-screen event windows: 0.")
        if state_counts["device_visible_no_confirmed_engagement"]:
            facts.append(
                "Device visibility often occurred without confirmed engagement evidence."
            )
        if not event_counts:
            facts.append("Stateful V2 selected event windows: 0.")
        for detector in episode_detectors:
            if detector.get("evidence_role") != "counter_evidence":
                continue
            facts.append(
                f"{detector['detector_name']} appeared across {detector['window_count']} sampled windows."
            )
        return facts

    @staticmethod
    def _quality_limitation_facts(
        quality_summary: dict[str, Any],
        states: list[dict[str, Any]],
        quality_only_observations: list[dict[str, Any]],
    ) -> list[str]:
        facts = []
        if quality_summary.get("motion_confounding_level") in {"elevated", "high"}:
            facts.append(
                f"Broad motion confounding level: {quality_summary.get('motion_confounding_level')}."
            )
        if quality_summary.get("hand_observability") in {"none", "low"}:
            facts.append(
                f"Hand observability level: {quality_summary.get('hand_observability')}."
            )
        if quality_summary.get("gaze_estimation_quality", "").startswith(
            "not_reliably"
        ):
            facts.append(
                f"Screen-orientation quality: {quality_summary.get('gaze_estimation_quality')}."
            )
        poor = sum(
            1
            for state in states
            if state["dominant_behavior_state"] == "quality_limited_unobservable"
        )
        if poor:
            facts.append(
                f"{poor} sampled windows were dominated by quality-limited observations."
            )
        quality_counts = Counter(
            event.get("trigger_type") for event in quality_only_observations
        )
        if quality_counts:
            facts.append(
                f"Quality-only observation trigger counts: {dict(quality_counts)}."
            )
        return facts

    @staticmethod
    def _auxiliary_observation_facts(
        auxiliary_observations: list[dict[str, Any]],
    ) -> list[str]:
        counts = Counter(event.get("trigger_type") for event in auxiliary_observations)
        facts = []
        if counts["blink_rate_change"]:
            facts.append(
                f"{counts['blink_rate_change']} eye-openness proxy observations were retained for context only."
            )
        if counts["facial_au_trend"]:
            facts.append(
                f"{counts['facial_au_trend']} facial-geometry proxy observations were retained for context only."
            )
        return facts
