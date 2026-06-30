"""PriVTE preprocessor v1.

V1 keeps the v0 public evidence schema but tightens event selection and global
aggregation. The goal is to make the current demo easier to tune: fewer
over-triggered event windows, more conservative repetitive-operation bins, and
explicit negative evidence for text-only screening.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .common import safe_ratio
from .privte_preprocessor_v0 import PriVTEPreprocessorV0Extractor


class PriVTEPreprocessorV1Extractor(PriVTEPreprocessorV0Extractor):
    """Conservative calibration pass over the v0 frame-proxy preprocessor."""

    name = "privte_preprocessor_v1"
    version = "v1"

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
            item
            for item in readable_windows
            if item["face_mesh_visibility_level"] != "none"
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
        blink_proxy_summary = self._blink_proxy_summary(readable_windows)
        facial_action_proxy_counts = self._facial_action_proxy_counts(readable_windows)
        interaction_summary = self._interaction_summary(readable_windows)
        event_windows = self._build_event_windows(readable_windows)
        trigger_counts = Counter(
            str(event.get("trigger_type")) for event in event_windows
        )

        motion_confounding_level = self._motion_confounding_level(readable_windows)
        quality_summary = {
            "overall_data_sufficiency": "adequate_for_frame_proxy_analysis",
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
            "motion_confounding_level": motion_confounding_level,
        }
        screen_gaze_ratio = screen_orientation_summary["ratio"]
        max_continuous_proxy = self._max_continuous_proxy_minutes(
            readable_windows,
            screen_orientation_summary["status"],
        )
        global_features = {
            "screen_gaze_ratio": screen_gaze_ratio,
            "screen_gaze_ratio_bin": screen_orientation_summary["ratio_bin"],
            "screen_orientation_proxy_status": screen_orientation_summary["status"],
            "screen_orientation_proxy_quality": screen_orientation_summary["quality"],
            "max_continuous_gaze_duration_minutes": max_continuous_proxy,
            "max_continuous_gaze_duration_bin": self._continuous_duration_bin(
                max_continuous_proxy
            ),
            "average_blink_rate_per_minute": None,
            "blink_rate_level": blink_proxy_summary["blink_rate_level"],
            "blink_rate_trend": blink_proxy_summary["blink_rate_trend"],
            "overall_posture_trend": self._posture_trend(readable_windows),
            "interaction_intensity": self._display_level(
                interaction_summary["interaction_intensity"]
            ),
            "repetitive_operation_level": interaction_summary[
                "repetitive_operation_level"
            ],
            "motion_confounding_level": motion_confounding_level,
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
            "event_trigger_counts": dict(trigger_counts),
            "confirmed_interaction_window_count": interaction_summary[
                "confirmed_interaction_window_count"
            ],
            "possible_interaction_window_count": interaction_summary[
                "possible_interaction_window_count"
            ],
            "device_region_motion_window_count": interaction_summary[
                "device_region_motion_window_count"
            ],
            "strong_repetitive_proxy_window_count": interaction_summary[
                "strong_repetitive_proxy_window_count"
            ],
            "negative_evidence_summary": self._negative_evidence_summary(
                readable_windows=readable_windows,
                event_windows=event_windows,
                screen_orientation_summary=screen_orientation_summary,
                interaction_summary=interaction_summary,
                quality_summary=quality_summary,
            ),
            "evidence_balance": self._evidence_balance(
                event_windows=event_windows,
                interaction_summary=interaction_summary,
                screen_orientation_summary=screen_orientation_summary,
                quality_summary=quality_summary,
            ),
        }
        limitations = [
            "calibrated_lightweight_frame_features",
            "heuristic_screen_device_visibility",
            "thresholded_event_window_selection",
            "hand_device_interaction_uses_hand_visibility_and_local_motion",
            "device_region_motion_without_hand_visibility_is_separated",
            "sparse_frame_eye_openness_features",
            "facial_geometry_action_features",
        ]
        missing_information = [
            "validated_eye_tracking",
            "clinical_facial_au_labels",
            "confirmed_touch_coordinates",
        ]
        return {
            "status": "preprocessor_v1_calibrated_frame_proxy_analysis",
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
                "confirmed_interaction_window_count": interaction_summary[
                    "confirmed_interaction_window_count"
                ],
                "possible_interaction_window_count": interaction_summary[
                    "possible_interaction_window_count"
                ],
                "device_region_motion_window_count": interaction_summary[
                    "device_region_motion_window_count"
                ],
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
            interaction_score = self._interaction_event_score(item)
            if (
                item["device_interaction_status"] in {"detected", "possible"}
                and interaction_score >= self._event_min_score("interaction")
            ):
                candidates.append(
                    self._scored_event_window(
                        item,
                        trigger_type="device_interaction_evidence",
                        trigger_source="mediapipe_hand_motion_listener",
                        interaction_pattern=item["motion_pattern"],
                        score=interaction_score,
                    )
                )

            region_score = self._device_region_event_score(item)
            if (
                item["device_interaction_status"] == "not_reliable"
                and region_score >= self._event_min_score("device_region")
            ):
                candidates.append(
                    self._scored_event_window(
                        item,
                        trigger_type="device_region_motion_proxy",
                        trigger_source="device_region_motion_listener",
                        interaction_pattern=item["motion_pattern"],
                        score=region_score,
                    )
                )

            gaze_score = self._screen_context_event_score(item)
            if item["screen_gaze_proxy"] and gaze_score >= self._event_min_score("gaze"):
                candidates.append(
                    self._scored_event_window(
                        item,
                        trigger_type="sustained_screen_oriented_proxy",
                        trigger_source="screen_orientation_proxy_listener",
                        interaction_pattern="face_and_screen_like_region_visible_with_limited_motion",
                        score=gaze_score,
                    )
                )
            elif (
                item["stable_screen_context"]
                and gaze_score >= self._event_min_score("stable_screen")
            ):
                candidates.append(
                    self._scored_event_window(
                        item,
                        trigger_type="stable_screen_context",
                        trigger_source="screen_context_listener",
                        interaction_pattern="low_motion_with_visible_screen_like_region",
                        score=gaze_score,
                    )
                )

            blink_score = self._blink_event_score(item)
            if blink_score >= self._event_min_score("blink"):
                candidates.append(
                    self._scored_event_window(
                        item,
                        trigger_type="blink_rate_change",
                        trigger_source="facemesh_eye_openness_listener",
                        interaction_pattern="eye_openness_change_proxy",
                        score=blink_score,
                    )
                )

            facial_score = self._facial_event_score(item)
            if facial_score >= self._event_min_score("facial"):
                candidates.append(
                    self._scored_event_window(
                        item,
                        trigger_type="facial_au_trend",
                        trigger_source="facemesh_action_proxy_listener",
                        interaction_pattern="facial_geometry_action_proxy",
                        score=facial_score,
                    )
                )

            quality_score = self._quality_drop_event_score(item)
            if quality_score >= self._event_min_score("quality"):
                candidates.append(
                    self._scored_event_window(
                        item,
                        trigger_type="quality_drop",
                        trigger_source="quality_gateway",
                        interaction_pattern="low_quality_window",
                        score=quality_score,
                    )
                )

        deduped = self._deduplicate_and_rank_events(candidates)
        for index, event in enumerate(deduped, start=1):
            event["window_id"] = f"Event_{index:02d}"
        return deduped

    def _scored_event_window(
        self,
        item: dict[str, Any],
        *,
        trigger_type: str,
        trigger_source: str,
        interaction_pattern: str,
        score: float,
    ) -> dict[str, Any]:
        event = self._event_window(
            item,
            trigger_type=trigger_type,
            trigger_source=trigger_source,
            interaction_pattern=interaction_pattern,
        )
        event["event_strength"] = self._score_strength(score)
        event["selection_policy"] = "v1_thresholded_event_selection"
        return event

    def _deduplicate_and_rank_events(
        self,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        ranked = sorted(
            candidates,
            key=lambda event: (
                self._strength_priority(str(event.get("event_strength"))),
                self._event_priority(str(event.get("trigger_type"))),
                event["window_order"],
            ),
        )
        by_window: dict[int, dict[str, Any]] = {}
        for event in ranked:
            window_order = int(event["window_order"])
            if window_order not in by_window:
                by_window[window_order] = event

        max_events = int(self.config.get("max_event_windows", 6))
        min_events = int(self.config.get("min_event_windows", 0))
        selected = list(by_window.values())[:max_events]
        if len(selected) < min_events:
            selected = ranked[: min(max_events, min_events)]
        return sorted(selected, key=lambda event: event["window_order"])

    def _interaction_summary(self, windows: list[dict[str, Any]]) -> dict[str, Any]:
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
        device_region = [
            item
            for item in windows
            if item.get("device_interaction_status") == "not_reliable"
            and item.get("device_region_motion_level") in {"elevated", "high"}
            and item.get("local_motion_burst_count", 0)
            >= int(self.config.get("repetition_min_burst_count", 2))
        ]
        strong = [
            item
            for item in possible
            if item.get("event_confidence") == "high"
            or item.get("hand_visibility_level") in {"medium", "high", "very_high"}
        ]
        possible_count = len(possible)
        confirmed_count = len(confirmed)
        device_region_count = len(device_region)
        strong_count = len(strong)
        if confirmed_count >= 2 or strong_count >= 3:
            interaction_intensity = "high"
        elif possible_count >= 2 or confirmed_count >= 1:
            interaction_intensity = "elevated"
        elif possible_count >= 1 or device_region_count >= 2:
            interaction_intensity = "low"
        else:
            interaction_intensity = "none"

        if strong_count >= 4 or (confirmed_count >= 2 and possible_count >= 3):
            repetitive_level = "high"
        elif strong_count >= 2 or confirmed_count >= 1 or possible_count >= 2:
            repetitive_level = "medium"
        elif possible_count >= 1 or device_region_count >= 2:
            repetitive_level = "low"
        else:
            repetitive_level = "none"

        return {
            "interaction_intensity": interaction_intensity,
            "repetitive_operation_level": repetitive_level,
            "confirmed_interaction_window_count": confirmed_count,
            "possible_interaction_window_count": possible_count,
            "device_region_motion_window_count": device_region_count,
            "strong_repetitive_proxy_window_count": strong_count,
        }

    def _negative_evidence_summary(
        self,
        *,
        readable_windows: list[dict[str, Any]],
        event_windows: list[dict[str, Any]],
        screen_orientation_summary: dict[str, Any],
        interaction_summary: dict[str, Any],
        quality_summary: dict[str, Any],
    ) -> list[str]:
        negatives = []
        if not event_windows:
            negatives.append("no_event_windows_passed_v1_thresholds")
        if interaction_summary["confirmed_interaction_window_count"] == 0:
            negatives.append("no_confirmed_hand_device_interaction_proxy")
        if interaction_summary["strong_repetitive_proxy_window_count"] == 0:
            negatives.append("no_strong_repetitive_operation_proxy")
        if screen_orientation_summary.get("status") != "estimated_from_visible_face_and_device":
            negatives.append("no_reliable_sustained_screen_orientation_proxy")
        if quality_summary.get("hand_observability") in {"none", "low"}:
            negatives.append("hand_visibility_limits_touch_interaction_evidence")
        if quality_summary.get("motion_confounding_level") in {"elevated", "high"}:
            negatives.append("global_motion_limits_behavior_interpretation")
        if readable_windows and all(
            item.get("stable_screen_context") is False for item in readable_windows
        ):
            negatives.append("no_stable_screen_context_window_observed")
        return negatives

    def _evidence_balance(
        self,
        *,
        event_windows: list[dict[str, Any]],
        interaction_summary: dict[str, Any],
        screen_orientation_summary: dict[str, Any],
        quality_summary: dict[str, Any],
    ) -> str:
        strong_events = sum(
            1 for event in event_windows if event.get("event_strength") == "high"
        )
        if not event_windows:
            return "no_selected_behavior_events"
        if (
            strong_events == 0
            and interaction_summary["confirmed_interaction_window_count"] == 0
            and screen_orientation_summary.get("status")
            != "estimated_from_visible_face_and_device"
        ):
            return "weak_or_confounded_proxy_evidence"
        if quality_summary.get("motion_confounding_level") in {"elevated", "high"}:
            return "behavior_proxy_present_but_motion_confounded"
        if strong_events >= 2:
            return "multiple_stronger_proxy_events"
        return "limited_proxy_evidence"

    def _interaction_event_score(self, item: dict[str, Any]) -> float:
        status = item.get("device_interaction_status")
        if status == "detected":
            score = 2.8
        elif status == "possible":
            score = 1.8
        else:
            return 0.0
        score += self._visibility_bonus(item.get("device_visibility_level"))
        score += self._hand_bonus(item.get("hand_visibility_level"))
        score += self._burst_bonus(item.get("local_motion_burst_count", 0))
        score += self._confidence_bonus(item.get("event_confidence"))
        score -= self._global_motion_penalty(item)
        return score

    def _device_region_event_score(self, item: dict[str, Any]) -> float:
        if item.get("device_region_motion_level") not in {"elevated", "high"}:
            return 0.0
        if item.get("device_visibility_level") not in {"high", "very_high"}:
            return 0.0
        score = 1.3
        score += self._visibility_bonus(item.get("device_visibility_level"))
        score += self._burst_bonus(item.get("local_motion_burst_count", 0))
        score += self._confidence_bonus(item.get("event_confidence"))
        score -= self._global_motion_penalty(item) * 1.5
        return score

    def _screen_context_event_score(self, item: dict[str, Any]) -> float:
        if not item.get("stable_screen_context") and not item.get("screen_gaze_proxy"):
            return 0.0
        score = 1.8
        score += self._visibility_bonus(item.get("device_visibility_level"))
        score += self._visibility_bonus(item.get("face_visibility_level"))
        if item.get("screen_gaze_proxy"):
            score += 0.8
        score -= self._global_motion_penalty(item)
        return score

    def _blink_event_score(self, item: dict[str, Any]) -> float:
        if item.get("blink_rate_change") not in {
            "elevated_eye_closure",
            "reduced_eye_openness",
        }:
            return 0.0
        if item.get("face_mesh_visibility_level") not in {"high", "very_high"}:
            return 0.0
        score = 1.4
        if item.get("eye_openness_level") == "low":
            score += 0.8
        score += self._visibility_bonus(item.get("face_mesh_visibility_level"))
        score += self._confidence_bonus(item.get("event_confidence"))
        score -= self._global_motion_penalty(item) * 0.5
        return score

    def _facial_event_score(self, item: dict[str, Any]) -> float:
        if not item.get("facial_au_codes"):
            return 0.0
        if item.get("face_mesh_visibility_level") not in {"high", "very_high"}:
            return 0.0
        score = 1.2 + 0.4 * min(len(item.get("facial_au_codes", [])), 3)
        score += self._visibility_bonus(item.get("face_mesh_visibility_level"))
        score += self._confidence_bonus(item.get("event_confidence"))
        score -= self._global_motion_penalty(item) * 0.5
        return score

    @staticmethod
    def _quality_drop_event_score(item: dict[str, Any]) -> float:
        score = 0.0
        if item.get("lighting_quality") in {"low", "very_low"}:
            score += 1.5
        if item.get("blur_quality") in {"low", "very_low"}:
            score += 1.5
        if item.get("face_visibility_level") == "none" and item.get(
            "device_visibility_level"
        ) == "none":
            score += 0.8
        return score

    def _event_min_score(self, event_type: str) -> float:
        defaults = {
            "interaction": 3.4,
            "device_region": 3.8,
            "gaze": 3.2,
            "stable_screen": 3.0,
            "blink": 2.8,
            "facial": 2.8,
            "quality": 2.5,
        }
        key = f"{event_type}_event_min_score"
        return float(self.config.get(key, defaults[event_type]))

    @staticmethod
    def _visibility_bonus(level: Any) -> float:
        return {
            "very_high": 0.9,
            "high": 0.7,
            "medium": 0.4,
            "low": 0.1,
        }.get(str(level), 0.0)

    @staticmethod
    def _hand_bonus(level: Any) -> float:
        return {
            "very_high": 1.0,
            "high": 0.9,
            "medium": 0.7,
            "low": 0.25,
        }.get(str(level), 0.0)

    @staticmethod
    def _burst_bonus(count: int) -> float:
        if count >= 4:
            return 1.2
        if count >= 3:
            return 0.9
        if count >= 2:
            return 0.6
        if count == 1:
            return 0.2
        return 0.0

    @staticmethod
    def _confidence_bonus(level: Any) -> float:
        return {"high": 0.4, "medium": 0.15}.get(str(level), 0.0)

    @staticmethod
    def _global_motion_penalty(item: dict[str, Any]) -> float:
        if item.get("global_motion_level") == "high":
            return 0.8
        if item.get("global_motion_level") == "elevated":
            return 0.4
        return 0.0

    @staticmethod
    def _score_strength(score: float) -> str:
        if score >= 4.5:
            return "high"
        if score >= 3.2:
            return "medium"
        return "low"

    @staticmethod
    def _strength_priority(strength: str) -> int:
        return {"high": 0, "medium": 1, "low": 2}.get(strength, 3)

    def _motion_confounding_level(self, windows: list[dict[str, Any]]) -> str:
        if not windows:
            return "not_computed"
        high = sum(1 for item in windows if item.get("global_motion_level") == "high")
        elevated_or_high = sum(
            1
            for item in windows
            if item.get("global_motion_level") in {"elevated", "high"}
        )
        high_ratio = safe_ratio(high, len(windows))
        elevated_ratio = safe_ratio(elevated_or_high, len(windows))
        if high_ratio >= float(self.config.get("motion_confounding_high_ratio", 0.75)):
            return "high"
        if elevated_ratio >= float(
            self.config.get("motion_confounding_elevated_ratio", 0.4)
        ):
            return "elevated"
        if elevated_or_high > 0:
            return "low"
        return "none"

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
        total_changed = elevated + reduced
        if elevated >= max(3, round(len(values) * 0.4)):
            trend = "frequent_eye_closure"
        elif reduced >= max(3, round(len(values) * 0.4)):
            trend = "frequent_reduced_eye_openness"
        elif total_changed > 0:
            trend = "occasional_eye_openness_change"
        else:
            trend = "mostly_stable"
        return {
            "blink_rate_level": "sparse_frame_proxy_available",
            "blink_rate_trend": trend,
            "eye_closure_proxy_level": PriVTEPreprocessorV1Extractor._count_level(
                total_changed
            ),
        }

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
            1
            for item in observable
            if item.get("global_motion_level") == "high"
        )
        if safe_ratio(confounded, len(observable)) >= 0.75:
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
            "ratio_bin": PriVTEPreprocessorV1Extractor._ratio_bin(ratio),
            "status": "estimated_from_visible_face_and_device",
            "quality": quality,
        }
