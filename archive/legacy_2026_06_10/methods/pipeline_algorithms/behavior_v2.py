"""PriVTE Behavior v2 evidence extractor.

Behavior v2 reuses the practical local CV extraction from Behavior v1, but
adds more concrete, privacy-filtered behavior evidence and a more conservative
signal calibration layer.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .behavior_v1 import PriVTEBehaviorV1Extractor


class PriVTEBehaviorV2Extractor(PriVTEBehaviorV1Extractor):
    """Behavior v2: concrete privacy-filtered behavior evidence."""

    name = "privte_behavior_v2"
    version = "v2"
    feature_schema_version = "privte_behavior_v2_feature_schema.v2"

    def _aggregate_frames(self, **kwargs: Any) -> dict[str, Any]:
        aggregate = super()._aggregate_frames(**kwargs)
        features = aggregate["behavior_v1_features"]
        key_window_summary = aggregate["key_window_summary"]
        quality = aggregate["quality_summary"]
        event_windows = aggregate["event_windows"]
        coarse_measurements = self._build_coarse_measurements(
            features=features,
            quality=quality,
            event_windows=event_windows,
        )

        concrete_evidence = self._build_concrete_behavior_evidence(
            features=features,
            key_window_summary=key_window_summary,
            quality=quality,
            coarse_measurements=coarse_measurements,
        )
        aggregate["behavior_v2_features"] = self._build_behavior_v2_features(
            features=features,
            key_window_summary=key_window_summary,
            quality=quality,
            coarse_measurements=coarse_measurements,
        )
        aggregate["concrete_behavior_evidence"] = concrete_evidence
        aggregate["privacy_preserving_behavior_summary"] = {
            "summary_type": "coarse_behavior_episode_summary",
            "no_raw_frames_or_images": True,
            "no_exact_coordinates": True,
            "no_ocr_or_app_content": True,
            "no_face_embeddings_or_landmark_sequences": True,
            "relative_time_only": True,
            "coarse_counts_and_sampled_frame_ratios_only": True,
        }
        quality["status"] = "computed_by_privte_behavior_v2"
        return aggregate

    def _build_behavior_v2_features(
        self,
        *,
        features: dict[str, Any],
        key_window_summary: dict[str, Any],
        quality: dict[str, Any],
        coarse_measurements: dict[str, Any],
    ) -> dict[str, Any]:
        direct_interaction = self._direct_interaction_level(features)
        continuity = self._screen_continuity_level(features)
        confounding = self._motion_confounding_level(features)
        absence_level = self._absence_evidence_level(features)
        return {
            "schema_note": (
                "Behavior v2 separates observability, direct interaction evidence, "
                "continuity evidence, absence evidence, and motion confounders."
            ),
            "device_observability_level": features.get("device_visible_ratio_bin"),
            "direct_interaction_evidence_level": direct_interaction,
            "screen_continuity_evidence_level": continuity,
            "absence_of_repetitive_operation_evidence_level": absence_level,
            "motion_confounding_level": confounding,
            "event_window_types": key_window_summary.get("event_window_types", {}),
            "coarse_behavior_measurements": coarse_measurements,
            "quality_overall": quality.get("overall"),
            "risk_strength_calibration": self._calibrated_signal_strength(
                features,
                quality.get("overall"),
            ),
        }

    def _build_concrete_behavior_evidence(
        self,
        *,
        features: dict[str, Any],
        key_window_summary: dict[str, Any],
        quality: dict[str, Any],
        coarse_measurements: dict[str, Any],
    ) -> list[dict[str, Any]]:
        event_types = key_window_summary.get("event_window_types", {})
        event_text = (
            ", ".join(f"{key}={value}" for key, value in event_types.items())
            if event_types
            else "none"
        )
        sampled = coarse_measurements.get("sampled_frame_count", "unknown")
        device = coarse_measurements.get("device_visible", {})
        stable = coarse_measurements.get("stable_screen_engagement", {})
        hand = coarse_measurements.get("hand_device_interaction", {})
        device_activity = coarse_measurements.get("device_region_activity", {})
        repetitive = coarse_measurements.get("repetitive_operation", {})
        confounder = coarse_measurements.get("motion_confounder", {})
        return [
            {
                "evidence_type": "device_or_screen_presence",
                "observation": (
                    "抽样帧中设备/屏幕样区域可见水平为 "
                    f"{features.get('device_visible_ratio_bin', 'unknown')}；"
                    f"约 {device.get('sampled_frame_hits', 'unknown')} 个抽样帧可见，"
                    f"占抽样帧 {device.get('sampled_frame_share_percent', 'unknown')}。"
                ),
                "measurement": {
                    "sampled_frames": sampled,
                    "device_visible_sampled_frames": device.get("sampled_frame_hits"),
                    "device_visible_share": device.get("sampled_frame_share_percent"),
                    "device_visible_share_range": device.get("sampled_frame_share_range"),
                },
                "behavior_meaning": (
                    "这说明视频里存在可观察的数字设备使用场景，但它本身不是风险证据。"
                ),
                "privacy_filter": "不输出图像、屏幕内容、设备位置坐标或应用名称。",
                "risk_relevance": "作为后续交互和持续参与证据的可观察性前提。",
            },
            {
                "evidence_type": "stable_screen_participation_episode",
                "observation": (
                    "稳定屏幕参与代理为 "
                    f"{features.get('stable_screen_engagement_proxy_ratio_bin', 'unknown')}；"
                    f"约 {stable.get('sampled_frame_hits', 'unknown')} 个抽样帧满足稳定参与条件，"
                    f"占抽样帧 {stable.get('sampled_frame_share_percent', 'unknown')}；"
                    "最长连续稳定片段为 "
                    f"{features.get('max_continuous_stable_engagement_bin', 'unknown')}，"
                    f"约 {stable.get('max_consecutive_sampled_frames', 'unknown')} 个连续抽样帧。"
                ),
                "measurement": {
                    "stable_sampled_frames": stable.get("sampled_frame_hits"),
                    "stable_share": stable.get("sampled_frame_share_percent"),
                    "stable_share_range": stable.get("sampled_frame_share_range"),
                    "max_consecutive_sampled_frames": stable.get(
                        "max_consecutive_sampled_frames"
                    ),
                    "stable_event_positions": stable.get("selected_event_positions"),
                },
                "behavior_meaning": (
                    "该证据表示设备可见且画面相对稳定的片段，不等同于精确注视或注意力。"
                ),
                "privacy_filter": "只保留早/中/晚覆盖和粗粒度连续性，不保留精确时间戳。",
                "risk_relevance": "可支持持续观看或持续投入的代理判断。",
            },
            {
                "evidence_type": "direct_hand_device_interaction",
                "observation": (
                    "手部可见水平为 "
                    f"{features.get('hand_visible_ratio_bin', 'unknown')}；"
                    "手-设备接近代理为 "
                    f"{features.get('hand_device_proximity_ratio_bin', 'unknown')}；"
                    "活跃手-设备交互代理为 "
                    f"{features.get('active_hand_device_interaction_proxy_ratio_bin', 'unknown')}；"
                    f"约 {hand.get('active_interaction_sampled_frames', 'unknown')} 个抽样帧"
                    "存在手部接近或设备区域活动。"
                ),
                "measurement": {
                    "hand_visible_sampled_frames": hand.get("hand_visible_sampled_frames"),
                    "hand_device_proximity_sampled_frames": hand.get(
                        "hand_device_proximity_sampled_frames"
                    ),
                    "active_interaction_sampled_frames": hand.get(
                        "active_interaction_sampled_frames"
                    ),
                    "active_interaction_share": hand.get(
                        "active_interaction_share_percent"
                    ),
                    "active_interaction_share_range": hand.get(
                        "active_interaction_share_range"
                    ),
                    "selected_hand_interaction_events": hand.get(
                        "selected_hand_interaction_events"
                    ),
                },
                "behavior_meaning": (
                    "这是比单纯设备可见更直接的操作线索，但仍不是已验证触摸事件。"
                ),
                "privacy_filter": "不输出手部关键点序列、坐标或截图。",
                "risk_relevance": "可支持点击、滑动或拿取设备等操作活跃性的代理判断。",
            },
            {
                "evidence_type": "device_region_local_activity",
                "observation": (
                    "设备区域局部活动代理为 "
                    f"{features.get('device_region_activity_proxy_ratio_bin', 'unknown')}；"
                    f"约 {device_activity.get('sampled_activity_points', 'unknown')} 个抽样点"
                    "出现设备区域局部活动；"
                    "入选关键事件中设备区域活动事件数量为 "
                    f"{device_activity.get('selected_device_activity_events', 'unknown')}。"
                ),
                "measurement": {
                    "device_region_activity_sampled_points": device_activity.get(
                        "sampled_activity_points"
                    ),
                    "device_region_activity_share": device_activity.get(
                        "activity_share_percent"
                    ),
                    "device_region_activity_share_range": device_activity.get(
                        "activity_share_range"
                    ),
                    "selected_device_activity_events": device_activity.get(
                        "selected_device_activity_events"
                    ),
                    "selected_event_positions": device_activity.get(
                        "selected_event_positions"
                    ),
                },
                "behavior_meaning": (
                    "该证据来自设备附近局部运动或画面变化，可能对应操作，也可能来自画面扰动。"
                ),
                "privacy_filter": "不输出屏幕 OCR、应用内容、图像或精确 ROI 坐标。",
                "risk_relevance": "作为交互活跃性的辅助证据，不能单独当作高风险证据。",
            },
            {
                "evidence_type": "repetitive_operation_or_absence",
                "observation": (
                    "重复操作代理窗口数量为 "
                    f"{features.get('repetitive_operation_proxy_count_bin', 'unknown')}；"
                    f"抽样范围内检测到 {repetitive.get('sampled_repetitive_points', 'unknown')} "
                    "个重复操作代理点。"
                ),
                "measurement": {
                    "repetitive_operation_sampled_points": repetitive.get(
                        "sampled_repetitive_points"
                    ),
                    "repetitive_operation_level": repetitive.get(
                        "sampled_repetitive_points_level"
                    ),
                    "selected_hand_interaction_events": repetitive.get(
                        "selected_hand_interaction_events"
                    ),
                },
                "behavior_meaning": (
                    "若该项较低，表示抽样范围内没有观察到足够重复操作代理证据。"
                ),
                "privacy_filter": "只输出粗粒度窗口数量，不输出动作轨迹。",
                "risk_relevance": "较高值支持频繁交互；较低值应降低风险强度确定性。",
            },
            {
                "evidence_type": "motion_confounder_check",
                "observation": (
                    "姿态/场景变化代理为 "
                    f"{features.get('posture_or_context_change_count_bin', 'unknown')}；"
                    "全局运动水平为 "
                    f"{features.get('global_motion_level', 'unknown')}；"
                    f"姿态/场景变化代理点为 {confounder.get('posture_or_context_change_points', 'unknown')}；"
                    f"关键事件类型计数为 {event_text}。"
                ),
                "measurement": {
                    "posture_or_context_change_points": confounder.get(
                        "posture_or_context_change_points"
                    ),
                    "global_motion_level": confounder.get("global_motion_level"),
                    "selected_global_motion_events": confounder.get(
                        "selected_global_motion_events"
                    ),
                    "selected_posture_or_context_events": confounder.get(
                        "selected_posture_or_context_events"
                    ),
                    "selected_event_positions": confounder.get("selected_event_positions"),
                },
                "behavior_meaning": (
                    "该证据用于判断设备附近运动是否可能被身体移动、镜头移动或场景扰动混淆。"
                ),
                "privacy_filter": "只输出运动类别和粗粒度强度，不输出场景描述或身份线索。",
                "risk_relevance": "混淆因素较强时，应降低对交互证据的确定性。",
            },
        ]

    def _build_coarse_measurements(
        self,
        *,
        features: dict[str, Any],
        quality: dict[str, Any],
        event_windows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        sampled_frames = int(quality.get("sampled_frame_count") or 0)
        event_type_counts = Counter(str(event.get("event_type")) for event in event_windows)
        event_position_counts = Counter(
            str(event.get("relative_position"))
            for event in event_windows
            if event.get("relative_position")
        )

        device = self._ratio_measurement(
            features.get("device_visible_ratio"),
            sampled_frames,
        )
        stable = self._ratio_measurement(
            features.get("stable_screen_engagement_proxy_ratio"),
            sampled_frames,
        )
        hand_visible = self._ratio_measurement(
            features.get("hand_visible_ratio"),
            sampled_frames,
        )
        hand_device = self._ratio_measurement(
            features.get("hand_device_proximity_ratio"),
            sampled_frames,
        )
        active = self._ratio_measurement(
            features.get("active_hand_device_interaction_proxy_ratio"),
            sampled_frames,
        )
        device_activity = self._ratio_measurement(
            features.get("device_region_activity_proxy_ratio"),
            sampled_frames,
        )

        return {
            "sampled_frame_count": sampled_frames,
            "event_position_distribution": dict(sorted(event_position_counts.items())),
            "device_visible": {
                "sampled_frame_hits": device["hits"],
                "sampled_frame_share_percent": device["percent"],
                "sampled_frame_share_range": device["range"],
            },
            "stable_screen_engagement": {
                "sampled_frame_hits": stable["hits"],
                "sampled_frame_share_percent": stable["percent"],
                "sampled_frame_share_range": stable["range"],
                "max_consecutive_sampled_frames": features.get(
                    "max_continuous_stable_engagement_frames",
                    0,
                ),
                "selected_event_positions": self._event_positions(
                    event_windows,
                    "stable_screen_engagement_proxy",
                ),
            },
            "hand_device_interaction": {
                "hand_visible_sampled_frames": hand_visible["hits"],
                "hand_visible_share_percent": hand_visible["percent"],
                "hand_device_proximity_sampled_frames": hand_device["hits"],
                "hand_device_proximity_share_percent": hand_device["percent"],
                "active_interaction_sampled_frames": active["hits"],
                "active_interaction_share_percent": active["percent"],
                "active_interaction_share_range": active["range"],
                "selected_hand_interaction_events": event_type_counts.get(
                    "hand_device_interaction_burst",
                    0,
                ),
            },
            "device_region_activity": {
                "sampled_activity_points": features.get(
                    "device_region_activity_proxy_count",
                    0,
                ),
                "activity_share_percent": device_activity["percent"],
                "activity_share_range": device_activity["range"],
                "selected_device_activity_events": event_type_counts.get(
                    "device_region_motion_burst",
                    0,
                ),
                "selected_event_positions": self._event_positions(
                    event_windows,
                    "device_region_motion_burst",
                ),
            },
            "repetitive_operation": {
                "sampled_repetitive_points": features.get(
                    "repetitive_operation_proxy_count",
                    0,
                ),
                "sampled_repetitive_points_level": features.get(
                    "repetitive_operation_proxy_count_bin",
                    "unknown",
                ),
                "selected_hand_interaction_events": event_type_counts.get(
                    "hand_device_interaction_burst",
                    0,
                ),
            },
            "motion_confounder": {
                "posture_or_context_change_points": features.get(
                    "posture_or_context_change_count",
                    0,
                ),
                "global_motion_level": features.get("global_motion_level", "unknown"),
                "selected_global_motion_events": event_type_counts.get(
                    "global_motion_burst",
                    0,
                ),
                "selected_posture_or_context_events": event_type_counts.get(
                    "posture_or_context_motion_burst",
                    0,
                ),
                "selected_event_positions": self._positions_for_event_types(
                    event_windows,
                    {
                        "global_motion_burst",
                        "posture_or_context_motion_burst",
                    },
                ),
            },
        }

    def _ratio_measurement(self, ratio: Any, sampled_frames: int) -> dict[str, Any]:
        if ratio is None:
            return {"hits": "unknown", "percent": "unknown", "range": "unknown"}
        value = float(ratio)
        hits = int(round(value * sampled_frames)) if sampled_frames > 0 else 0
        return {
            "hits": hits,
            "percent": f"{int(round(value * 100))}%",
            "range": self._ratio_range(value),
        }

    def _ratio_range(self, ratio: float) -> str:
        if ratio <= 0:
            return "0%"
        if ratio < 0.05:
            return "<5%"
        if ratio < 0.25:
            return "5-25%"
        if ratio < 0.6:
            return "25-60%"
        if ratio < 0.85:
            return "60-85%"
        return "85-100%"

    def _event_positions(
        self,
        event_windows: list[dict[str, Any]],
        event_type: str,
    ) -> dict[str, int]:
        return self._positions_for_event_types(event_windows, {event_type})

    def _positions_for_event_types(
        self,
        event_windows: list[dict[str, Any]],
        event_types: set[str],
    ) -> dict[str, int]:
        counter = Counter(
            str(event.get("relative_position"))
            for event in event_windows
            if event.get("event_type") in event_types and event.get("relative_position")
        )
        return dict(sorted(counter.items()))

    def _screen_continuity_level(self, features: dict[str, Any]) -> str:
        stable = str(features.get("stable_screen_engagement_proxy_ratio_bin", "unknown"))
        max_stable = str(features.get("max_continuous_stable_engagement_bin", "unknown"))
        if stable in {"high", "very_high"} or max_stable == "high":
            return "strong"
        if stable == "medium" or max_stable == "medium":
            return "moderate"
        if stable in {"low", "none"} and max_stable in {"low", "none"}:
            return "limited"
        return "uncertain"

    def _direct_interaction_level(self, features: dict[str, Any]) -> str:
        hand_interaction = str(
            features.get("active_hand_device_interaction_proxy_ratio_bin", "unknown")
        )
        repetitive = str(features.get("repetitive_operation_proxy_count_bin", "unknown"))
        device_activity = str(
            features.get("device_region_activity_proxy_count_bin", "unknown")
        )
        if hand_interaction in {"high", "very_high"} or repetitive == "high":
            return "strong_direct_or_repetitive_proxy"
        if hand_interaction == "medium" or repetitive == "medium":
            return "moderate_direct_proxy"
        if device_activity in {"high", "very_high"}:
            return "indirect_device_region_activity_only"
        if device_activity == "medium":
            return "limited_indirect_activity"
        return "limited_direct_interaction_evidence"

    def _motion_confounding_level(self, features: dict[str, Any]) -> str:
        posture = str(features.get("posture_or_context_change_count_bin", "unknown"))
        global_motion = str(features.get("global_motion_level", "unknown"))
        if posture in {"high", "very_high"} or global_motion in {"high", "very_high"}:
            return "high_potential_confounding"
        if posture == "medium" or global_motion in {"elevated", "medium"}:
            return "moderate_potential_confounding"
        return "low_or_unclear_confounding"

    def _absence_evidence_level(self, features: dict[str, Any]) -> str:
        repetitive = str(features.get("repetitive_operation_proxy_count_bin", "unknown"))
        hand_interaction = str(
            features.get("active_hand_device_interaction_proxy_ratio_bin", "unknown")
        )
        stable = str(features.get("stable_screen_engagement_proxy_ratio_bin", "unknown"))
        if repetitive in {"none", "low"} and hand_interaction in {"none", "low"}:
            if stable in {"none", "low"}:
                return "no_clear_sustained_or_repetitive_proxy_observed"
            return "no_clear_repetitive_operation_proxy_observed"
        return "limited_absence_evidence"

    def _calibrated_signal_strength(self, features: dict[str, Any], quality: str | None) -> str:
        if quality not in {"usable_behavior_frame_quality", "usable_frame_quality"}:
            return "insufficient_video_proxy_signal"

        device = str(features.get("device_visible_ratio_bin", "unknown"))
        stable = str(features.get("stable_screen_engagement_proxy_ratio_bin", "unknown"))
        max_stable = str(features.get("max_continuous_stable_engagement_bin", "unknown"))
        hand_interaction = str(
            features.get("active_hand_device_interaction_proxy_ratio_bin", "unknown")
        )
        device_activity = str(
            features.get("device_region_activity_proxy_count_bin", "unknown")
        )
        repetitive = str(features.get("repetitive_operation_proxy_count_bin", "unknown"))
        confounding = self._motion_confounding_level(features)

        observable = device in {"medium", "high", "very_high"}
        continuity = stable in {"medium", "high", "very_high"} or max_stable in {
            "medium",
            "high",
        }
        direct_interaction = hand_interaction in {"medium", "high", "very_high"}
        repeated = repetitive in {"medium", "high"}
        indirect_activity = device_activity in {"medium", "high", "very_high"}
        high_confounding = confounding == "high_potential_confounding"

        if not observable:
            return "weak_video_proxy_signal"
        if continuity and direct_interaction and repeated and not high_confounding:
            return "strong_video_proxy_signal"
        if continuity and (direct_interaction or repeated) and not high_confounding:
            return "moderate_video_proxy_signal"
        if continuity and indirect_activity and not high_confounding:
            return "mild_video_proxy_signal"
        if indirect_activity or continuity or direct_interaction:
            return "mild_video_proxy_signal"
        return "weak_video_proxy_signal"

    def _build_visual_proxy_lines(self, aggregate: dict[str, Any]) -> list[str]:
        features = aggregate["behavior_v1_features"]
        v2_features = aggregate.get("behavior_v2_features", {})
        event_types = aggregate["key_window_summary"]["event_window_types"]
        event_summary = (
            ", ".join(f"{key}={value}" for key, value in event_types.items())
            if event_types
            else "none"
        )
        quality = aggregate["quality_summary"]
        return [
            "PriVTE-Behavior v2分析状态: computed_by_privte_behavior_v2",
            f"设备/屏幕可观察性: {features['device_visible_ratio_bin']}",
            f"稳定屏幕参与片段: {v2_features.get('screen_continuity_evidence_level', 'unknown')}",
            f"直接交互证据: {v2_features.get('direct_interaction_evidence_level', 'unknown')}",
            f"重复操作缺失/存在证据: {v2_features.get('absence_of_repetitive_operation_evidence_level', 'unknown')}",
            f"设备区域局部活动: {features['device_region_activity_proxy_count_bin']}",
            f"运动混淆检查: {v2_features.get('motion_confounding_level', 'unknown')}",
            f"关键事件窗口类型: {event_summary}",
            f"质量总评: {quality['overall']}",
        ]
