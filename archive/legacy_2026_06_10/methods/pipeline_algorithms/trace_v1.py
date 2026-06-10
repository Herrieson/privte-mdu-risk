"""PriVTE-Trace v1 evidence extractor.

Trace v1 reuses the practical Behavior v3 perception backbone, then replaces
the episode-first output emphasis with a normal-use-aware trajectory summary.
The goal is to separate ordinary device use from risk-relevant use patterns
before rendering text for the LLM.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .behavior_v3_temporal import PriVTEBehaviorV3TemporalExtractor


class PriVTETraceV1Extractor(PriVTEBehaviorV3TemporalExtractor):
    """Trace v1: calibrated trajectory evidence over temporal behavior episodes."""

    name = "privte_trace_v1"
    version = "trace_v1"
    feature_schema_version = "privte_trace_v1_schema.v0"

    RISK_STATES = {
        "active_hand_device_operation",
        "repetitive_operation",
    }
    ENGAGEMENT_STATES = RISK_STATES | {"passive_screen_engagement"}

    def _aggregate_frames(self, **kwargs: Any) -> dict[str, Any]:
        aggregate = super()._aggregate_frames(**kwargs)
        trace_summary = self._build_trace_risk_summary(aggregate)
        aggregate["privte_trace_features"] = {
            "schema_note": (
                "Trace v1 converts privacy-filtered temporal behavior states into "
                "normal-use-aware trajectory evidence."
            ),
            "risk_semantics": {
                "ordinary_use_is_not_automatically_risk": True,
                "risk_requires_intensity_continuity_repetition_or_phase_consistency": True,
                "screen_presence_alone_is_observability_not_risk": True,
            },
            "trace_risk_summary": trace_summary,
            "privacy_policy": {
                "no_raw_frames_or_images": True,
                "no_exact_coordinates": True,
                "no_exact_timestamps": True,
                "no_screen_content_or_app_names": True,
                "no_free_form_scene_or_appearance_description": True,
                "coarse_trajectory_metrics_only": True,
            },
        }
        aggregate["trace_risk_summary"] = trace_summary
        aggregate["trace_behavior_narrative"] = self._build_trace_narrative(trace_summary)
        privacy_summary = aggregate.get("privacy_preserving_behavior_summary", {})
        privacy_summary.update(
            {
                "summary_type": "normal_use_aware_trace_summary",
                "normal_use_not_automatically_risk": True,
                "coarse_trajectory_metrics_only": True,
                "selected_episode_sequence_retained_as_supporting_detail": True,
            }
        )
        aggregate["privacy_preserving_behavior_summary"] = privacy_summary
        aggregate["quality_summary"]["status"] = "computed_by_privte_trace_v1"
        return aggregate

    def _build_trace_risk_summary(self, aggregate: dict[str, Any]) -> dict[str, Any]:
        temporal_summary = aggregate.get("temporal_sequence_summary", {})
        temporal_sequence = aggregate.get("temporal_behavior_sequence", [])
        behavior_v1 = aggregate.get("behavior_v1_features", {})
        quality = aggregate.get("quality_summary", {})

        total_points = int(aggregate.get("sampled_frame_count") or 0)
        if not total_points:
            total_points = sum(
                int(value or 0)
                for value in temporal_summary.get("frame_state_counts", {}).values()
            )
        total = max(total_points, 1)

        stable_points = int(temporal_summary.get("stable_screen_point_count") or 0)
        active_points = int(temporal_summary.get("active_operation_point_count") or 0)
        direct_points = int(temporal_summary.get("direct_operation_point_count") or 0)
        repetitive_points = int(
            temporal_summary.get("repetitive_operation_point_count") or 0
        )
        confounded_points = int(
            temporal_summary.get("confounded_activity_point_count") or 0
        )
        visible_only_points = int(
            temporal_summary.get("visible_without_engagement_point_count") or 0
        )
        device_visible_ratio = self._float_or_none(behavior_v1.get("device_visible_ratio"))
        hand_visible_ratio = self._float_or_none(behavior_v1.get("hand_visible_ratio"))
        hand_device_ratio = self._float_or_none(
            behavior_v1.get("hand_device_proximity_ratio")
        )
        device_activity_ratio = self._float_or_none(
            behavior_v1.get("device_region_activity_proxy_ratio")
        )
        stable_share = stable_points / total
        active_share = active_points / total
        direct_share = direct_points / total
        repetitive_share = repetitive_points / total
        confounded_share = confounded_points / total
        visible_only_share = visible_only_points / total

        phase_counts = self._phase_counts(temporal_sequence)
        risk_phase_count = sum(1 for value in phase_counts["risk"].values() if value > 0)
        engagement_phase_count = sum(
            1 for value in phase_counts["engagement"].values() if value > 0
        )

        score_parts = self._trace_score_parts(
            stable_share=stable_share,
            active_share=active_share,
            direct_points=direct_points,
            repetitive_points=repetitive_points,
            risk_phase_count=risk_phase_count,
            engagement_phase_count=engagement_phase_count,
            longest_stable_bin=str(
                temporal_summary.get("longest_stable_engagement_bin", "unknown")
            ),
            confounded_share=confounded_share,
        )
        raw_score = sum(part["value"] for part in score_parts)
        risk_score = max(0, min(10, raw_score))
        quality_overall = str(quality.get("overall", "unknown"))
        strength = self._trace_strength(
            risk_score=risk_score,
            quality_overall=quality_overall,
            total_points=total_points,
        )
        ordinary_pattern = self._ordinary_use_pattern(
            strength=strength,
            stable_share=stable_share,
            active_share=active_share,
            repetitive_points=repetitive_points,
            visible_only_share=visible_only_share,
            confounded_share=confounded_share,
            device_visible_ratio=device_visible_ratio,
        )
        risk_indicators, ordinary_indicators, counter_indicators = (
            self._trace_indicator_text(
                total_points=total_points,
                stable_share=stable_share,
                active_share=active_share,
                direct_points=direct_points,
                repetitive_points=repetitive_points,
                confounded_share=confounded_share,
                visible_only_share=visible_only_share,
                risk_phase_count=risk_phase_count,
                engagement_phase_count=engagement_phase_count,
                device_visible_ratio=device_visible_ratio,
                hand_visible_ratio=hand_visible_ratio,
                hand_device_ratio=hand_device_ratio,
                device_activity_ratio=device_activity_ratio,
                longest_stable_bin=str(
                    temporal_summary.get("longest_stable_engagement_bin", "unknown")
                ),
            )
        )
        return {
            "summary_type": "normal_use_aware_behavior_trace",
            "risk_pattern_strength": strength,
            "risk_pattern_score": risk_score,
            "risk_pattern_score_parts": score_parts,
            "ordinary_use_pattern": ordinary_pattern,
            "quality_overall": quality_overall,
            "sampled_points": total_points,
            "coarse_rates": {
                "device_visible_share": self._rounded(device_visible_ratio),
                "stable_engagement_share": self._rounded(stable_share),
                "active_operation_share": self._rounded(active_share),
                "direct_operation_share": self._rounded(direct_share),
                "repetitive_operation_share": self._rounded(repetitive_share),
                "visible_without_engagement_share": self._rounded(visible_only_share),
                "confounded_activity_share": self._rounded(confounded_share),
            },
            "coarse_levels": {
                "device_observability": self._share_level(device_visible_ratio),
                "stable_engagement_burden": self._share_level(stable_share),
                "active_operation_density": self._activity_level(active_share),
                "direct_operation_density": self._point_level(direct_points),
                "repetition_density": self._point_level(repetitive_points),
                "visible_only_context": self._share_level(visible_only_share),
                "motion_confounding": self._activity_level(confounded_share),
                "phase_consistency": self._phase_consistency_level(risk_phase_count),
                "engagement_phase_coverage": self._phase_consistency_level(
                    engagement_phase_count
                ),
            },
            "phase_counts": phase_counts,
            "risk_pattern_indicators": risk_indicators,
            "ordinary_or_low_risk_indicators": ordinary_indicators,
            "counter_risk_indicators": counter_indicators,
            "temporal_summary_link": {
                "main_observation_pattern": temporal_summary.get(
                    "main_observation_pattern"
                ),
                "dominant_states": temporal_summary.get("dominant_states", []),
                "selected_episode_count": temporal_summary.get("selected_episode_count"),
                "candidate_episode_count": temporal_summary.get(
                    "candidate_episode_count"
                ),
            },
        }

    def _trace_score_parts(
        self,
        *,
        stable_share: float,
        active_share: float,
        direct_points: int,
        repetitive_points: int,
        risk_phase_count: int,
        engagement_phase_count: int,
        longest_stable_bin: str,
        confounded_share: float,
    ) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []

        def add(name: str, value: int, evidence: str) -> None:
            parts.append({"name": name, "value": value, "evidence": evidence})

        if stable_share >= float(self.config.get("trace_stable_share_high", 0.5)):
            add("stable_engagement_burden", 2, "stable_share>=high_threshold")
        elif stable_share >= float(self.config.get("trace_stable_share_medium", 0.25)):
            add("stable_engagement_burden", 1, "stable_share>=medium_threshold")

        if active_share >= float(self.config.get("trace_active_share_high", 0.14)):
            add("active_operation_density", 2, "active_share>=high_threshold")
        elif active_share >= float(self.config.get("trace_active_share_medium", 0.06)):
            add("active_operation_density", 1, "active_share>=medium_threshold")

        if direct_points >= int(self.config.get("trace_direct_points_high", 8)):
            add("direct_operation_points", 2, "direct_points>=high_threshold")
        elif direct_points >= int(self.config.get("trace_direct_points_medium", 3)):
            add("direct_operation_points", 1, "direct_points>=medium_threshold")

        if repetitive_points >= int(self.config.get("trace_repetitive_points_high", 7)):
            add("repetitive_operation_points", 2, "repetitive_points>=high_threshold")
        elif repetitive_points >= int(
            self.config.get("trace_repetitive_points_medium", 3)
        ):
            add(
                "repetitive_operation_points",
                1,
                "repetitive_points>=medium_threshold",
            )

        if risk_phase_count >= 3:
            add("risk_phase_consistency", 2, "risk_states_seen_in_all_phase_bins")
        elif risk_phase_count >= 2:
            add("risk_phase_consistency", 1, "risk_states_seen_in_multiple_phase_bins")

        if engagement_phase_count >= 3:
            add("engagement_phase_coverage", 1, "engagement_seen_in_all_phase_bins")

        if longest_stable_bin == "high":
            add("longest_stable_engagement", 1, "longest_stable_bin=high")

        if confounded_share >= float(self.config.get("trace_confounded_share_high", 0.18)):
            add("motion_confounding_penalty", -2, "confounded_share>=high_threshold")
        elif confounded_share >= float(
            self.config.get("trace_confounded_share_medium", 0.1)
        ):
            add("motion_confounding_penalty", -1, "confounded_share>=medium_threshold")
        return parts

    def _trace_strength(
        self,
        *,
        risk_score: int,
        quality_overall: str,
        total_points: int,
    ) -> str:
        min_points = int(self.config.get("trace_min_sampled_points", 40))
        if quality_overall not in {
            "usable_behavior_frame_quality",
            "usable_frame_quality",
            "partial_behavior_frame_quality",
        }:
            return "insufficient_observability"
        if total_points < min_points:
            return "insufficient_observability"
        if risk_score >= 7:
            return "strong_risk_pattern"
        if risk_score >= 4:
            return "moderate_risk_pattern"
        if risk_score >= 2:
            return "mild_risk_pattern"
        return "no_clear_risk_pattern"

    def _ordinary_use_pattern(
        self,
        *,
        strength: str,
        stable_share: float,
        active_share: float,
        repetitive_points: int,
        visible_only_share: float,
        confounded_share: float,
        device_visible_ratio: float | None,
    ) -> str:
        if strength == "insufficient_observability":
            return "insufficient_observability"
        if strength in {"moderate_risk_pattern", "strong_risk_pattern"}:
            return "risk_pattern_observed"
        if (
            (device_visible_ratio or 0.0) >= 0.2
            and active_share < 0.05
            and repetitive_points == 0
            and stable_share < 0.35
        ):
            return "ordinary_or_low_intensity_device_use"
        if visible_only_share >= 0.25 and active_share < 0.06 and repetitive_points == 0:
            return "device_visible_without_risk_pattern"
        if confounded_share >= 0.1 and active_share < 0.08:
            return "motion_confounded_low_certainty_use"
        return "limited_risk_pattern_evidence"

    def _trace_indicator_text(
        self,
        *,
        total_points: int,
        stable_share: float,
        active_share: float,
        direct_points: int,
        repetitive_points: int,
        confounded_share: float,
        visible_only_share: float,
        risk_phase_count: int,
        engagement_phase_count: int,
        device_visible_ratio: float | None,
        hand_visible_ratio: float | None,
        hand_device_ratio: float | None,
        device_activity_ratio: float | None,
        longest_stable_bin: str,
    ) -> tuple[list[str], list[str], list[str]]:
        risk: list[str] = []
        ordinary: list[str] = []
        counter: list[str] = []

        if stable_share >= 0.5:
            risk.append(
                f"稳定屏幕参与占抽样点约 {self._percent_text(stable_share)}，处于高持续负荷。"
            )
        elif stable_share >= 0.25 and (
            active_share >= 0.06 or direct_points or repetitive_points
        ):
            risk.append(
                f"稳定屏幕参与占抽样点约 {self._percent_text(stable_share)}，并伴随交互或重复指标。"
            )
        else:
            ordinary.append(
                f"稳定屏幕参与占抽样点约 {self._percent_text(stable_share)}，未单独构成风险性模式。"
            )
        if active_share >= 0.06:
            risk.append(f"主动操作占抽样点约 {self._percent_text(active_share)}。")
        elif active_share > 0:
            ordinary.append(
                f"主动操作占抽样点约 {self._percent_text(active_share)}，密度较低。"
            )
        if direct_points:
            risk.append(f"直接手-设备操作指标点数={direct_points}。")
        if repetitive_points:
            risk.append(f"重复操作指标点数={repetitive_points}。")
        if risk_phase_count >= 2:
            risk.append(f"主动或重复操作跨 {risk_phase_count} 个早/中/晚阶段出现。")
        if engagement_phase_count >= 3 and (stable_share >= 0.35 or risk_phase_count):
            risk.append("屏幕参与行为覆盖 early、middle、late 三个阶段。")
        elif engagement_phase_count >= 3:
            ordinary.append("屏幕参与覆盖 early、middle、late，但未伴随跨阶段主动或重复操作。")
        if longest_stable_bin == "high":
            risk.append("最长连续稳定参与片段处于 high 档。")

        if not risk:
            risk.append("未形成明确风险性使用轨迹指标。")

        if device_visible_ratio is not None:
            ordinary.append(
                f"设备可观察比例约 {self._percent_text(device_visible_ratio)}，"
                "该项只说明可观察性，不单独构成风险。"
            )
        if visible_only_share >= 0.2:
            ordinary.append(
                f"设备可见但未形成参与或操作的抽样点占约 {self._percent_text(visible_only_share)}。"
            )
        if active_share < 0.06 and repetitive_points == 0:
            ordinary.append("主动操作密度低且未观察到重复操作，支持普通或低强度使用解释。")
        if hand_visible_ratio is not None and hand_visible_ratio < 0.05:
            ordinary.append("手部可见比例较低，直接操作证据应谨慎解释。")

        if repetitive_points == 0:
            counter.append("未观察到重复操作指标点。")
        if direct_points == 0:
            counter.append("未观察到直接手-设备操作指标点。")
        if risk_phase_count <= 1:
            counter.append("主动或重复操作没有形成跨阶段一致模式。")
        if confounded_share >= 0.1:
            counter.append(
                f"混淆活动占约 {self._percent_text(confounded_share)}，部分设备区域活动可能来自姿态或场景运动。"
            )
        if total_points <= 0:
            counter.append("缺少可用抽样点。")
        if hand_device_ratio is not None and hand_device_ratio < 0.05:
            counter.append("手-设备接近比例较低。")
        if device_activity_ratio is not None and device_activity_ratio < 0.05:
            counter.append("设备区域活动比例较低。")
        return risk, ordinary, counter

    def _build_trace_narrative(self, summary: dict[str, Any]) -> list[str]:
        levels = summary.get("coarse_levels", {})
        rates = summary.get("coarse_rates", {})
        return [
            (
                "Trace轨迹结论: "
                f"风险性使用轨迹强度={summary.get('risk_pattern_strength')}; "
                f"普通使用解释={summary.get('ordinary_use_pattern')}; "
                f"轨迹分数={summary.get('risk_pattern_score')}/10。"
            ),
            (
                "Trace全局比例: "
                f"稳定参与={rates.get('stable_engagement_share')}; "
                f"主动操作={rates.get('active_operation_share')}; "
                f"重复操作={rates.get('repetitive_operation_share')}; "
                f"设备可见但未参与/操作={rates.get('visible_without_engagement_share')}。"
            ),
            (
                "Trace分级指标: "
                f"稳定负荷={levels.get('stable_engagement_burden')}; "
                f"主动操作密度={levels.get('active_operation_density')}; "
                f"重复密度={levels.get('repetition_density')}; "
                f"阶段一致性={levels.get('phase_consistency')}; "
                f"混淆水平={levels.get('motion_confounding')}。"
            ),
        ]

    def _phase_counts(
        self,
        temporal_sequence: list[dict[str, Any]],
    ) -> dict[str, dict[str, int]]:
        phase_counts = {
            "engagement": Counter(),
            "risk": Counter(),
            "background": Counter(),
        }
        for step in temporal_sequence:
            phases = self._split_relative_position(str(step.get("relative_position", "")))
            state = str(step.get("state", "unknown"))
            bucket = (
                "risk"
                if state in self.RISK_STATES
                else "engagement"
                if state in self.ENGAGEMENT_STATES
                else "background"
            )
            for phase in phases:
                phase_counts[bucket][phase] += 1
        return {
            key: {phase: counter.get(phase, 0) for phase in ("early", "middle", "late")}
            for key, counter in phase_counts.items()
        }

    def _split_relative_position(self, value: str) -> list[str]:
        phases = [phase for phase in ("early", "middle", "late") if phase in value]
        return phases or ["unknown"]

    def _share_level(self, value: float | None) -> str:
        if value is None:
            return "unknown"
        if value <= 0:
            return "none"
        if value < 0.1:
            return "very_low"
        if value < 0.25:
            return "low"
        if value < 0.5:
            return "medium"
        if value < 0.75:
            return "high"
        return "very_high"

    def _activity_level(self, value: float | None) -> str:
        if value is None:
            return "unknown"
        if value <= 0:
            return "none"
        if value < 0.03:
            return "very_low"
        if value < 0.08:
            return "low"
        if value < 0.16:
            return "medium"
        return "high"

    def _point_level(self, points: int) -> str:
        if points <= 0:
            return "none"
        if points <= 2:
            return "low"
        if points <= 6:
            return "medium"
        return "high"

    def _phase_consistency_level(self, phase_count: int) -> str:
        if phase_count <= 0:
            return "none"
        if phase_count == 1:
            return "single_phase"
        if phase_count == 2:
            return "multi_phase"
        return "all_phases"

    def _float_or_none(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _rounded(self, value: float | None) -> float | None:
        if value is None:
            return None
        return round(value, 4)

    def _percent_text(self, value: float | None) -> str:
        if value is None:
            return "unknown"
        return f"{round(value * 100)}%"
