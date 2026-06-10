"""PriVTE Behavior v3 temporal evidence extractor.

Behavior v3 keeps the local perception backbone from Behavior v2, then converts
ordered sampled-frame proxies into a privacy-filtered temporal behavior episode
sequence. It is a controlled, task-specific video-to-text encoder, not an
open-ended video captioner.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .behavior_v1 import median_value
from .behavior_v2 import PriVTEBehaviorV2Extractor


class PriVTEBehaviorV3TemporalExtractor(PriVTEBehaviorV2Extractor):
    """Behavior v3: privacy-safe temporal behavior episode sequence."""

    name = "privte_behavior_v3_temporal"
    version = "v3_temporal"
    feature_schema_version = "privte_behavior_v3_temporal_schema.v0"

    STATE_PRIORITY = {
        "repetitive_operation": 90,
        "active_hand_device_operation": 80,
        "passive_screen_engagement": 70,
        "indirect_device_region_activity": 55,
        "confounded_device_region_activity": 45,
        "device_visible_only": 40,
        "posture_or_context_motion": 30,
        "no_device_observed": 20,
        "insufficient_quality": 10,
        "transition_or_uncertain": 0,
    }
    BACKGROUND_STATES = {
        "device_visible_only",
        "no_device_observed",
        "insufficient_quality",
        "transition_or_uncertain",
    }

    def _aggregate_frames(self, **kwargs: Any) -> dict[str, Any]:
        aggregate = super()._aggregate_frames(**kwargs)
        frame_records = list(kwargs.get("frame_records", []))
        temporal_points = self._build_temporal_points(frame_records)
        episode_candidates = self._build_episode_candidates(temporal_points)
        temporal_sequence = self._build_temporal_sequence(episode_candidates)
        sequence_summary = self._build_sequence_summary(
            temporal_points=temporal_points,
            episode_candidates=episode_candidates,
            temporal_sequence=temporal_sequence,
            aggregate=aggregate,
        )
        temporal_narrative = self._build_temporal_narrative(
            temporal_sequence,
            sequence_summary,
        )

        aggregate["behavior_v3_temporal_features"] = {
            "schema_note": (
                "Behavior v3 converts privacy-filtered visual proxy signals into "
                "a selected finite temporal behavior episode sequence."
            ),
            "state_vocabulary": list(self.STATE_PRIORITY),
            "sequence_type": "selected_behavior_episode_sequence",
            "sequence_summary": sequence_summary,
            "risk_strength_calibration": sequence_summary[
                "temporal_video_proxy_signal_strength"
            ],
            "privacy_policy": {
                "no_raw_frames_or_images": True,
                "no_exact_coordinates": True,
                "no_exact_timestamps": True,
                "relative_order_only": True,
                "no_screen_content_or_app_names": True,
                "no_free_form_scene_or_appearance_description": True,
            },
        }
        aggregate["temporal_behavior_sequence"] = temporal_sequence
        aggregate["temporal_behavior_narrative"] = temporal_narrative
        aggregate["temporal_sequence_summary"] = sequence_summary
        privacy_summary = aggregate.get("privacy_preserving_behavior_summary", {})
        privacy_summary.update(
            {
                "summary_type": "temporal_behavior_state_sequence_summary",
                "relative_order_only": True,
                "no_free_form_scene_or_appearance_description": True,
                "finite_behavior_state_vocabulary": True,
                "selected_behavior_episode_sequence": True,
            }
        )
        aggregate["privacy_preserving_behavior_summary"] = privacy_summary
        aggregate["quality_summary"]["status"] = "computed_by_privte_behavior_v3_temporal"
        return aggregate

    def _build_temporal_points(self, frame_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        thresholds = self._temporal_thresholds(frame_records)
        points: list[dict[str, Any]] = []
        for index, frame in enumerate(frame_records):
            flags = self._frame_temporal_flags(frame, thresholds)
            state = self._classify_frame_state(flags)
            points.append(
                {
                    "order": index,
                    "clip_order": int(frame.get("clip_order") or 0),
                    "relative_position": str(frame.get("relative_position") or "unknown"),
                    "state": state,
                    "flags": flags,
                }
            )
        self._mark_repetitive_operation_runs(points)
        return points

    def _mark_repetitive_operation_runs(self, points: list[dict[str, Any]]) -> None:
        repetitive_min_points = int(self.config.get("temporal_repetitive_min_points", 3))
        gap_tolerance = int(self.config.get("temporal_repetitive_gap_tolerance", 1))
        run: list[dict[str, Any]] = []
        gap_points: list[dict[str, Any]] = []
        previous_clip_order: int | None = None

        def flush_run() -> None:
            if len(run) >= repetitive_min_points:
                for item in run:
                    item["state"] = "repetitive_operation"
                    item["flags"]["repetitive_operation"] = True

        for point in points:
            clip_order = int(point.get("clip_order") or 0)
            active_candidate = bool(point["flags"].get("active_operation_candidate"))
            same_clip = previous_clip_order is None or clip_order == previous_clip_order
            if active_candidate and same_clip:
                gap_points = []
                run.append(point)
            elif run and same_clip and len(gap_points) < gap_tolerance:
                gap_points.append(point)
            else:
                flush_run()
                run = [point] if active_candidate else []
                gap_points = []
            previous_clip_order = clip_order
        flush_run()

    def _temporal_thresholds(self, frame_records: list[dict[str, Any]]) -> dict[str, float]:
        global_motion_values = [
            frame["global_motion"]
            for frame in frame_records
            if frame.get("global_motion") is not None
        ]
        near_motion_values = [
            frame["near_device_motion"]
            for frame in frame_records
            if frame.get("near_device_motion") is not None
        ]
        hand_motion_values = [
            frame["hand_motion"]
            for frame in frame_records
            if frame.get("hand_motion") is not None
        ]
        pose_motion_values = [
            frame["pose_motion"]
            for frame in frame_records
            if frame.get("pose_motion") is not None
        ]
        return {
            "global_motion": self._absolute_motion_threshold(
                median_value(global_motion_values)
            ),
            "near_device_motion": self._absolute_motion_threshold(
                median_value(near_motion_values)
            ),
            "hand_motion": self._relative_motion_threshold(
                median_value(hand_motion_values),
                minimum=float(self.config.get("hand_motion_min_spike_threshold", 0.06)),
            ),
            "pose_motion": self._relative_motion_threshold(
                median_value(pose_motion_values),
                minimum=float(self.config.get("pose_motion_min_spike_threshold", 0.04)),
            ),
        }

    def _frame_temporal_flags(
        self,
        frame: dict[str, Any],
        thresholds: dict[str, float],
    ) -> dict[str, Any]:
        quality_usable = bool(frame.get("quality_usable"))
        device_visible = bool(frame.get("device_visible"))
        hand_device_proximity = bool(frame.get("hand_device_proximity"))
        face_context = bool(frame.get("face_device_cooccurrence")) or bool(
            frame.get("face_device_alignment_proxy")
        )
        global_motion = frame.get("global_motion")
        near_device_motion = frame.get("near_device_motion")
        hand_motion = frame.get("hand_motion")
        pose_motion = frame.get("pose_motion")

        global_motion_burst = (
            global_motion is not None
            and float(global_motion) >= thresholds["global_motion"]
        )
        device_region_activity = (
            device_visible
            and near_device_motion is not None
            and float(near_device_motion) >= thresholds["near_device_motion"]
        )
        hand_motion_burst = (
            hand_motion is not None
            and float(hand_motion) >= thresholds["hand_motion"]
        )
        direct_hand_operation = (
            quality_usable
            and device_visible
            and hand_device_proximity
            and hand_motion_burst
        )
        posture_motion = (
            pose_motion is not None
            and float(pose_motion) >= thresholds["pose_motion"]
            and global_motion_burst
        )
        motion_confounding = device_region_activity and (
            posture_motion or global_motion_burst
        )
        unconfounded_device_region_activity = (
            device_region_activity and not motion_confounding
        )
        active_operation_candidate = direct_hand_operation or (
            quality_usable
            and hand_device_proximity
            and unconfounded_device_region_activity
        )
        stable_screen_engagement = (
            device_visible
            and quality_usable
            and not global_motion_burst
            and not active_operation_candidate
            and (face_context or hand_device_proximity)
        )
        return {
            "quality_usable": quality_usable,
            "device_visible": device_visible,
            "hand_device_proximity": hand_device_proximity,
            "face_device_context": face_context,
            "stable_screen_engagement": stable_screen_engagement,
            "device_region_activity": device_region_activity,
            "unconfounded_device_region_activity": unconfounded_device_region_activity,
            "direct_hand_operation": direct_hand_operation,
            "active_operation_candidate": active_operation_candidate,
            "global_motion_burst": global_motion_burst,
            "posture_or_context_motion": posture_motion,
            "motion_confounding": motion_confounding,
            "repetitive_operation": False,
        }

    def _classify_frame_state(self, flags: dict[str, Any]) -> str:
        if not flags["quality_usable"]:
            return "insufficient_quality"
        if not flags["device_visible"]:
            if flags["posture_or_context_motion"] or flags["global_motion_burst"]:
                return "posture_or_context_motion"
            return "no_device_observed"
        if flags["active_operation_candidate"]:
            return "active_hand_device_operation"
        if flags["motion_confounding"]:
            return "confounded_device_region_activity"
        if flags["unconfounded_device_region_activity"]:
            return "indirect_device_region_activity"
        if flags["stable_screen_engagement"]:
            return "passive_screen_engagement"
        return "device_visible_only"

    def _build_episode_candidates(
        self,
        temporal_points: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not temporal_points:
            return []
        candidates: list[dict[str, Any]] = []
        current: list[dict[str, Any]] = []

        def flush() -> None:
            if not current:
                return
            state_counts = Counter(point["state"] for point in current)
            state = state_counts.most_common(1)[0][0]
            flags = [point["flags"] for point in current]
            metrics = self._window_metrics(flags)
            candidates.append(
                {
                    "state": state,
                    "points": list(current),
                    "state_counts": dict(sorted(state_counts.items())),
                    "metrics": metrics,
                    "start_order": int(current[0]["order"]),
                    "end_order": int(current[-1]["order"]),
                    "clip_order": int(current[0].get("clip_order") or 0),
                    "relative_position": self._combine_positions(
                        [point["relative_position"] for point in current]
                    ),
                }
            )
            current.clear()

        for point in temporal_points:
            if not current:
                current.append(point)
                continue
            same_state = point["state"] == current[-1]["state"]
            same_clip = int(point.get("clip_order") or 0) == int(
                current[-1].get("clip_order") or 0
            )
            if same_state and same_clip:
                current.append(point)
            else:
                flush()
                current.append(point)
        flush()
        return candidates

    def _build_temporal_sequence(
        self,
        episode_candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        max_steps = int(self.config.get("max_temporal_steps", 9))
        selected = self._select_episode_candidates(episode_candidates, max_steps)
        sequence: list[dict[str, Any]] = []
        previous_state = None
        for order, candidate in enumerate(selected, start=1):
            state = candidate["state"]
            metrics = candidate["metrics"]
            transition = (
                "sequence_start"
                if previous_state is None
                else (
                    "state_continues"
                    if previous_state == state
                    else f"{previous_state}_to_{state}"
                )
            )
            sequence.append(
                {
                    "step_id": f"episode_{order:02d}",
                    "order": order,
                    "sequence_type": "selected_behavior_episode",
                    "relative_position": candidate["relative_position"],
                    "state": state,
                    "duration_bin": self._duration_bin(
                        int(metrics.get("sampled_points") or 0)
                    ),
                    "sampled_points": int(metrics.get("sampled_points") or 0),
                    "state_counts": candidate["state_counts"],
                    "evidence": metrics,
                    "confidence": self._state_confidence(state, metrics),
                    "support_level": self._state_support_level(state, metrics),
                    "transition": transition,
                    "selection_reason": self._episode_selection_reason(candidate),
                    "evidence_summary": self._state_evidence_summary(state, metrics),
                    "risk_relevance": self._state_risk_relevance(state),
                    "privacy_filter": (
                        "relative order and coarse episode metrics only; no frames, "
                        "coordinates, OCR, app names, or exact timestamps."
                    ),
                }
            )
            previous_state = state
        return sequence

    def _select_episode_candidates(
        self,
        candidates: list[dict[str, Any]],
        max_steps: int,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        max_steps = max(1, max_steps)
        if len(candidates) <= max_steps:
            return candidates

        max_background_steps = int(self.config.get("temporal_max_background_steps", 2))
        key_candidates = [
            candidate
            for candidate in candidates
            if candidate["state"] not in self.BACKGROUND_STATES
        ]
        if not key_candidates:
            key_candidates = candidates
            max_background_steps = max_steps

        ranked_key = sorted(
            key_candidates,
            key=self._episode_score,
            reverse=True,
        )
        selected = ranked_key[:max_steps]

        if len(selected) < max_steps and max_background_steps > 0:
            selected_ids = {id(candidate) for candidate in selected}
            background = [
                candidate
                for candidate in candidates
                if id(candidate) not in selected_ids
                and candidate["state"] in self.BACKGROUND_STATES
            ]
            background = sorted(background, key=self._episode_score, reverse=True)
            free_slots = min(max_steps - len(selected), max_background_steps)
            selected.extend(background[:free_slots])

        return sorted(selected, key=lambda item: item["start_order"])

    def _episode_score(self, candidate: dict[str, Any]) -> float:
        metrics = candidate["metrics"]
        state = candidate["state"]
        sampled_points = float(metrics.get("sampled_points") or 0)
        stable = float(metrics.get("stable_screen_points") or 0)
        active = float(metrics.get("active_hand_device_points") or 0)
        repetitive = float(metrics.get("repetitive_operation_points") or 0)
        confounded = float(metrics.get("confounded_activity_points") or 0)
        priority = float(self.STATE_PRIORITY.get(state, 0))
        return (
            priority
            + sampled_points * 0.3
            + stable * 0.5
            + active * 1.0
            + repetitive * 1.5
            + confounded * 0.4
        )

    def _episode_selection_reason(self, candidate: dict[str, Any]) -> str:
        state = candidate["state"]
        if state in {"active_hand_device_operation", "repetitive_operation"}:
            return "selected_as_direct_or_repetitive_operation_episode"
        if state == "passive_screen_engagement":
            return "selected_as_stable_screen_engagement_episode"
        if state == "confounded_device_region_activity":
            return "selected_as_motion_confounder_episode"
        if state == "indirect_device_region_activity":
            return "selected_as_indirect_device_activity_episode"
        return "selected_as_background_or_quality_context_episode"

    def _select_window_state(self, state_counts: Counter[str]) -> str:
        if not state_counts:
            return "transition_or_uncertain"
        min_key_points = int(self.config.get("temporal_min_key_state_points", 2))
        for state, _priority in sorted(
            self.STATE_PRIORITY.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            if state_counts.get(state, 0) >= min_key_points:
                return state
        return state_counts.most_common(1)[0][0]

    def _window_metrics(self, flags: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(flags)
        if total == 0:
            return {}
        quality_hits = sum(1 for item in flags if item["quality_usable"])
        device_hits = sum(1 for item in flags if item["device_visible"])
        stable_hits = sum(1 for item in flags if item["stable_screen_engagement"])
        active_hits = sum(
            1 for item in flags if item["active_operation_candidate"]
        )
        direct_hits = sum(
            1 for item in flags if item["direct_hand_operation"]
        )
        repetitive_hits = sum(1 for item in flags if item["repetitive_operation"])
        indirect_activity_hits = sum(
            1
            for item in flags
            if item["unconfounded_device_region_activity"]
            and not item["active_operation_candidate"]
        )
        unconfounded_activity_hits = sum(
            1 for item in flags if item["unconfounded_device_region_activity"]
        )
        confounded_hits = sum(1 for item in flags if item["motion_confounding"])
        posture_hits = sum(1 for item in flags if item["posture_or_context_motion"])
        hand_proximity_hits = sum(1 for item in flags if item["hand_device_proximity"])
        visible_only_hits = sum(
            1
            for item in flags
            if item["device_visible"]
            and not item["stable_screen_engagement"]
            and not item["active_operation_candidate"]
            and not item["device_region_activity"]
        )
        return {
            "sampled_points": total,
            "usable_quality_points": quality_hits,
            "device_visible_points": device_hits,
            "stable_screen_points": stable_hits,
            "active_hand_device_points": active_hits,
            "direct_hand_operation_points": direct_hits,
            "repetitive_operation_points": repetitive_hits,
            "indirect_device_activity_points": indirect_activity_hits,
            "unconfounded_device_activity_points": unconfounded_activity_hits,
            "confounded_activity_points": confounded_hits,
            "posture_or_context_motion_points": posture_hits,
            "hand_device_proximity_points": hand_proximity_hits,
            "visible_without_engagement_points": visible_only_hits,
            "usable_quality_share": self._percent(quality_hits, total),
            "device_visible_share": self._percent(device_hits, total),
            "stable_screen_share": self._percent(stable_hits, total),
            "active_operation_share": self._percent(active_hits, total),
            "direct_operation_share": self._percent(direct_hits, total),
            "confounded_activity_share": self._percent(confounded_hits, total),
        }

    def _percent(self, numerator: int, denominator: int) -> str:
        if denominator <= 0:
            return "unknown"
        return f"{round(100 * numerator / denominator)}%"

    def _combine_positions(self, positions: list[str]) -> str:
        unique = []
        for position in positions:
            if position not in unique:
                unique.append(position)
        ordered = [item for item in ("early", "middle", "late") if item in unique]
        if not ordered:
            return unique[0] if unique else "unknown"
        if len(ordered) == 1:
            return ordered[0]
        return "_to_".join(ordered)

    def _duration_bin(self, sampled_points: int) -> str:
        if sampled_points <= 0:
            return "none"
        if sampled_points <= 2:
            return "very_short"
        if sampled_points <= 6:
            return "short"
        if sampled_points <= 18:
            return "medium"
        return "long"

    def _state_confidence(self, state: str, metrics: dict[str, Any]) -> str:
        sampled_points = int(metrics.get("sampled_points") or 0)
        quality_points = int(metrics.get("usable_quality_points") or 0)
        if sampled_points <= 0 or quality_points / max(sampled_points, 1) < 0.5:
            return "low"
        if state in {
            "confounded_device_region_activity",
            "transition_or_uncertain",
            "insufficient_quality",
        }:
            return "low"
        if state in {"active_hand_device_operation", "repetitive_operation"}:
            active_points = int(metrics.get("active_hand_device_points") or 0)
            direct_points = int(metrics.get("direct_hand_operation_points") or 0)
            if direct_points >= 3 or active_points >= 4:
                return "medium"
            return "low"
        if state == "passive_screen_engagement":
            stable_points = int(metrics.get("stable_screen_points") or 0)
            return "medium" if stable_points >= 3 else "low"
        return "medium"

    def _state_support_level(self, state: str, metrics: dict[str, Any]) -> str:
        active = int(metrics.get("active_hand_device_points") or 0)
        repetitive = int(metrics.get("repetitive_operation_points") or 0)
        stable = int(metrics.get("stable_screen_points") or 0)
        confounded = int(metrics.get("confounded_activity_points") or 0)
        sampled = max(int(metrics.get("sampled_points") or 0), 1)
        if state == "repetitive_operation" and repetitive >= 3:
            return "strong_operation_proxy"
        if state == "active_hand_device_operation" and active / sampled >= 0.5:
            return "moderate_operation_proxy"
        if state == "passive_screen_engagement" and stable / sampled >= 0.5:
            return "moderate_engagement_proxy"
        if state == "confounded_device_region_activity" and confounded:
            return "counter_evidence_or_confounder"
        if state == "indirect_device_region_activity":
            return "weak_indirect_activity_proxy"
        if state in self.BACKGROUND_STATES:
            return "background_or_absence_context"
        return "weak_or_uncertain_proxy"

    def _state_evidence_summary(self, state: str, metrics: dict[str, Any]) -> str:
        device = metrics.get("device_visible_points", 0)
        stable = metrics.get("stable_screen_points", 0)
        active = metrics.get("active_hand_device_points", 0)
        repetitive = metrics.get("repetitive_operation_points", 0)
        indirect = metrics.get("indirect_device_activity_points", 0)
        confounded = metrics.get("confounded_activity_points", 0)
        sampled = metrics.get("sampled_points", 0)
        if state == "insufficient_quality":
            return f"该episode质量不足，usable 点={metrics.get('usable_quality_points', 0)}/{sampled}。"
        if state == "no_device_observed":
            return f"该episode内未形成设备可见证据，抽样点={sampled}。"
        if state == "device_visible_only":
            return f"设备可见点={device}/{sampled}，但未形成稳定参与或主动操作episode。"
        if state == "passive_screen_engagement":
            return f"稳定屏幕参与点={stable}/{sampled}，直接操作点={active}。"
        if state == "active_hand_device_operation":
            return f"主动手-设备操作代理点={active}/{sampled}，重复操作点={repetitive}。"
        if state == "repetitive_operation":
            return f"重复操作代理点={repetitive}/{sampled}，直接操作点={active}。"
        if state == "indirect_device_region_activity":
            return f"未混淆的设备区域活动点={indirect}/{sampled}，但直接手-设备操作证据有限。"
        if state == "confounded_device_region_activity":
            return f"设备区域活动伴随姿态/场景运动混淆，混淆点={confounded}/{sampled}。"
        if state == "posture_or_context_motion":
            return "该episode以姿态或场景运动为主，不能直接解释为设备使用。"
        return "该episode状态不稳定，应降低确定性。"

    def _state_risk_relevance(self, state: str) -> str:
        relevance = {
            "insufficient_quality": "证据质量不足，应降低置信度或触发人工复核。",
            "no_device_observed": "未观察到设备使用场景，通常不支持风险升高。",
            "device_visible_only": "设备可见是可观察性前提，但不能单独视为风险证据。",
            "passive_screen_engagement": "可支持持续屏幕参与代理，但不等同于主动操作或成瘾。",
            "active_hand_device_operation": "可支持主动设备操作代理，是比设备可见更强的行为证据。",
            "repetitive_operation": "可支持频繁或重复操作代理，是较强的风险相关线索。",
            "indirect_device_region_activity": "只能作为设备附近活动的辅助证据，弱于直接手-设备操作。",
            "confounded_device_region_activity": "存在运动混淆，不宜据此推高风险。",
            "posture_or_context_motion": "主要用于识别混淆因素，不直接支持风险升高。",
        }
        return relevance.get(state, "状态不确定，仅作为辅助证据。")

    def _build_sequence_summary(
        self,
        *,
        temporal_points: list[dict[str, Any]],
        episode_candidates: list[dict[str, Any]],
        temporal_sequence: list[dict[str, Any]],
        aggregate: dict[str, Any],
    ) -> dict[str, Any]:
        frame_state_counts = Counter(point["state"] for point in temporal_points)
        candidate_state_counts = Counter(
            candidate["state"] for candidate in episode_candidates
        )
        step_state_counts = Counter(step["state"] for step in temporal_sequence)
        active_steps = sum(
            1
            for step in temporal_sequence
            if step["state"] in {"active_hand_device_operation", "repetitive_operation"}
        )
        active_points = sum(
            1
            for point in temporal_points
            if point["flags"]["active_operation_candidate"]
        )
        direct_points = sum(
            1 for point in temporal_points if point["flags"]["direct_hand_operation"]
        )
        repetitive_points = sum(
            1 for point in temporal_points if point["flags"]["repetitive_operation"]
        )
        stable_points = sum(
            1 for point in temporal_points if point["flags"]["stable_screen_engagement"]
        )
        visible_only_points = sum(
            1
            for point in temporal_points
            if point["flags"]["device_visible"]
            and not point["flags"]["stable_screen_engagement"]
            and not point["flags"]["active_operation_candidate"]
            and not point["flags"]["device_region_activity"]
        )
        confounded_points = sum(
            1 for point in temporal_points if point["flags"]["motion_confounding"]
        )
        unconfounded_activity_points = sum(
            1
            for point in temporal_points
            if point["flags"]["unconfounded_device_region_activity"]
        )
        repetitive_steps = step_state_counts.get("repetitive_operation", 0)
        stable_steps = step_state_counts.get("passive_screen_engagement", 0)
        confounded_steps = step_state_counts.get(
            "confounded_device_region_activity",
            0,
        )
        indirect_steps = step_state_counts.get("indirect_device_region_activity", 0)
        device_only_steps = step_state_counts.get("device_visible_only", 0)
        active_episodes = (
            candidate_state_counts.get("active_hand_device_operation", 0)
            + candidate_state_counts.get("repetitive_operation", 0)
        )
        repetitive_episodes = candidate_state_counts.get("repetitive_operation", 0)
        stable_episodes = candidate_state_counts.get("passive_screen_engagement", 0)
        confounded_episodes = candidate_state_counts.get(
            "confounded_device_region_activity",
            0,
        )
        indirect_episodes = candidate_state_counts.get(
            "indirect_device_region_activity",
            0,
        )
        device_only_episodes = candidate_state_counts.get("device_visible_only", 0)
        engagement_episodes = stable_episodes + active_episodes
        dominant_states = [
            state for state, _count in frame_state_counts.most_common(3)
        ]
        features = aggregate.get("behavior_v1_features", {})
        quality = aggregate.get("quality_summary", {}).get("overall")
        total_points = len(temporal_points)
        signal_strength = self._temporal_signal_strength(
            quality=quality,
            total_points=total_points,
            stable_points=stable_points,
            active_points=active_points,
            direct_points=direct_points,
            repetitive_points=repetitive_points,
            confounded_points=confounded_points,
            stable_episodes=stable_episodes,
            active_episodes=active_episodes,
            repetitive_episodes=repetitive_episodes,
            indirect_episodes=indirect_episodes,
            confounded_episodes=confounded_episodes,
            device_only_episodes=device_only_episodes,
        )
        return {
            "sequence_type": "selected_behavior_episode_sequence",
            "frame_state_counts": dict(sorted(frame_state_counts.items())),
            "candidate_episode_state_counts": dict(
                sorted(candidate_state_counts.items())
            ),
            "step_state_counts": dict(sorted(step_state_counts.items())),
            "dominant_states": dominant_states,
            "candidate_episode_count": len(episode_candidates),
            "selected_episode_count": len(temporal_sequence),
            "omitted_low_signal_episode_count": max(
                len(episode_candidates) - len(temporal_sequence),
                0,
            ),
            "engagement_episode_count": engagement_episodes,
            "passive_engagement_episode_count": stable_episodes,
            "active_operation_episode_count": active_episodes,
            "repetitive_operation_episode_count": repetitive_episodes,
            "confounded_activity_episode_count": confounded_episodes,
            "indirect_activity_episode_count": indirect_episodes,
            "device_visible_only_episode_count": device_only_episodes,
            "active_operation_step_count": active_steps,
            "active_operation_point_count": active_points,
            "direct_operation_point_count": direct_points,
            "repetitive_operation_step_count": repetitive_steps,
            "repetitive_operation_point_count": repetitive_points,
            "stable_screen_step_count": stable_steps,
            "stable_screen_point_count": stable_points,
            "confounded_activity_step_count": confounded_steps,
            "confounded_activity_point_count": confounded_points,
            "indirect_activity_step_count": indirect_steps,
            "unconfounded_device_activity_point_count": unconfounded_activity_points,
            "device_visible_only_step_count": device_only_steps,
            "visible_without_engagement_point_count": visible_only_points,
            "longest_stable_engagement_bin": features.get(
                "max_continuous_stable_engagement_bin",
                "unknown",
            ),
            "main_observation_pattern": self._main_observation_pattern(
                stable_points=stable_points,
                active_points=active_points,
                repetitive_points=repetitive_points,
                confounded_points=confounded_points,
                visible_only_points=visible_only_points,
                total_points=total_points,
            ),
            "temporal_video_proxy_signal_strength": signal_strength,
        }

    def _temporal_signal_strength(
        self,
        *,
        quality: str | None,
        total_points: int,
        stable_points: int,
        active_points: int,
        direct_points: int,
        repetitive_points: int,
        confounded_points: int,
        stable_episodes: int,
        active_episodes: int,
        repetitive_episodes: int,
        indirect_episodes: int,
        confounded_episodes: int,
        device_only_episodes: int,
    ) -> str:
        if quality not in {"usable_behavior_frame_quality", "usable_frame_quality"}:
            return "insufficient_video_proxy_signal"
        total = max(total_points, 1)
        stable_share = stable_points / total
        active_share = active_points / total
        confounded_share = confounded_points / total
        confounder_heavy = confounded_share >= 0.2 or confounded_episodes >= 3
        if (
            stable_share >= 0.4
            and active_share >= 0.12
            and repetitive_episodes >= 1
            and repetitive_points >= 3
            and not confounder_heavy
        ):
            return "strong_video_proxy_signal"
        if (
            stable_share >= 0.3
            and (active_share >= 0.08 or direct_points >= 6)
            and active_episodes >= 1
            and not confounder_heavy
        ):
            return "moderate_video_proxy_signal"
        if (
            stable_episodes >= 1
            and active_episodes >= 1
            and active_points >= 4
            and confounded_episodes <= 2
        ):
            return "mild_video_proxy_signal"
        if stable_share >= 0.15 and stable_episodes >= 1:
            return "mild_video_proxy_signal"
        if indirect_episodes >= 2 and confounded_episodes <= 1:
            return "mild_video_proxy_signal"
        if device_only_episodes >= 1 or stable_episodes >= 1 or indirect_episodes >= 1:
            return "weak_video_proxy_signal"
        return "weak_video_proxy_signal"

    def _main_observation_pattern(
        self,
        *,
        stable_points: int,
        active_points: int,
        repetitive_points: int,
        confounded_points: int,
        visible_only_points: int,
        total_points: int,
    ) -> str:
        total = max(total_points, 1)
        if confounded_points / total >= 0.25:
            return "motion_confounded_device_region_activity"
        if repetitive_points >= 3:
            return "repetitive_operation_observed"
        if active_points / total >= 0.08:
            return "active_operation_observed"
        if stable_points / total >= 0.3:
            return "passive_screen_engagement_dominant"
        if stable_points or active_points:
            return "limited_engagement_proxy_observed"
        if visible_only_points:
            return "device_visible_without_engagement"
        return "insufficient_observable_device_use"

    def _build_temporal_narrative(
        self,
        temporal_sequence: list[dict[str, Any]],
        sequence_summary: dict[str, Any],
    ) -> list[str]:
        lines = []
        for step in temporal_sequence:
            lines.append(
                f"{step['step_id']}: 位置={step['relative_position']}；"
                f"状态={step['state']}；覆盖={step['duration_bin']}；"
                f"支持级别={step['support_level']}；"
                f"证据={step['evidence_summary']}；置信度={step['confidence']}。"
            )
        lines.append(
            "V3时序主证据汇总: "
            f"序列类型={sequence_summary.get('sequence_type', 'unknown')}；"
            f"主导状态={', '.join(sequence_summary.get('dominant_states', [])) or 'none'}；"
            f"主要模式={sequence_summary.get('main_observation_pattern', 'unknown')}；"
            f"候选episode数={sequence_summary.get('candidate_episode_count', 0)}；"
            f"已选episode数={sequence_summary.get('selected_episode_count', 0)}；"
            f"稳定参与episode数={sequence_summary.get('passive_engagement_episode_count', 0)}；"
            f"主动/重复操作episode数={sequence_summary.get('active_operation_episode_count', 0)}；"
            f"混淆episode数={sequence_summary.get('confounded_activity_episode_count', 0)}。"
        )
        return lines

    def _build_visual_proxy_lines(self, aggregate: dict[str, Any]) -> list[str]:
        lines = super()._build_visual_proxy_lines(aggregate)
        summary = aggregate.get("temporal_sequence_summary", {})
        lines.extend(
            [
                "PriVTE-Behavior v3时序状态: computed_by_privte_behavior_v3_temporal",
                "时序序列类型: "
                + str(summary.get("sequence_type", "unknown")),
                "时序主导状态: "
                + ", ".join(summary.get("dominant_states", []) or ["none"]),
                "时序主要模式: "
                + str(summary.get("main_observation_pattern", "unknown")),
                "稳定参与episode数: "
                + str(summary.get("passive_engagement_episode_count", "unknown")),
                "主动/重复操作episode数: "
                + str(summary.get("active_operation_episode_count", "unknown")),
                "重复操作episode数: "
                + str(summary.get("repetitive_operation_episode_count", "unknown")),
                "混淆活动episode数: "
                + str(summary.get("confounded_activity_episode_count", "unknown")),
                "时序证据强度: "
                + str(
                    summary.get(
                        "temporal_video_proxy_signal_strength",
                        "unknown",
                    )
                ),
            ]
        )
        return lines
