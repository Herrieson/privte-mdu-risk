"""LLM-ready evidence package construction for PriVTE records."""

from __future__ import annotations

from typing import Any


RISK_LEVELS = [
    "no_observed_risk",
    "mild_risk",
    "moderate_risk",
    "high_risk",
    "insufficient_evidence",
]


LLM_HIDDEN_MISSING_INFORMATION = {
    "exact_touch_events",
    "direct_gaze_estimation",
    "validated_affect_or_fatigue_labels",
    "screen_content_ocr",
    "high_dimensional_pose_or_face_mesh",
    "questionnaire_input",
    "exact_heart_rate_input",
    "app_name_input",
}


def _bin(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    return str(value)


def _is_at_least(level: str, accepted: set[str]) -> bool:
    return level in accepted


def _event_phrase(event_type: str) -> str:
    phrases = {
        "near_device_motion_burst": "设备附近运动突增",
        "global_motion_burst": "全局运动突增",
        "stable_screen_viewing_proxy": "稳定屏幕观看代理窗口",
        "hand_device_interaction_burst": "手-设备交互突增",
        "device_region_motion_burst": "设备区域运动突增",
        "posture_or_context_motion_burst": "姿态或场景运动突增",
        "stable_screen_engagement_proxy": "稳定屏幕参与代理窗口",
    }
    return phrases.get(event_type, event_type)


def _quality_recommendation(overall: str) -> str:
    if overall in {
        "insufficient_video",
        "frame_analysis_unavailable",
        "behavior_frame_analysis_unavailable",
        "low_frame_quality",
        "low_behavior_frame_quality",
    }:
        return "证据包内部质量不足，应降低置信度并考虑 insufficient_evidence 或人工复核。"
    if overall in {
        "partial_frame_quality",
        "partial_behavior_frame_quality",
        "file_level_quality_only",
    }:
        return "证据质量部分可用，应降低置信度并说明不确定性。"
    if overall in {
        "usable_frame_quality",
        "usable_behavior_frame_quality",
        "usable_container_quality",
    }:
        return "证据质量可用，可基于本证据包进行风险筛查。"
    return "证据质量未知，模型应谨慎判断并说明不确定性。"


def _llm_visible_missing_information(items: list[str]) -> list[str]:
    """Keep only evidence-internal missingness visible to the LLM prompt."""

    return [item for item in items if item not in LLM_HIDDEN_MISSING_INFORMATION]


def _interaction_level(near_motion: str, interaction_bursts: str) -> str:
    if near_motion in {"elevated", "high"} and interaction_bursts == "high":
        return "high"
    if near_motion in {"elevated", "high"} or interaction_bursts in {"medium", "high"}:
        return "medium"
    if near_motion in {"low", "medium"} or interaction_bursts in {"none", "low"}:
        return "low"
    return "unknown"


def _face_based_observability(face_visibility: str, cooccurrence: str) -> str:
    if face_visibility in {"high", "very_high"} and cooccurrence in {"high", "very_high"}:
        return "high"
    if face_visibility in {"medium", "high", "very_high"} and cooccurrence in {
        "medium",
        "high",
        "very_high",
    }:
        return "medium"
    if face_visibility in {"none", "low"} or cooccurrence in {"none", "low"}:
        return "low"
    return "unknown"


def _append_observation(
    observations: list[dict[str, Any]],
    *,
    signal: str,
    value: str,
    interpretation: str,
    risk_relevance: str,
    quality: str,
) -> None:
    observations.append(
        {
            "signal": signal,
            "value": value,
            "interpretation": interpretation,
            "risk_relevance": risk_relevance,
            "quality": quality,
        }
    )


def _build_flowlite_behavior_package(video: dict[str, Any], quality: dict[str, Any]) -> dict[str, Any]:
    features = video.get("global_features", {})
    key_windows = video.get("key_window_summary", {})
    event_windows = video.get("event_windows", [])
    overall_quality = _bin(quality.get("overall"))

    valid_quality = _bin(features.get("valid_frame_ratio_bin"))
    face_visibility = _bin(features.get("face_visible_ratio_bin"))
    screen_visibility = _bin(features.get("screen_like_region_visible_ratio_bin"))
    cooccurrence = _bin(features.get("face_screen_cooccurrence_ratio_bin"))
    stable_viewing = _bin(features.get("stable_viewing_proxy_ratio_bin"))
    near_motion = _bin(features.get("near_device_motion_level"))
    interaction_bursts = _bin(features.get("interaction_burst_count_bin"))
    motion_bursts = _bin(features.get("motion_burst_count_bin"))
    global_motion = _bin(features.get("global_motion_level"))
    active_interaction = _interaction_level(near_motion, interaction_bursts)
    face_observability = _face_based_observability(face_visibility, cooccurrence)

    observations: list[dict[str, Any]] = []
    supporting: list[str] = []
    uncertainty: list[str] = []

    if overall_quality != "usable_frame_quality":
        uncertainty.append(_quality_recommendation(overall_quality))
    if valid_quality not in {"high", "very_high"}:
        uncertainty.append("有效帧质量不足够高，行为证据的可靠性下降。")

    _append_observation(
        observations,
        signal="device_use_observability",
        value=screen_visibility,
        interpretation="视频中可见设备/屏幕样区域的比例水平。",
        risk_relevance="这是判断数字设备使用行为是否可观察的基础条件，不直接等同于风险。",
        quality=overall_quality,
    )
    if _is_at_least(screen_visibility, {"high", "very_high"}):
        supporting.append("设备/屏幕样区域可见性较高，说明视频中存在较充分的设备使用观察窗口。")
    elif screen_visibility in {"none", "low"}:
        uncertainty.append("设备/屏幕样区域可见性较低，数字设备使用行为证据不足。")

    _append_observation(
        observations,
        signal="sustained_screen_engagement_proxy",
        value=stable_viewing,
        interpretation="设备/屏幕样区域可见且全局运动不突增的聚合比例。",
        risk_relevance=(
            "可作为持续观看或稳定投入的代理线索；只能支持风险筛查，不能等同于注意力或成瘾。"
        ),
        quality=overall_quality,
    )
    if _is_at_least(stable_viewing, {"high", "very_high"}):
        supporting.append("稳定观看代理比例较高，支持存在较持续的屏幕参与线索。")
    elif stable_viewing == "medium":
        supporting.append("稳定观看代理比例为中等，提示存在一定屏幕参与，但强度有限。")
    elif stable_viewing in {"none", "low"}:
        uncertainty.append("稳定观看代理比例较低，持续屏幕参与证据较弱。")

    _append_observation(
        observations,
        signal="active_device_interaction_proxy",
        value=active_interaction,
        interpretation=(
            f"由近设备运动水平={near_motion} 与交互突增窗口数量={interaction_bursts} 综合得到。"
        ),
        risk_relevance="较高值可作为频繁操作、滑动/点击或设备附近动作活跃的代理线索。",
        quality=overall_quality,
    )
    if active_interaction == "high":
        supporting.append("活跃设备交互代理信号较高，提示设备附近存在较频繁的操作或姿态变化。")
    elif active_interaction == "medium":
        supporting.append("活跃设备交互代理信号为中等，存在一定操作活跃性但强度有限。")
    elif active_interaction == "low":
        uncertainty.append("近设备运动水平不高，交互活跃性证据有限。")

    _append_observation(
        observations,
        signal="repetitive_operation_proxy",
        value=interaction_bursts,
        interpretation="相对个体内运动基线的设备附近运动突增次数。",
        risk_relevance="较多突增窗口可提示重复操作或频繁交互片段，但不识别具体 app 或内容。",
        quality=overall_quality,
    )
    if interaction_bursts == "high":
        supporting.append("交互突增窗口数量较高，支持频繁交互或操作变化的代理证据。")
    elif interaction_bursts in {"none", "low"}:
        uncertainty.append("交互突增窗口较少，频繁操作证据不足。")

    _append_observation(
        observations,
        signal="face_based_engagement_observability",
        value=face_observability,
        interpretation=(
            f"由人脸可见性={face_visibility} 与人脸/屏幕共现={cooccurrence} 综合得到。"
        ),
        risk_relevance="用于评估屏幕相关行为上下文的可观察性。",
        quality=overall_quality,
    )
    if face_observability == "high":
        supporting.append("人脸与设备/屏幕样区域共现较高，屏幕相关行为观察更可靠。")
    elif face_observability == "medium":
        supporting.append("人脸相关可观察性为中等，部分屏幕相关行为线索可用。")
    elif face_observability == "low":
        uncertainty.append("人脸与设备/屏幕样区域共现较低，屏幕相关行为上下文证据较弱。")

    _append_observation(
        observations,
        signal="posture_or_context_change_proxy",
        value=motion_bursts,
        interpretation=f"由全局运动水平={global_motion} 与全局运动突增窗口数量={motion_bursts} 得到。",
        risk_relevance="可作为姿态变化、离开/返回或拍摄场景扰动的弱代理线索，需要与设备交互区分。",
        quality=overall_quality,
    )
    if motion_bursts == "high":
        supporting.append("全局运动突增较多，可能提示姿态频繁变化，但需要与设备附近运动区分。")
    elif motion_bursts in {"none", "low"}:
        uncertainty.append("姿态或场景运动变化证据较少，不能支持姿态相关风险线索。")

    rendered_events = []
    for index, event in enumerate(event_windows, start=1):
        event_type = _bin(event.get("event_type"))
        rendered_events.append(
            {
                "window_id": f"event_{index:02d}",
                "relative_position": _bin(event.get("relative_position")),
                "observation": _event_phrase(event_type),
                "strength": _bin(event.get("strength")),
                "quality": _bin(event.get("quality")),
                "risk_relevance": (
                    "用于提示值得关注的行为片段；仅表示代理事件窗口，不表示具体内容或诊断。"
                ),
            }
        )

    event_types = key_windows.get("event_window_types", {})
    if not rendered_events:
        uncertainty.append("未检测到可用关键事件窗口，模型不应推断隐藏行为。")

    if not supporting:
        supporting.append("未形成明确支持风险升高的高质量行为证据。")

    evidence_strength = "weak_video_proxy_signal"
    moderate_signal_count = sum(
        [
            screen_visibility in {"medium", "high", "very_high"},
            stable_viewing in {"medium", "high", "very_high"},
            active_interaction in {"medium", "high"},
            interaction_bursts == "high",
        ]
    )
    strong_signal_count = sum(
        [
            screen_visibility in {"high", "very_high"},
            stable_viewing in {"high", "very_high"},
            active_interaction == "high",
            interaction_bursts == "high",
        ]
    )
    if overall_quality == "usable_frame_quality" and moderate_signal_count >= 3:
        evidence_strength = "moderate_video_proxy_signal"
    if overall_quality == "usable_frame_quality" and strong_signal_count >= 4:
        evidence_strength = "strong_video_proxy_signal"
    if overall_quality in {"insufficient_video", "frame_analysis_unavailable", "low_frame_quality"}:
        evidence_strength = "insufficient_video_proxy_signal"

    return {
        "behavior_observations": observations,
        "key_event_windows": rendered_events,
        "key_event_window_counts": event_types,
        "risk_relevant_synthesis": {
            "video_proxy_signal_strength": evidence_strength,
            "signals_supporting_closer_review": supporting,
            "signals_reducing_certainty": uncertainty,
            "quality_recommendation": _quality_recommendation(overall_quality),
        },
    }


def _build_non_behavior_package(video: dict[str, Any], quality: dict[str, Any]) -> dict[str, Any]:
    overall_quality = _bin(quality.get("overall"))
    visual_proxy_features = video.get("visual_proxy_features", [])
    observations = [
        {
            "signal": "video_availability_or_metadata",
            "value": "; ".join(str(item) for item in visual_proxy_features) or "not_available",
            "interpretation": "当前 extractor 只提供视频可用性或容器级信息。",
            "risk_relevance": "不能直接支持行为风险判断，只能用于判断数据是否可分析。",
            "quality": overall_quality,
        }
    ]
    return {
        "behavior_observations": observations,
        "key_event_windows": [],
        "key_event_window_counts": {},
        "risk_relevant_synthesis": {
            "video_proxy_signal_strength": "insufficient_video_proxy_signal",
            "signals_supporting_closer_review": [
                "当前输出没有帧级行为代理证据，不能据此进行可靠风险分级。"
            ],
            "signals_reducing_certainty": [
                "当前输出缺少可用于筛查的行为证据。"
            ],
            "quality_recommendation": _quality_recommendation(overall_quality),
        },
    }


def _build_behavior_v1_package(video: dict[str, Any], quality: dict[str, Any]) -> dict[str, Any]:
    features = video.get("behavior_v1_features", {})
    key_windows = video.get("key_window_summary", {})
    event_windows = video.get("event_windows", [])
    overall_quality = _bin(quality.get("overall"))

    device_visibility = _bin(features.get("device_visible_ratio_bin"))
    hand_visibility = _bin(features.get("hand_visible_ratio_bin"))
    hand_device = _bin(features.get("hand_device_proximity_ratio_bin"))
    stable_engagement = _bin(features.get("stable_screen_engagement_proxy_ratio_bin"))
    active_interaction = _bin(
        features.get("active_hand_device_interaction_proxy_ratio_bin")
    )
    device_activity = _bin(features.get("device_region_activity_proxy_ratio_bin"))
    device_activity_windows = _bin(
        features.get("device_region_activity_proxy_count_bin")
    )
    repetitive_ops = _bin(features.get("repetitive_operation_proxy_count_bin"))
    face_context = _bin(features.get("face_device_cooccurrence_ratio_bin"))
    head_device = _bin(features.get("face_device_alignment_proxy_ratio_bin"))
    posture_change = _bin(features.get("posture_or_context_change_count_bin"))
    max_stable = _bin(features.get("max_continuous_stable_engagement_bin"))

    observations: list[dict[str, Any]] = []
    supporting: list[str] = []
    uncertainty: list[str] = []

    if overall_quality not in {"usable_behavior_frame_quality", "usable_frame_quality"}:
        uncertainty.append(_quality_recommendation(overall_quality))

    _append_observation(
        observations,
        signal="device_use_observability",
        value=device_visibility,
        interpretation="由设备/屏幕检测或备用屏幕样区域检测得到的可观察性水平。",
        risk_relevance="这是视频中数字设备使用行为能否被观察的基础条件。",
        quality=overall_quality,
    )
    if device_visibility in {"high", "very_high"}:
        supporting.append("设备/屏幕可观察性较高，视频中存在较充分的设备使用观察窗口。")
    elif device_visibility in {"none", "low"}:
        uncertainty.append("设备/屏幕可观察性较低，数字设备使用行为证据不足。")

    _append_observation(
        observations,
        signal="sustained_screen_engagement_proxy",
        value=stable_engagement,
        interpretation="设备可见、质量可用且行为上下文相对稳定的聚合比例。",
        risk_relevance="较高值可作为持续屏幕参与代理证据，不能等同于成瘾或精确注意力。",
        quality=overall_quality,
    )
    if stable_engagement in {"high", "very_high"}:
        supporting.append("稳定屏幕参与代理较高，支持存在较持续的屏幕参与线索。")
    elif stable_engagement == "medium":
        supporting.append("稳定屏幕参与代理为中等，提示存在一定屏幕参与但强度有限。")

    _append_observation(
        observations,
        signal="max_continuous_stable_engagement_proxy",
        value=max_stable,
        interpretation="抽样帧序列中连续稳定屏幕参与代理的最长片段。",
        risk_relevance="用于补充持续性证据；该值受抽样策略影响，不是精确时长。",
        quality=overall_quality,
    )

    _append_observation(
        observations,
        signal="hand_device_interaction_proxy",
        value=active_interaction,
        interpretation=(
            f"由手部可见性={hand_visibility}、手-设备接近={hand_device} "
            f"和设备区域活动={device_activity} 综合得到。"
        ),
        risk_relevance="较高值可作为操作、触摸样动作或滑动样动作活跃的代理线索。",
        quality=overall_quality,
    )
    if active_interaction in {"high", "very_high"}:
        supporting.append("手-设备交互代理较高，提示存在较活跃的设备操作线索。")
    elif active_interaction == "medium":
        supporting.append("手-设备交互代理为中等，存在一定操作线索。")
    elif active_interaction in {"none", "low"}:
        uncertainty.append("手-设备交互代理较弱，操作活跃性证据有限。")

    _append_observation(
        observations,
        signal="device_region_activity_proxy",
        value=device_activity,
        interpretation=f"由设备区域短间隔局部运动突增得到，窗口数量={device_activity_windows}。",
        risk_relevance=(
            "可作为设备附近滑动、点击、触摸样动作或屏幕区域变化的弱代理；"
            "不识别具体内容、app 或真实触摸事件。"
        ),
        quality=overall_quality,
    )
    if device_activity in {"high", "very_high"} or device_activity_windows in {
        "high",
        "very_high",
    }:
        supporting.append("设备区域活动代理较高，支持存在设备附近活跃交互线索。")
    elif device_activity == "medium" or device_activity_windows == "medium":
        supporting.append("设备区域活动代理为中等，提示存在一定设备附近活动线索。")
    elif device_activity in {"none", "low"} and device_activity_windows in {"none", "low"}:
        uncertainty.append("设备区域活动代理较弱，不能充分支持活跃操作线索。")

    _append_observation(
        observations,
        signal="repetitive_operation_proxy",
        value=repetitive_ops,
        interpretation="由手部接近设备且手部运动突增的窗口数量得到。",
        risk_relevance="较多窗口可提示重复操作或频繁交互，但不是已验证的点击/滑动标签。",
        quality=overall_quality,
    )
    if repetitive_ops == "high":
        supporting.append("重复操作代理窗口较多，支持频繁交互或重复操作线索。")
    elif repetitive_ops in {"none", "low"}:
        uncertainty.append("重复操作代理窗口较少，重复操作证据不足。")

    _append_observation(
        observations,
        signal="face_device_context_observability",
        value=face_context,
        interpretation=f"由人脸-设备共现={face_context} 和头部/设备相对上下文={head_device} 得到。",
        risk_relevance="用于评估屏幕相关行为上下文的可观察性。",
        quality=overall_quality,
    )
    if face_context in {"none", "low"}:
        uncertainty.append("人脸-设备上下文较弱，屏幕相关行为上下文证据较弱。")

    _append_observation(
        observations,
        signal="posture_or_context_change_proxy",
        value=posture_change,
        interpretation="由姿态中心变化和全局运动突增聚合得到。",
        risk_relevance="可作为姿态变化、离开/返回或拍摄场景扰动的弱代理线索。",
        quality=overall_quality,
    )

    rendered_events = []
    for index, event in enumerate(event_windows, start=1):
        event_type = _bin(event.get("event_type"))
        rendered_events.append(
            {
                "window_id": f"event_{index:02d}",
                "relative_position": _bin(event.get("relative_position")),
                "observation": _event_phrase(event_type),
                "strength": _bin(event.get("strength")),
                "quality": _bin(event.get("quality")),
                "risk_relevance": "用于提示值得关注的行为片段；仅表示代理事件窗口，不表示诊断。",
            }
        )

    moderate_signal_count = sum(
        [
            device_visibility in {"medium", "high", "very_high"},
            stable_engagement in {"medium", "high", "very_high"},
            max_stable in {"medium", "high"},
            active_interaction in {"medium", "high", "very_high"},
            device_activity in {"medium", "high", "very_high"}
            or device_activity_windows in {"medium", "high", "very_high"},
            face_context in {"medium", "high", "very_high"},
            repetitive_ops in {"medium", "high"},
        ]
    )
    strong_signal_count = sum(
        [
            device_visibility in {"high", "very_high"},
            stable_engagement in {"high", "very_high"},
            max_stable == "high",
            active_interaction in {"high", "very_high"},
            device_activity in {"high", "very_high"}
            or device_activity_windows in {"high", "very_high"},
            repetitive_ops == "high",
        ]
    )
    evidence_strength = "weak_video_proxy_signal"
    usable_quality = overall_quality == "usable_behavior_frame_quality"
    partial_quality = overall_quality == "partial_behavior_frame_quality"
    interaction_available = (
        active_interaction in {"medium", "high", "very_high"}
        or device_activity in {"medium", "high", "very_high"}
        or device_activity_windows in {"medium", "high", "very_high"}
    )
    if usable_quality and moderate_signal_count >= 2:
        evidence_strength = "mild_video_proxy_signal"
    if usable_quality and moderate_signal_count >= 4 and interaction_available:
        evidence_strength = "moderate_video_proxy_signal"
    if usable_quality and moderate_signal_count >= 5 and strong_signal_count >= 4:
        evidence_strength = "strong_video_proxy_signal"
    if partial_quality and moderate_signal_count >= 4 and interaction_available:
        evidence_strength = "mild_video_proxy_signal"
    if overall_quality in {
        "behavior_frame_analysis_unavailable",
        "insufficient_video",
        "low_behavior_frame_quality",
    }:
        evidence_strength = "insufficient_video_proxy_signal"

    return {
        "behavior_observations": observations,
        "key_event_windows": rendered_events,
        "key_event_window_counts": key_windows.get("event_window_types", {}),
        "risk_relevant_synthesis": {
            "video_proxy_signal_strength": evidence_strength,
            "signals_supporting_closer_review": supporting
            or ["未形成明确支持风险升高的高质量行为证据。"],
            "signals_reducing_certainty": uncertainty,
            "quality_recommendation": _quality_recommendation(overall_quality),
        },
    }


def _build_behavior_v2_package(video: dict[str, Any], quality: dict[str, Any]) -> dict[str, Any]:
    package = _build_behavior_v1_package(video, quality)
    v2_features = video.get("behavior_v2_features", {})
    concrete_evidence = video.get("concrete_behavior_evidence", [])
    privacy_summary = video.get("privacy_preserving_behavior_summary", {})
    calibrated_strength = v2_features.get("risk_strength_calibration")
    if calibrated_strength:
        package["risk_relevant_synthesis"]["video_proxy_signal_strength"] = calibrated_strength

    supporting = package["risk_relevant_synthesis"]["signals_supporting_closer_review"]
    uncertainty = package["risk_relevant_synthesis"]["signals_reducing_certainty"]

    direct_level = _bin(v2_features.get("direct_interaction_evidence_level"))
    continuity_level = _bin(v2_features.get("screen_continuity_evidence_level"))
    absence_level = _bin(v2_features.get("absence_of_repetitive_operation_evidence_level"))
    confounding_level = _bin(v2_features.get("motion_confounding_level"))
    coarse_measurements = v2_features.get("coarse_behavior_measurements", {})

    if continuity_level in {"moderate", "strong"}:
        supporting.append(f"V2连续性证据为 {continuity_level}，支持存在粗粒度稳定屏幕参与片段。")
    if direct_level in {"strong_direct_or_repetitive_proxy", "moderate_direct_proxy"}:
        supporting.append(f"V2直接交互证据为 {direct_level}，支持存在更直接的设备操作代理线索。")
    elif direct_level in {
        "limited_direct_interaction_evidence",
        "limited_indirect_activity",
        "indirect_device_region_activity_only",
    }:
        uncertainty.append(f"V2直接交互证据为 {direct_level}，不宜仅凭设备区域活动推高风险。")
    if absence_level in {
        "no_clear_sustained_or_repetitive_proxy_observed",
        "no_clear_repetitive_operation_proxy_observed",
    }:
        uncertainty.append(f"V2未观察到证据: {absence_level}。")
    if confounding_level in {"moderate_potential_confounding", "high_potential_confounding"}:
        uncertainty.append(f"V2运动混淆检查为 {confounding_level}，设备附近活动可能受姿态或场景运动影响。")

    package["concrete_behavior_evidence"] = concrete_evidence
    package["privacy_preserving_behavior_summary"] = privacy_summary
    package["behavior_v2_summary"] = {
        "device_observability_level": v2_features.get("device_observability_level"),
        "screen_continuity_evidence_level": continuity_level,
        "direct_interaction_evidence_level": direct_level,
        "absence_of_repetitive_operation_evidence_level": absence_level,
        "motion_confounding_level": confounding_level,
        "risk_strength_calibration": calibrated_strength,
        "coarse_behavior_measurements": coarse_measurements,
    }
    return package


def _build_behavior_v3_temporal_package(
    video: dict[str, Any],
    quality: dict[str, Any],
) -> dict[str, Any]:
    package = _build_behavior_v1_package(video, quality)
    v2_features = video.get("behavior_v2_features", {})
    v3_features = video.get("behavior_v3_temporal_features", {})
    sequence_summary = video.get("temporal_sequence_summary", {})
    temporal_sequence = video.get("temporal_behavior_sequence", [])
    temporal_narrative = video.get("temporal_behavior_narrative", [])
    concrete_evidence = [
        item
        for item in video.get("concrete_behavior_evidence", [])
        if item.get("evidence_type") != "repetitive_operation_or_absence"
    ]
    privacy_summary = video.get("privacy_preserving_behavior_summary", {})
    calibrated_strength = (
        v3_features.get("risk_strength_calibration")
        or sequence_summary.get("temporal_video_proxy_signal_strength")
    )
    if calibrated_strength:
        package["risk_relevant_synthesis"]["video_proxy_signal_strength"] = (
            calibrated_strength
        )

    supporting: list[str] = []
    uncertainty: list[str] = []
    package["risk_relevant_synthesis"]["signals_supporting_closer_review"] = supporting
    package["risk_relevant_synthesis"]["signals_reducing_certainty"] = uncertainty

    selected_episode_count = int(sequence_summary.get("selected_episode_count") or 0)
    candidate_episode_count = int(sequence_summary.get("candidate_episode_count") or 0)
    omitted_episode_count = int(
        sequence_summary.get("omitted_low_signal_episode_count") or 0
    )
    engagement_episodes = int(sequence_summary.get("engagement_episode_count") or 0)
    stable_episodes = int(
        sequence_summary.get("passive_engagement_episode_count") or 0
    )
    active_episodes = int(
        sequence_summary.get("active_operation_episode_count") or 0
    )
    repetitive_episodes = int(
        sequence_summary.get("repetitive_operation_episode_count") or 0
    )
    confounded_episodes = int(
        sequence_summary.get("confounded_activity_episode_count") or 0
    )
    indirect_episodes = int(sequence_summary.get("indirect_activity_episode_count") or 0)
    device_only_episodes = int(
        sequence_summary.get("device_visible_only_episode_count") or 0
    )
    stable_steps = int(sequence_summary.get("stable_screen_step_count") or 0)
    active_steps = int(sequence_summary.get("active_operation_step_count") or 0)
    active_points = int(sequence_summary.get("active_operation_point_count") or 0)
    direct_points = int(sequence_summary.get("direct_operation_point_count") or 0)
    repetitive_steps = int(sequence_summary.get("repetitive_operation_step_count") or 0)
    repetitive_points = int(
        sequence_summary.get("repetitive_operation_point_count") or 0
    )
    stable_points = int(sequence_summary.get("stable_screen_point_count") or 0)
    confounded_steps = int(sequence_summary.get("confounded_activity_step_count") or 0)
    confounded_points = int(sequence_summary.get("confounded_activity_point_count") or 0)
    device_only_steps = int(sequence_summary.get("device_visible_only_step_count") or 0)
    visible_only_points = int(
        sequence_summary.get("visible_without_engagement_point_count") or 0
    )
    dominant_states = sequence_summary.get("dominant_states", [])
    main_pattern = _bin(sequence_summary.get("main_observation_pattern"))

    if temporal_sequence:
        supporting.append(
            "V3主证据为按先后顺序选择的隐私安全行为episode序列，优先于旧版全局比例综合。"
        )
    if selected_episode_count:
        supporting.append(
            f"V3选取 {selected_episode_count}/{candidate_episode_count} 个关键episode；"
            f"未输出的 {omitted_episode_count} 个低信号episode仅作为聚合计数保留。"
        )
    if stable_episodes or stable_points:
        supporting.append(f"V3观察到稳定屏幕参与时序步骤数={stable_steps}。")
        supporting.append(
            f"V3稳定屏幕参与episode数={stable_episodes}，抽样点数={stable_points}。"
        )
    if active_episodes and active_points >= 4:
        supporting.append(
            f"V3主动/重复操作episode数={active_episodes}，主动操作抽样点数={active_points}，直接操作点数={direct_points}。"
        )
    elif active_points:
        uncertainty.append(
            f"V3仅观察到少量主动操作抽样点={active_points}，不足以单独支持较高风险判断。"
        )
    if repetitive_episodes and repetitive_points >= 3:
        supporting.append(
            f"V3观察到重复操作代理episode数={repetitive_episodes}，重复操作抽样点数={repetitive_points}。"
        )
    elif repetitive_points:
        uncertainty.append(
            f"V3仅观察到少量重复操作抽样点={repetitive_points}，重复性证据较弱。"
        )
    if device_only_episodes or visible_only_points:
        uncertainty.append(
            f"V3设备仅可见episode数={device_only_episodes}，仅可见且未形成参与/操作的抽样点数={visible_only_points}；设备可见不能单独推高风险。"
        )
    if not active_episodes and not active_points:
        uncertainty.append("V3未观察到清晰主动手-设备操作episode。")
    if not repetitive_episodes and not repetitive_points:
        uncertainty.append("V3未观察到重复操作代理步骤，应降低 moderate/high 判断确定性。")
    if confounded_episodes or confounded_points:
        uncertainty.append(
            f"V3观察到混淆设备区域活动episode数={confounded_episodes}，抽样点数={confounded_points}，设备附近活动可能受姿态或场景运动影响。"
        )
    if main_pattern in {
        "device_visible_without_engagement",
        "insufficient_observable_device_use",
        "motion_confounded_device_region_activity",
    }:
        uncertainty.append(f"V3主要观察模式为 {main_pattern}，不应将其解释为强风险证据。")
    if calibrated_strength in {
        "weak_video_proxy_signal",
        "insufficient_video_proxy_signal",
    }:
        uncertainty.append(f"V3时序证据强度为 {calibrated_strength}。")
    if not supporting:
        supporting.append("V3未形成明确支持风险升高的高质量时序行为代理证据。")

    package["concrete_behavior_evidence"] = concrete_evidence
    package["privacy_preserving_behavior_summary"] = privacy_summary
    package["temporal_behavior_sequence"] = temporal_sequence
    package["temporal_behavior_narrative"] = temporal_narrative
    package["temporal_sequence_summary"] = sequence_summary
    package["behavior_v3_temporal_summary"] = {
        "sequence_type": sequence_summary.get("sequence_type"),
        "dominant_states": dominant_states,
        "main_observation_pattern": main_pattern,
        "selected_episode_count": selected_episode_count,
        "candidate_episode_count": candidate_episode_count,
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
        "device_visible_only_step_count": device_only_steps,
        "visible_without_engagement_point_count": visible_only_points,
        "risk_strength_calibration": calibrated_strength,
        "behavior_v2_background_calibration": v2_features.get(
            "risk_strength_calibration"
        ),
        "state_vocabulary": v3_features.get("state_vocabulary", []),
        "privacy_policy": v3_features.get("privacy_policy", {}),
    }
    return package


def _trace_strength_to_signal(strength: str) -> str:
    mapping = {
        "insufficient_observability": "insufficient_behavior_signal",
        "no_clear_risk_pattern": "low_behavior_signal",
        "mild_risk_pattern": "mild_behavior_signal",
        "moderate_risk_pattern": "moderate_behavior_signal",
        "strong_risk_pattern": "strong_behavior_signal",
    }
    return mapping.get(strength, "unknown_behavior_signal")


def _build_trace_v1_package(
    video: dict[str, Any],
    quality: dict[str, Any],
) -> dict[str, Any]:
    package = _build_behavior_v3_temporal_package(video, quality)
    trace_summary = video.get("trace_risk_summary", {})
    trace_narrative = video.get("trace_behavior_narrative", [])
    strength = _bin(trace_summary.get("risk_pattern_strength"))
    ordinary_pattern = _bin(trace_summary.get("ordinary_use_pattern"))
    coarse_levels = trace_summary.get("coarse_levels", {})
    coarse_rates = trace_summary.get("coarse_rates", {})

    package["risk_relevant_synthesis"]["video_proxy_signal_strength"] = (
        _trace_strength_to_signal(strength)
    )

    observations: list[dict[str, Any]] = [
        {
            "signal": "trace_risk_pattern_strength",
            "value": strength,
            "interpretation": (
                "Trace v1 对全局行为轨迹的风险性模式强度校准；普通设备使用本身不自动计为风险。"
            ),
            "risk_relevance": (
                "用于区分普通/低强度使用与持续、高频、重复或跨阶段一致的风险性使用模式。"
            ),
            "quality": trace_summary.get("quality_overall", "unknown"),
        },
        {
            "signal": "ordinary_use_pattern",
            "value": ordinary_pattern,
            "interpretation": "Trace v1 对普通使用、低强度使用或风险性模式的总体解释。",
            "risk_relevance": "用于防止将设备可见或一般屏幕参与直接解释为风险。",
            "quality": trace_summary.get("quality_overall", "unknown"),
        },
        {
            "signal": "stable_engagement_burden",
            "value": _bin(coarse_levels.get("stable_engagement_burden")),
            "interpretation": (
                f"稳定参与粗比例={coarse_rates.get('stable_engagement_share')}。"
            ),
            "risk_relevance": "持续参与是风险相关模式的一部分，但需结合交互密度和重复性。",
            "quality": trace_summary.get("quality_overall", "unknown"),
        },
        {
            "signal": "active_operation_density",
            "value": _bin(coarse_levels.get("active_operation_density")),
            "interpretation": (
                f"主动操作粗比例={coarse_rates.get('active_operation_share')}。"
            ),
            "risk_relevance": "主动操作密度用于判断是否超出普通可见设备使用。",
            "quality": trace_summary.get("quality_overall", "unknown"),
        },
        {
            "signal": "repetition_density",
            "value": _bin(coarse_levels.get("repetition_density")),
            "interpretation": (
                f"重复操作粗比例={coarse_rates.get('repetitive_operation_share')}。"
            ),
            "risk_relevance": "重复性是比单次交互更强的风险相关线索。",
            "quality": trace_summary.get("quality_overall", "unknown"),
        },
        {
            "signal": "risk_phase_consistency",
            "value": _bin(coarse_levels.get("phase_consistency")),
            "interpretation": "主动或重复操作是否跨 early/middle/late 多阶段出现。",
            "risk_relevance": "跨阶段一致性用于区分局部短时行为和更稳定的风险性模式。",
            "quality": trace_summary.get("quality_overall", "unknown"),
        },
        {
            "signal": "motion_confounding",
            "value": _bin(coarse_levels.get("motion_confounding")),
            "interpretation": (
                f"运动混淆粗比例={coarse_rates.get('confounded_activity_share')}。"
            ),
            "risk_relevance": "混淆较高时，应降低对设备区域活动的风险解释强度。",
            "quality": trace_summary.get("quality_overall", "unknown"),
        },
    ]
    package["behavior_observations"] = observations

    risk_indicators = trace_summary.get("risk_pattern_indicators", [])
    ordinary_indicators = trace_summary.get("ordinary_or_low_risk_indicators", [])
    counter_indicators = trace_summary.get("counter_risk_indicators", [])

    if strength in {"no_clear_risk_pattern", "insufficient_observability"}:
        supporting = [
            "Trace v1 未形成明确风险性使用轨迹；应主要依据普通/低风险使用证据和反向证据判断。"
        ]
    else:
        supporting = list(risk_indicators) or [
            "Trace v1 形成风险性使用轨迹，但缺少可渲染的具体支持条目。"
        ]
    uncertainty = list(ordinary_indicators) + list(counter_indicators)
    if not uncertainty:
        uncertainty = ["未形成额外反向证据。"]

    package["risk_relevant_synthesis"]["signals_supporting_closer_review"] = supporting
    package["risk_relevant_synthesis"]["signals_reducing_certainty"] = uncertainty
    package["trace_risk_summary"] = trace_summary
    package["trace_behavior_narrative"] = trace_narrative
    package["trace_feature_table"] = {
        "risk_pattern_strength": strength,
        "risk_pattern_score": trace_summary.get("risk_pattern_score"),
        "ordinary_use_pattern": ordinary_pattern,
        "coarse_rates": coarse_rates,
        "coarse_levels": coarse_levels,
        "phase_counts": trace_summary.get("phase_counts", {}),
    }
    return package


def build_llm_evidence_package(sample_id: str, evidence: dict[str, Any]) -> dict[str, Any]:
    """Build a label-free evidence package that can be rendered for LLM input."""

    features = evidence["feature_blocks"]
    video = features["video_proxy_summary"]
    quality = features["quality_summary"]

    if video.get("privte_trace_features"):
        behavior_package = _build_trace_v1_package(video, quality)
    elif video.get("behavior_v3_temporal_features"):
        behavior_package = _build_behavior_v3_temporal_package(video, quality)
    elif video.get("behavior_v2_features"):
        behavior_package = _build_behavior_v2_package(video, quality)
    elif video.get("behavior_v1_features"):
        behavior_package = _build_behavior_v1_package(video, quality)
    elif video.get("global_features"):
        behavior_package = _build_flowlite_behavior_package(video, quality)
    else:
        behavior_package = _build_non_behavior_package(video, quality)

    privacy = evidence["privacy_processing_summary"]
    missing_information = _llm_visible_missing_information(evidence["missing_information"])

    return {
        "schema_version": "mdu_riskbench_llm_evidence.v0",
        "sample_id": sample_id,
        "task": {
            "name": "Text-only Minor Digital Use Risk Screening",
            "instruction": (
                "基于 PriVTE 生成的隐私过滤行为证据，判断未成年人数字设备使用风险等级。"
            ),
            "not_diagnosis": True,
            "allowed_risk_levels": RISK_LEVELS,
        },
        "decision_rules": [
            "只能使用本 evidence package 中的文本证据。",
            "将本 evidence package 视为该评测样本的完整输入，不要推断未呈现的额外模态。",
            "不要把行为线索解释为医学诊断、心理状态确认或成瘾结论。",
            "只有当本证据包内部明确显示质量、可见性或行为证据不足时，才考虑 insufficient_evidence。",
            "missing_information 只填写证据包内部明确说明的质量、可见性或行为证据不足；否则可为空列表。",
            "输出必须同时说明支持证据、降低确定性的证据、缺失信息和是否需要人工复核。",
        ],
        "screening_rubric": {
            "no_observed_risk": (
                "可存在普通或低强度设备使用，但未观察到持续、高频、重复或跨阶段一致的风险性使用模式，且证据质量足够。"
            ),
            "mild_risk": (
                "存在局部或轻度风险性使用迹象，例如短时稳定参与或少量交互增强，但持续性、重复性或跨阶段一致性有限。"
            ),
            "moderate_risk": (
                "同时存在较持续参与、较高交互密度、重复操作或跨阶段一致的风险性使用模式，且质量可用。"
            ),
            "high_risk": (
                "证据显示强持续参与、强重复交互和多阶段一致的风险性使用模式；仍需人工复核，不能视为诊断。"
            ),
            "insufficient_evidence": (
                "证据包内部质量、可见性、关键行为指标或覆盖度不足，无法支持可靠筛查判断。"
            ),
        },
        "observation_scope": {
            "input_modalities_used_for_model": ["privacy_filtered_behavior_text"],
            "video_clips_available": video.get("num_video_clips", 0),
            "selected_video_clips_analyzed": video.get("selected_video_files"),
            "sampled_frame_count": video.get("sampled_frame_count"),
            "coverage_relative_position_counts": video.get("key_window_summary", {}).get(
                "coverage_relative_position_counts",
                {},
            ),
            "quality_overall": quality.get("overall", "unknown"),
        },
        **behavior_package,
        "missing_information": missing_information,
        "privacy_constraints": {
            "raw_video_included": privacy.get("raw_video_included", False),
            "raw_images_included": privacy.get("raw_images_included", False),
            "raw_audio_included": privacy.get("raw_audio_included", False),
            "ocr_text_included": privacy.get("ocr_text_included", False),
            "asr_text_included": privacy.get("asr_text_included", False),
            "face_embeddings_included": privacy.get("face_embeddings_included", False),
            "high_dimensional_landmarks_included": privacy.get(
                "high_dimensional_landmarks_included",
                False,
            ),
            "questionnaire_answers_included": privacy.get(
                "questionnaire_answers_included",
                False,
            ),
            "exact_heart_rate_values_included": privacy.get(
                "exact_heart_rate_values_included",
                False,
            ),
            "app_names_included": privacy.get("app_names_included", False),
            "raw_paths_included": privacy.get("raw_paths_included", False),
            "exact_timestamps_included": privacy.get("exact_timestamps_included", False),
        },
        "requested_model_output": {
            "format": "json",
            "fields": {
                "risk_level": RISK_LEVELS,
                "confidence": ["low", "medium", "high"],
                "supporting_evidence": "list[str]",
                "uncertainty_or_counter_evidence": "list[str]",
                "missing_information": "list[str]",
                "needs_human_review": "bool",
            },
        },
    }
