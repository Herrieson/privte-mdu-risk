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
        return "证据质量不足，模型应优先考虑 insufficient_evidence 或人工复核。"
    if overall in {
        "partial_frame_quality",
        "partial_behavior_frame_quality",
        "file_level_quality_only",
    }:
        return "证据质量部分可用，模型应降低置信度并更积极触发人工复核。"
    if overall in {
        "usable_frame_quality",
        "usable_behavior_frame_quality",
        "usable_container_quality",
    }:
        return "证据质量可用，但结论仍只能是基于视频代理证据的风险筛查。"
    return "证据质量未知，模型应谨慎判断并说明不确定性。"


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
        uncertainty.append("有效帧质量不足够高，视频代理证据的可靠性下降。")

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
        risk_relevance="该信号只表示能否观察面向屏幕相关线索；本版不做身份、人脸嵌入或精确凝视估计。",
        quality=overall_quality,
    )
    if face_observability == "high":
        supporting.append("人脸与设备/屏幕样区域共现较高，屏幕相关行为观察更可靠。")
    elif face_observability == "medium":
        supporting.append("人脸相关可观察性为中等，部分屏幕相关行为线索可用。")
    elif face_observability == "low":
        uncertainty.append("人脸与设备/屏幕样区域共现较低，不能可靠判断凝视、表情或面向屏幕状态。")

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

    _append_observation(
        observations,
        signal="affect_fatigue_or_gaze_proxy",
        value="not_available_in_flowlite_v0",
        interpretation="本版没有直接眼动、眨眼、面部动作单元、负向表情或疲劳趋势估计。",
        risk_relevance="不能根据本输出判断主观渴求、挫败、疲劳、情绪或成瘾状态。",
        quality=overall_quality,
    )
    uncertainty.append("本版缺少直接手部关键点、精确凝视、表情动作单元和疲劳趋势证据。")

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
        supporting.append("未形成明确支持风险升高的高质量视频代理证据。")

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
                "缺少屏幕参与、交互突增、姿态、手部、注视和表情代理特征。"
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
        risk_relevance="该信号表示面向屏幕相关线索是否可观察；不是精确 gaze。",
        quality=overall_quality,
    )
    if face_context in {"none", "low"}:
        uncertainty.append("人脸-设备上下文较弱，不能可靠判断面向屏幕或凝视状态。")

    _append_observation(
        observations,
        signal="posture_or_context_change_proxy",
        value=posture_change,
        interpretation="由姿态中心变化和全局运动突增聚合得到。",
        risk_relevance="可作为姿态变化、离开/返回或拍摄场景扰动的弱代理线索。",
        quality=overall_quality,
    )

    _append_observation(
        observations,
        signal="affect_fatigue_or_gaze_proxy",
        value="not_validated_in_behavior_v1",
        interpretation="本版未输出经验证的情绪、疲劳或精确凝视证据。",
        risk_relevance="不能根据本输出判断主观渴求、挫败、疲劳、情绪或成瘾状态。",
        quality=overall_quality,
    )
    uncertainty.append("本版手-设备交互、头部朝向和姿态变化仍是代理证据，不是临床或主观状态标签。")

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
            or ["未形成明确支持风险升高的高质量视频代理证据。"],
            "signals_reducing_certainty": uncertainty,
            "quality_recommendation": _quality_recommendation(overall_quality),
        },
    }


def build_llm_evidence_package(sample_id: str, evidence: dict[str, Any]) -> dict[str, Any]:
    """Build a label-free evidence package that can be rendered for LLM input."""

    features = evidence["feature_blocks"]
    video = features["video_proxy_summary"]
    quality = features["quality_summary"]
    modality = evidence["modality_availability"]

    if video.get("behavior_v1_features"):
        behavior_package = _build_behavior_v1_package(video, quality)
    elif video.get("global_features"):
        behavior_package = _build_flowlite_behavior_package(video, quality)
    else:
        behavior_package = _build_non_behavior_package(video, quality)

    privacy = evidence["privacy_processing_summary"]
    missing_information = evidence["missing_information"]

    return {
        "schema_version": "mdu_riskbench_llm_evidence.v0",
        "sample_id": sample_id,
        "task": {
            "name": "Text-only Minor Digital Use Risk Screening",
            "instruction": (
                "基于 PriVTE 生成的隐私过滤视频代理证据，判断未成年人数字设备使用风险等级。"
            ),
            "not_diagnosis": True,
            "allowed_risk_levels": RISK_LEVELS,
        },
        "decision_rules": [
            "只能使用本 evidence package 中的文本证据。",
            "不要把视频代理线索解释为医学诊断、心理状态确认或成瘾结论。",
            "证据质量低或关键模态缺失时，应降低置信度并考虑 insufficient_evidence。",
            "输出必须同时说明支持证据、降低确定性的证据、缺失信息和是否需要人工复核。",
        ],
        "screening_rubric": {
            "no_observed_risk": (
                "未观察到持续屏幕参与、频繁交互或重复操作代理证据，且证据质量足够。"
            ),
            "mild_risk": (
                "存在一定屏幕参与或交互代理证据，但强度、持续性或一致性有限。"
            ),
            "moderate_risk": (
                "同时存在较持续屏幕参与和较频繁设备交互代理证据，且质量可用。"
            ),
            "high_risk": (
                "视频代理证据显示强持续屏幕参与、强重复交互和多窗口一致性；仍需人工复核，不能视为诊断。"
            ),
            "insufficient_evidence": (
                "视频质量、可见性、关键行为代理或覆盖度不足，无法支持可靠筛查判断。"
            ),
        },
        "observation_scope": {
            "input_modalities_used_for_model": ["video_proxy_text"],
            "available_but_not_used_as_input": {
                "heart_rate_available": modality.get("has_heart_rate", False),
                "app_usage_available": modality.get("has_app_usage", False),
                "questionnaire_available": modality.get("has_questionnaire", False),
            },
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
