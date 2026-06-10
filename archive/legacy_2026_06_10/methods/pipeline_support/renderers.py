"""Text renderers for PriVTE evidence records."""

from __future__ import annotations

from typing import Any


def text_bool(value: bool) -> str:
    return "yes" if value else "no"


def render_feature_item(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return "; ".join(f"{key}={value}" for key, value in item.items())
    return str(item)


def render_bool_flag(value: bool) -> str:
    return "False" if value is False else "True"


def render_dict_items(values: dict[str, Any]) -> str:
    if not values:
        return "none"
    return ", ".join(
        f"{key}={render_measurement_value(value)}" for key, value in values.items()
    )


def render_measurement_value(value: Any) -> str:
    if isinstance(value, dict):
        return "{" + render_dict_items(value) + "}"
    return str(value)


def render_optional(value: Any) -> str:
    if value is None:
        return "unknown"
    return str(value)


def render_signal_strength(value: Any) -> str:
    return render_optional(value).replace("_video_proxy_", "_behavior_").replace(
        "video_proxy_",
        "behavior_",
    )


def normalize_llm_prompt_text(text: str) -> str:
    replacements = {
        "视频代理证据": "行为证据",
        "视频代理线索": "行为线索",
        "代理证据": "行为证据",
        "代理线索": "行为线索",
        "代理": "指标",
        "_proxy": "_indicator",
        "proxy": "indicator",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def render_temporal_step_evidence(evidence: dict[str, Any]) -> str:
    keys = [
        "sampled_points",
        "device_visible_share",
        "stable_screen_share",
        "active_operation_share",
        "direct_operation_share",
        "usable_quality_share",
        "confounded_activity_share",
        "hand_device_proximity_points",
        "unconfounded_device_activity_points",
        "repetitive_operation_points",
        "visible_without_engagement_points",
    ]
    return render_dict_items(
        {key: evidence.get(key) for key in keys if key in evidence}
    )


def build_text_evidence(record: dict[str, Any]) -> str:
    package = record["llm_evidence_package"]
    task = package["task"]
    scope = package["observation_scope"]
    synthesis = package["risk_relevant_synthesis"]
    requested_output = package["requested_model_output"]["fields"]

    lines = [
        f"样本编号: {record['sample_id']}",
        f"任务: {task['name']}",
        f"任务说明: {task['instruction']}",
        "边界: 这是风险筛查输出，不是医学诊断或成瘾诊断。",
        "允许风险等级: " + ", ".join(task["allowed_risk_levels"]),
        "",
        "模型判定规则:",
    ]
    lines.extend(f"- {rule}" for rule in package["decision_rules"])
    lines.extend(
        [
            "",
            "风险筛查 rubric:",
        ]
    )
    lines.extend(
        f"- {level}: {description}"
        for level, description in package["screening_rubric"].items()
    )
    lines.extend(
        [
            "",
            "观察范围:",
            "- 模型输入模态: " + ", ".join(scope["input_modalities_used_for_model"]),
            f"- 视频片段数量: {scope['video_clips_available']}",
            "- 已抽样分析视频片段: "
            + render_optional(scope.get("selected_video_clips_analyzed")),
            "- 已抽样分析帧数: " + render_optional(scope.get("sampled_frame_count")),
            "- 覆盖位置计数: "
            + render_dict_items(scope.get("coverage_relative_position_counts", {})),
            f"- 证据质量总评: {scope['quality_overall']}",
            "",
            "行为证据观察:",
        ]
    )
    trace_summary = package.get("trace_risk_summary", {})
    trace_narrative = package.get("trace_behavior_narrative", [])
    if trace_summary:
        lines.append("隐私安全风险性使用轨迹摘要:")
        for item in trace_narrative:
            lines.append(f"- {item}")
        lines.append(
            "- Trace核心结论: "
            + render_dict_items(
                {
                    "risk_pattern_strength": trace_summary.get(
                        "risk_pattern_strength"
                    ),
                    "risk_pattern_score": trace_summary.get("risk_pattern_score"),
                    "ordinary_use_pattern": trace_summary.get("ordinary_use_pattern"),
                    "quality_overall": trace_summary.get("quality_overall"),
                    "sampled_points": trace_summary.get("sampled_points"),
                }
            )
        )
        lines.append(
            "- Trace粗粒度比例: "
            + render_dict_items(trace_summary.get("coarse_rates", {}))
        )
        lines.append(
            "- Trace粗粒度等级: "
            + render_dict_items(trace_summary.get("coarse_levels", {}))
        )
        risk_indicators = trace_summary.get("risk_pattern_indicators", [])
        ordinary_indicators = trace_summary.get("ordinary_or_low_risk_indicators", [])
        counter_indicators = trace_summary.get("counter_risk_indicators", [])
        if risk_indicators:
            lines.append("风险性使用模式支持证据:")
            lines.extend(f"- {item}" for item in risk_indicators)
        if ordinary_indicators:
            lines.append("普通或低风险使用解释证据:")
            lines.extend(f"- {item}" for item in ordinary_indicators)
        if counter_indicators:
            lines.append("降低风险强度或确定性的证据:")
            lines.extend(f"- {item}" for item in counter_indicators)
        lines.append("")

    temporal_narrative = package.get("temporal_behavior_narrative", [])
    temporal_sequence = package.get("temporal_behavior_sequence", [])
    temporal_summary = package.get("temporal_sequence_summary", {})
    if temporal_narrative or temporal_sequence:
        lines.append("隐私安全时序行为摘要:")
        if temporal_summary:
            lines.append(
                "- 时序类型: "
                + render_optional(temporal_summary.get("sequence_type"))
                + "；主要模式: "
                + render_optional(temporal_summary.get("main_observation_pattern"))
            )
        if temporal_narrative:
            lines.extend(f"- {item}" for item in temporal_narrative)
        if temporal_summary:
            temporal_signal_strength = None
            raw_temporal_signal = temporal_summary.get(
                "temporal_video_proxy_signal_strength"
            )
            if raw_temporal_signal:
                temporal_signal_strength = render_signal_strength(raw_temporal_signal)
            lines.append(
                "- 时序汇总: "
                + render_dict_items(
                    {
                        "dominant_states": ", ".join(
                            temporal_summary.get("dominant_states", []) or ["none"]
                        ),
                        "candidate_episode_count": temporal_summary.get(
                            "candidate_episode_count"
                        ),
                        "selected_episode_count": temporal_summary.get(
                            "selected_episode_count"
                        ),
                        "engagement_episode_count": temporal_summary.get(
                            "engagement_episode_count"
                        ),
                        "passive_engagement_episode_count": temporal_summary.get(
                            "passive_engagement_episode_count"
                        ),
                        "active_operation_step_count": temporal_summary.get(
                            "active_operation_step_count"
                        ),
                        "active_operation_point_count": temporal_summary.get(
                            "active_operation_point_count"
                        ),
                        "direct_operation_point_count": temporal_summary.get(
                            "direct_operation_point_count"
                        ),
                        "repetitive_operation_step_count": temporal_summary.get(
                            "repetitive_operation_step_count"
                        ),
                        "repetitive_operation_point_count": temporal_summary.get(
                            "repetitive_operation_point_count"
                        ),
                        "stable_screen_step_count": temporal_summary.get(
                            "stable_screen_step_count"
                        ),
                        "stable_screen_point_count": temporal_summary.get(
                            "stable_screen_point_count"
                        ),
                        "confounded_activity_step_count": temporal_summary.get(
                            "confounded_activity_step_count"
                        ),
                        "confounded_activity_point_count": temporal_summary.get(
                            "confounded_activity_point_count"
                        ),
                        "visible_without_engagement_point_count": temporal_summary.get(
                            "visible_without_engagement_point_count"
                        ),
                        "temporal_signal_strength": temporal_signal_strength,
                    }
                )
            )
        if temporal_sequence:
            lines.append("隐私安全时序状态序列:")
            for step in temporal_sequence:
                lines.append(
                    "- "
                f"{step['step_id']}: "
                f"位置={step['relative_position']}; "
                f"状态={step['state']}; "
                f"覆盖={step['duration_bin']}; "
                f"置信度={step['confidence']}; "
                f"支持级别={step.get('support_level', 'unknown')}; "
                f"选择原因={step.get('selection_reason', 'unknown')}; "
                "证据="
                + render_temporal_step_evidence(step.get("evidence", {}))
            )
        lines.append("")

    concrete_evidence = package.get("concrete_behavior_evidence", [])
    if concrete_evidence:
        lines.append("隐私过滤行为证据:")
        for item in concrete_evidence:
            lines.extend(
                [
                    f"- 证据类型: {item['evidence_type']}",
                    f"  观察内容: {item['observation']}",
                    "  粗粒度计量: "
                    + render_dict_items(item.get("measurement", {})),
                ]
            )
        lines.append("")

    observations = package["behavior_observations"]
    if observations:
        for item in observations:
            lines.extend(
                [
                    f"- 信号: {item['signal']}",
                    f"  取值: {item['value']}",
                    f"  行为解释: {item['interpretation']}",
                    f"  证据质量: {item['quality']}",
                ]
            )
    else:
        lines.append("- 当前 evidence package 没有可用行为代理观察。")

    lines.append("")
    lines.append("关键事件窗口:")
    event_windows = package["key_event_windows"]
    if event_windows:
        for event in event_windows:
            lines.append(
                "- "
                f"{event['window_id']}: "
                f"位置={event['relative_position']}; "
                f"观察={event['observation']}; "
                f"强度={event['strength']}; "
                f"质量={event['quality']}"
            )
    else:
        lines.append("- none")

    signal_strength = render_signal_strength(synthesis["video_proxy_signal_strength"])
    lines.extend(
        [
            "- 关键事件类型计数: "
            + render_dict_items(package.get("key_event_window_counts", {})),
            "",
            "风险相关综合:",
            f"- 行为证据强度: {signal_strength}",
            "- 支持进一步关注的证据:",
        ]
    )
    lines.extend(
        f"  - {item}" for item in synthesis["signals_supporting_closer_review"]
    )
    lines.append("- 降低确定性或反向证据:")
    lines.extend(f"  - {item}" for item in synthesis["signals_reducing_certainty"])
    lines.extend(
        [
            f"- 质量建议: {synthesis['quality_recommendation']}",
        ]
    )
    visible_missing = package.get("missing_information", [])
    if visible_missing:
        lines.extend(["", "证据包内部缺失或限制:"])
        lines.extend(f"- {item}" for item in visible_missing)
    lines.extend(
        [
            "",
            "隐私处理:",
            "- PriVTE 已完成隐私过滤；模型只需根据本 evidence package 作出筛查判断。",
        ]
    )
    lines.extend(
        [
            "",
            "要求模型输出 JSON:",
            f"- risk_level: {requested_output['risk_level']}",
            f"- confidence: {requested_output['confidence']}",
            "- supporting_evidence: list[str]",
            "- uncertainty_or_counter_evidence: list[str]",
            "- missing_information: list[str]",
            "- needs_human_review: bool",
        ]
    )
    return normalize_llm_prompt_text("\n".join(lines))
