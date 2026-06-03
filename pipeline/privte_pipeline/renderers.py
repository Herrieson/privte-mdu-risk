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
    return ", ".join(f"{key}={value}" for key, value in values.items())


def render_optional(value: Any) -> str:
    if value is None:
        return "unknown"
    return str(value)


def build_text_evidence(record: dict[str, Any]) -> str:
    package = record["llm_evidence_package"]
    task = package["task"]
    scope = package["observation_scope"]
    available_not_used = scope["available_but_not_used_as_input"]
    synthesis = package["risk_relevant_synthesis"]
    privacy = package["privacy_constraints"]
    requested_output = package["requested_model_output"]["fields"]

    lines = [
        f"样本编号: {record['sample_id']}",
        f"任务: {task['name']}",
        f"任务说明: {task['instruction']}",
        "边界: 这是基于视频代理证据的风险筛查，不是成瘾诊断。",
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
            "- 心率可用但不作为输入: "
            + text_bool(available_not_used["heart_rate_available"]),
            "- 应用记录可用但不作为输入: "
            + text_bool(available_not_used["app_usage_available"]),
            "- 问卷可用但不作为输入: "
            + text_bool(available_not_used["questionnaire_available"]),
            "",
            "行为代理观察:",
        ]
    )
    observations = package["behavior_observations"]
    if observations:
        for item in observations:
            lines.extend(
                [
                    f"- 信号: {item['signal']}",
                    f"  取值: {item['value']}",
                    f"  行为解释: {item['interpretation']}",
                    f"  风险相关性: {item['risk_relevance']}",
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
                f"质量={event['quality']}; "
                f"风险相关性={event['risk_relevance']}"
            )
    else:
        lines.append("- none")

    lines.extend(
        [
            "- 关键事件类型计数: "
            + render_dict_items(package.get("key_event_window_counts", {})),
            "",
            "风险相关综合:",
            f"- 视频代理证据强度: {synthesis['video_proxy_signal_strength']}",
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
            "",
            "缺失信息:",
        ]
    )
    lines.extend(f"- {item}" for item in package["missing_information"])
    lines.extend(
        [
            "",
            "隐私约束:",
        ]
    )
    lines.extend(
        f"- {key}: {render_bool_flag(value)}"
        for key, value in privacy.items()
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
    return "\n".join(lines)
