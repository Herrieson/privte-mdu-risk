"""Text renderers for PriVTE evidence records."""

from __future__ import annotations

from typing import Any


def _render_value(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, dict):
        return "{" + _render_dict(value) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(_render_value(item) for item in value) + "]"
    return str(value)


def _render_dict(values: dict[str, Any]) -> str:
    if not values:
        return "none"
    return ", ".join(f"{key}={_render_value(value)}" for key, value in values.items())


def _render_event(event: Any, index: int) -> str:
    if isinstance(event, dict):
        event_id = event.get("event_id") or event.get("window_id") or f"event_{index:02d}"
        fields = {key: value for key, value in event.items() if key not in {"event_id", "window_id"}}
        return f"{event_id}: {_render_dict(fields)}"
    return f"event_{index:02d}: {_render_value(event)}"


def build_text_evidence(record: dict[str, Any]) -> str:
    package = record["llm_evidence_package"]
    task = package["task"]
    scope = package["observation_scope"]
    session_metadata = package.get("session_metadata", {})
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
    lines.extend(["", "风险筛查 rubric:"])
    lines.extend(
        f"- {level}: {description}"
        for level, description in package["screening_rubric"].items()
    )

    lines.extend(
        [
            "",
            "观察范围:",
            "- 模型输入模态: " + ", ".join(scope["input_modalities_used_for_model"]),
            f"- 视频状态: {scope.get('video_status', 'unknown')}",
            f"- 视频片段数量: {_render_value(scope.get('video_clips_available'))}",
            "- 已分析视频片段: "
            + _render_value(scope.get("selected_video_clips_analyzed")),
            "- 已抽样分析帧数: " + _render_value(scope.get("sampled_frame_count")),
            "- 覆盖位置计数: "
            + _render_dict(scope.get("coverage_relative_position_counts", {})),
            f"- 证据质量总评: {scope.get('quality_overall', 'unknown')}",
            "",
            "会话元数据:",
        ]
    )
    if session_metadata:
        lines.extend(
            f"- {key}: {_render_value(value)}"
            for key, value in session_metadata.items()
        )
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "全局统计信息:",
        ]
    )

    global_features = package.get("global_features", {})
    if global_features:
        lines.extend(
            f"- {key}: {_render_value(value)}"
            for key, value in global_features.items()
        )
    else:
        lines.append("- none")

    lines.extend(["", "事件窗口:"])
    event_windows = package.get("event_windows", [])
    if event_windows:
        lines.extend(
            f"- {_render_event(event, index)}"
            for index, event in enumerate(event_windows, start=1)
        )
    else:
        lines.append("- none")

    lines.extend(["", "质量摘要:"])
    quality_summary = package.get("quality_summary", {})
    if quality_summary:
        lines.extend(
            f"- {key}: {_render_value(value)}"
            for key, value in quality_summary.items()
        )
    else:
        lines.append("- none")

    limitations = package.get("limitations", [])
    if limitations:
        lines.extend(["", "限制说明:"])
        lines.extend(f"- {item}" for item in limitations)

    missing_information = package.get("missing_information", [])
    if missing_information:
        lines.extend(["", "缺失信息:"])
        lines.extend(f"- {item}" for item in missing_information)

    privacy = package.get("privacy_processing_summary", {})
    lines.extend(["", "隐私处理:"])
    if privacy:
        lines.extend(f"- {key}: {_render_value(value)}" for key, value in privacy.items())
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "要求模型输出 JSON:",
            f"- risk_level: {requested_output['risk_level']}",
            f"- confidence: {requested_output['confidence']}",
            f"- supporting_evidence: {requested_output['supporting_evidence']}",
            "- uncertainty_or_counter_evidence: "
            + requested_output["uncertainty_or_counter_evidence"],
            f"- missing_information: {requested_output['missing_information']}",
            f"- needs_human_review: {requested_output['needs_human_review']}",
        ]
    )
    return "\n".join(lines)
