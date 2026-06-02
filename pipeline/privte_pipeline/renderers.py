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


def build_text_evidence(record: dict[str, Any]) -> str:
    evidence = record["evidence"]
    modality = evidence["modality_availability"]
    features = evidence["feature_blocks"]
    video = features["video_proxy_summary"]
    heart_rate = features["heart_rate_summary"]
    app_usage = features["app_usage_summary"]
    questionnaire = features["questionnaire_status"]
    quality = features["quality_summary"]
    limitations = evidence["limitations"]
    missing = evidence["missing_information"]
    privacy = evidence["privacy_processing_summary"]

    lines = [
        f"样本编号: {record['sample_id']}",
        "输入类型: text-only PriVTE evidence",
        "任务定位: 基于可观察代理证据的未成年人数字设备使用风险筛查，不是诊断。",
        "",
        "模态可用性:",
        f"- 视频片段: {text_bool(modality['has_video'])}, 数量 {video['num_video_clips']}",
        f"- 心率记录: {text_bool(modality['has_heart_rate'])}, 非空片段 {heart_rate['num_nonempty_heart_rate_clips']}",
        f"- 应用使用记录: {text_bool(modality['has_app_usage'])}, 记录数 {app_usage['num_usage_records']}",
        f"- 问卷: {text_bool(questionnaire['available'])}, 不作为 text-only 视频代理输入证据",
        "",
        "视频代理证据:",
    ]
    if video["visual_proxy_features"]:
        lines.extend(f"- {render_feature_item(item)}" for item in video["visual_proxy_features"])
    else:
        lines.append("- 当前 extractor 尚未提供视觉行为代理特征。")
    computed_quality_fields = quality.get("computed_quality_fields", [])
    lines.extend(
        [
            "- LLM 输入不包含视频、图像、音频或关键帧。",
            "",
            "质量与缺失信息:",
            f"- 质量评估状态: {quality['status']}",
            "- 已计算质量字段: "
            + (", ".join(computed_quality_fields) if computed_quality_fields else "none"),
            "- 缺失字段: " + ", ".join(missing),
            "",
            "隐私处理:",
            f"- raw video included: {privacy['raw_video_included']}",
            f"- raw images included: {privacy['raw_images_included']}",
            f"- questionnaire answers included: {privacy['questionnaire_answers_included']}",
            f"- exact heart-rate values included: {privacy['exact_heart_rate_values_included']}",
            f"- app names included: {privacy['app_names_included']}",
            "",
            "限制:",
        ]
    )
    lines.extend(f"- {item}" for item in limitations)
    return "\n".join(lines)
