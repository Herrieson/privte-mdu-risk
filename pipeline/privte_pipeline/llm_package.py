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


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _video_summary(feature_blocks: dict[str, Any]) -> dict[str, Any]:
    return _as_dict(feature_blocks.get("video_proxy_summary"))


def _preprocessor_evidence(feature_blocks: dict[str, Any]) -> dict[str, Any]:
    return _as_dict(feature_blocks.get("preprocessor_evidence"))


def _session_metadata(feature_blocks: dict[str, Any]) -> dict[str, Any]:
    preprocessor = _preprocessor_evidence(feature_blocks)
    video = _video_summary(feature_blocks)
    return _as_dict(preprocessor.get("session_metadata") or video.get("session_metadata"))


def _global_features(feature_blocks: dict[str, Any]) -> dict[str, Any]:
    preprocessor = _preprocessor_evidence(feature_blocks)
    if preprocessor.get("global_features"):
        return _as_dict(preprocessor.get("global_features"))
    video = _video_summary(feature_blocks)
    return _as_dict(video.get("global_features") or feature_blocks.get("global_features"))


def _event_windows(feature_blocks: dict[str, Any]) -> list[Any]:
    preprocessor = _preprocessor_evidence(feature_blocks)
    if preprocessor.get("event_windows"):
        return _as_list(preprocessor.get("event_windows"))
    video = _video_summary(feature_blocks)
    return _as_list(video.get("event_windows") or feature_blocks.get("event_windows"))


def _quality_summary(feature_blocks: dict[str, Any]) -> dict[str, Any]:
    preprocessor = _preprocessor_evidence(feature_blocks)
    if preprocessor.get("quality_summary"):
        return _as_dict(preprocessor.get("quality_summary"))
    video = _video_summary(feature_blocks)
    return _as_dict(feature_blocks.get("quality_summary") or video.get("quality_summary"))


def _stateful_behavior_summary(global_features: dict[str, Any]) -> dict[str, Any]:
    return _as_dict(global_features.get("stateful_behavior_summary"))


def build_llm_evidence_package(sample_id: str, evidence: dict[str, Any]) -> dict[str, Any]:
    """Build a label-free, schema-first evidence package for LLM input."""

    feature_blocks = _as_dict(evidence.get("feature_blocks"))
    video = _video_summary(feature_blocks)
    session_metadata = _session_metadata(feature_blocks)
    quality = _quality_summary(feature_blocks)
    global_features = _global_features(feature_blocks)
    event_windows = _event_windows(feature_blocks)
    stateful_behavior = _stateful_behavior_summary(global_features)

    package = {
        "schema_version": "mdu_riskbench_llm_evidence.v1",
        "sample_id": sample_id,
        "task": {
            "name": "Text-only Minor Digital Use Risk Screening",
            "instruction": (
                "基于 PriVTE 生成的隐私过滤视频行为证据，判断未成年人数字设备使用风险等级。"
            ),
            "not_diagnosis": True,
            "allowed_risk_levels": RISK_LEVELS,
        },
        "decision_rules": [
            "只能使用本 evidence package 中的文本证据。",
            "不要使用证据包未呈现的视频、音频、问卷、心率或应用信息。",
            "按证据角色、窗口数量、持续性和质量字段综合判断风险等级。",
            "关键行为证据或可见性不足时，降低置信度并考虑 insufficient_evidence。",
            "输出必须说明支持证据、降低确定性的证据、缺失信息和是否需要人工复核。",
        ],
        "screening_rubric": {
            "no_observed_risk": (
                "未观察到持续屏幕参与、频繁交互、重复操作或跨窗口一致的风险性行为证据，且证据质量足够。"
            ),
            "mild_risk": (
                "存在局部或轻度风险性使用迹象，但强度、持续性、重复性或一致性有限。"
            ),
            "moderate_risk": (
                "同时存在较持续参与、较高交互密度、重复操作或多个事件窗口一致的风险性使用模式。"
            ),
            "high_risk": (
                "证据显示强持续参与、强重复交互和多窗口一致的风险性使用模式。"
            ),
            "insufficient_evidence": (
                "证据包内部质量、可见性、关键行为指标或覆盖度不足，无法支持可靠筛查判断。"
            ),
        },
        "observation_scope": {
            "input_modalities_used_for_model": ["privacy_filtered_video_text"],
            "video_status": video.get("status", "unknown"),
            "video_clips_available": video.get("num_video_clips"),
            "selected_video_clips_analyzed": video.get("selected_video_clips_analyzed"),
            "sampled_frame_count": video.get("sampled_frame_count"),
            "coverage_relative_position_counts": video.get(
                "coverage_relative_position_counts", {}
            ),
            "quality_overall": quality.get(
                "overall",
                quality.get("overall_data_sufficiency", "unknown"),
            ),
        },
        "session_metadata": session_metadata,
        "global_features": global_features,
        "event_windows": event_windows,
        "quality_summary": quality,
        "limitations": _as_list(evidence.get("limitations")),
        "missing_information": _as_list(evidence.get("missing_information")),
        "privacy_processing_summary": _as_dict(
            evidence.get("privacy_processing_summary")
        ),
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
    if stateful_behavior:
        package["stateful_behavior"] = stateful_behavior
    return package
