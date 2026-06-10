#!/bin/bash

# 如果任何一个命令失败，立即停止执行
set -e

echo "=== 开始执行：刷新 LLM 证据文本 (Refresh Evidence Text) ==="

uv run python scripts/refresh_llm_evidence_text.py \
    --input-jsonl outputs/quickstart/all_current_privte_flowlite/flowlite_mvp_evidence.all_current.jsonl \
    --output-jsonl outputs/quickstart/all_current_privte_flowlite/flowlite_mvp_evidence.all_current.refreshed.jsonl \
    --overwrite

uv run python scripts/refresh_llm_evidence_text.py \
    --input-jsonl outputs/quickstart/all_current_privte_behavior_v1/behavior_v1_mvp_evidence.all_current.jsonl \
    --output-jsonl outputs/quickstart/all_current_privte_behavior_v1/behavior_v1_mvp_evidence.all_current.refreshed.jsonl \
    --overwrite

uv run python scripts/refresh_llm_evidence_text.py \
    --input-jsonl outputs/quickstart/all_current_privte_behavior_v2/behavior_v2_mvp_evidence.all_current.jsonl \
    --output-jsonl outputs/quickstart/all_current_privte_behavior_v2/behavior_v2_mvp_evidence.all_current.refreshed.jsonl \
    --overwrite

uv run python scripts/refresh_llm_evidence_text.py \
    --input-jsonl outputs/quickstart/all_current_privte_behavior_v3_temporal/behavior_v3_temporal_mvp_evidence.all_current.jsonl \
    --output-jsonl outputs/quickstart/all_current_privte_behavior_v3_temporal/behavior_v3_temporal_mvp_evidence.all_current.refreshed.jsonl \
    --overwrite

uv run python scripts/refresh_llm_evidence_text.py \
    --input-jsonl outputs/quickstart/all_current_privte_behavior_v3_temporal_episode/behavior_v3_temporal_mvp_evidence.all_current.jsonl \
    --output-jsonl outputs/quickstart/all_current_privte_behavior_v3_temporal_episode/behavior_v3_temporal_mvp_evidence.all_current.refreshed.jsonl \
    --overwrite


echo "=== 开始执行：运行 LLM 基线评估 (Run LLM Baseline) ==="

uv run python experiments/llm/run_llm_baseline.py \
    --evidence-jsonl outputs/quickstart/all_current_privte_flowlite/flowlite_mvp_evidence.all_current.refreshed.jsonl \
    --evidence-tag all_current_privte_flowlite_refreshed \
    --model gpt-5.5 \
    --model-tag gpt-5-5 \
    --token-budget-param max_completion_tokens \
    --max-tokens 25000 \
    --sleep-sec 2 \
    --overwrite

uv run python experiments/llm/run_llm_baseline.py \
    --evidence-jsonl outputs/quickstart/all_current_privte_behavior_v1/behavior_v1_mvp_evidence.all_current.refreshed.jsonl \
    --evidence-tag all_current_privte_behavior_v1_refreshed \
    --model gpt-5.5 \
    --model-tag gpt-5-5 \
    --token-budget-param max_completion_tokens \
    --max-tokens 25000 \
    --sleep-sec 2 \
    --overwrite

uv run python experiments/llm/run_llm_baseline.py \
    --evidence-jsonl outputs/quickstart/all_current_privte_behavior_v2/behavior_v2_mvp_evidence.all_current.refreshed.jsonl \
    --evidence-tag all_current_privte_behavior_v2_refreshed \
    --model gpt-5.5 \
    --model-tag gpt-5-5 \
    --token-budget-param max_completion_tokens \
    --max-tokens 25000 \
    --sleep-sec 2 \
    --overwrite

uv run python experiments/llm/run_llm_baseline.py \
    --evidence-jsonl outputs/quickstart/all_current_privte_behavior_v3_temporal/behavior_v3_temporal_mvp_evidence.all_current.refreshed.jsonl \
    --evidence-tag all_current_privte_behavior_v3_temporal_refreshed \
    --model gpt-5.5 \
    --model-tag gpt-5-5 \
    --token-budget-param max_completion_tokens \
    --max-tokens 25000 \
    --sleep-sec 2 \
    --overwrite

uv run python experiments/llm/run_llm_baseline.py \
    --evidence-jsonl outputs/quickstart/all_current_privte_behavior_v3_temporal_episode/behavior_v3_temporal_mvp_evidence.all_current.refreshed.jsonl \
    --evidence-tag all_current_privte_behavior_v3_temporal_episode_refreshed \
    --model gpt-5.5 \
    --model-tag gpt-5-5 \
    --token-budget-param max_completion_tokens \
    --max-tokens 25000 \
    --sleep-sec 2 \
    --overwrite

echo "=== 所有任务执行完毕！ ==="