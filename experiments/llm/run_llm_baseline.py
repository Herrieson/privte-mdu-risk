#!/usr/bin/env python3
"""Run a text-only LLM baseline on PriVTE evidence JSONL.

This runner is intentionally provider-light. It calls any OpenAI-compatible
`/chat/completions` endpoint using only the Python standard library.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


RISK_LEVELS = [
    "no_observed_risk",
    "mild_risk",
    "moderate_risk",
    "high_risk",
    "insufficient_evidence",
]


SYSTEM_PROMPT = """你是 text-only 风险筛查证据综合器。只能使用用户提供的 PriVTE evidence JSON；不得推断未呈现信息；不得输出医学诊断、心理状态确认或成瘾结论。只输出 JSON。"""


JSON_OUTPUT_INSTRUCTION = {
    "risk_level": RISK_LEVELS,
    "confidence": ["low", "medium", "high"],
    "supporting_evidence": ["list[str]"],
    "uncertainty_or_counter_evidence": ["list[str]"],
    "missing_information": ["list[str]"],
    "needs_human_review": "bool",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path: Path, item: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(item, file, ensure_ascii=False, indent=2)
        file.write("\n")


def build_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    text_evidence = record["text_evidence"]
    user_prompt = (
        text_evidence
        + "\n\n请严格按照以下 JSON schema 输出，不要输出 markdown：\n"
        + json.dumps(JSON_OUTPUT_INSTRUCTION, ensure_ascii=False, indent=2)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def parse_json_response(text: str) -> tuple[dict[str, Any] | None, str | None]:
    stripped = strip_code_fence(text)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            return None, "no_json_object_found"
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            return None, f"json_decode_error:{exc}"
    if not isinstance(parsed, dict):
        return None, "json_response_not_object"
    risk_level = parsed.get("risk_level")
    if risk_level not in RISK_LEVELS:
        return parsed, f"invalid_risk_level:{risk_level}"
    return parsed, None


def select_token_budget_param(model: str, requested: str) -> str:
    if requested != "auto":
        return requested
    normalized_model = model.lower()
    if "gpt-5" in normalized_model:
        return "max_completion_tokens"
    if re.search(r"(^|[/_-])o[134](?:-|$)", normalized_model):
        return "max_completion_tokens"
    return "max_tokens"


def chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float | None,
    max_tokens: int,
    token_budget_param: str,
    timeout_sec: int,
    json_mode: bool,
    user_agent: str,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    payload[token_budget_param] = max_tokens
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": user_agent,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def label_from_record(record: dict[str, Any]) -> str | None:
    label = record.get("target_label", {})
    if not label.get("available"):
        return None
    risk_level = label.get("risk_level")
    return risk_level if risk_level in RISK_LEVELS else None


def prediction_risk_level(item: dict[str, Any]) -> str | None:
    prediction = item.get("prediction")
    if not isinstance(prediction, dict):
        return None
    risk_level = prediction.get("risk_level")
    return risk_level if risk_level in RISK_LEVELS else None


def metric_report(items: list[dict[str, Any]]) -> dict[str, Any]:
    latest_by_sample = {
        item.get("sample_id"): item
        for item in items
        if item.get("sample_id")
    }
    items = list(latest_by_sample.values())
    valid_items = [
        item
        for item in items
        if item.get("target_label") in RISK_LEVELS
        and prediction_risk_level(item) in RISK_LEVELS
    ]
    confusion = {
        gold: {pred: 0 for pred in RISK_LEVELS}
        for gold in RISK_LEVELS
    }
    for item in valid_items:
        risk_level = prediction_risk_level(item)
        if risk_level:
            confusion[item["target_label"]][risk_level] += 1

    total = len(valid_items)
    correct = sum(confusion[label][label] for label in RISK_LEVELS)
    per_label = {}
    f1_values = []
    for label in RISK_LEVELS:
        tp = confusion[label][label]
        fp = sum(confusion[gold][label] for gold in RISK_LEVELS if gold != label)
        fn = sum(confusion[label][pred] for pred in RISK_LEVELS if pred != label)
        support = sum(confusion[label].values())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        if support:
            f1_values.append(f1)
        per_label[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    return {
        "records": len(items),
        "valid_predictions": total,
        "invalid_predictions": len(items) - total,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "macro_f1_observed_labels": (
            round(sum(f1_values) / len(f1_values), 4) if f1_values else 0.0
        ),
        "target_label_counts": dict(Counter(item.get("target_label") for item in items)),
        "predicted_label_counts": dict(
            Counter(prediction_risk_level(item) or "<invalid>" for item in items)
        ),
        "error_counts": dict(
            Counter(item.get("error") for item in items if item.get("error"))
        ),
        "per_label": per_label,
        "confusion": confusion,
    }


def read_existing_predictions(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen = set()
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                item = json.loads(line)
                if not item.get("error") and prediction_risk_level(item) in RISK_LEVELS:
                    seen.add(item.get("sample_id"))
    return seen


def is_retryable_error(error: Exception) -> bool:
    message = str(error)
    return "HTTP 429" in message or "rate limit" in message.lower()


def slugify_tag(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    slug = slug.strip("._-")
    return slug or "untagged"


def infer_evidence_tag(evidence_jsonl: Path) -> str:
    stem = evidence_jsonl.stem
    parent = evidence_jsonl.parent.name
    return parent or stem


def resolve_output_dir(args: argparse.Namespace) -> tuple[Path, str, str]:
    evidence_tag = slugify_tag(args.evidence_tag or infer_evidence_tag(args.evidence_jsonl))
    model_name = args.model or "dry_run"
    model_tag = slugify_tag(args.model_tag or model_name)
    if args.output_dir:
        return args.output_dir, evidence_tag, model_tag
    return args.output_root / evidence_tag / model_tag, evidence_tag, model_tag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-jsonl",
        type=Path,
        default=Path("outputs/quickstart/default/evidence.dataset.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Exact output directory. If omitted, the runner writes to "
            "--output-root/<evidence-tag>/<model-tag>/."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/llm_baselines"),
        help="Root directory used when --output-dir is omitted.",
    )
    parser.add_argument(
        "--evidence-tag",
        default=None,
        help="Evidence pipeline tag used in the automatic output directory.",
    )
    parser.add_argument(
        "--model-tag",
        default=None,
        help="Model/provider tag used in the automatic output directory.",
    )
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL"))
    parser.add_argument(
        "--base-url",
        default=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
    )
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=(
            "Sampling temperature. Omitted by default because some "
            "reasoning/GPT-5-compatible endpoints only accept their default."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument(
        "--token-budget-param",
        choices=("auto", "max_tokens", "max_completion_tokens"),
        default=os.environ.get("LLM_TOKEN_BUDGET_PARAM", "auto"),
        help=(
            "Parameter name used for the completion token budget. "
            "Use max_completion_tokens for GPT-5/reasoning-style models."
        ),
    )
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--sleep-sec", type=float, default=0.0)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retries per sample for retryable endpoint errors such as HTTP 429.",
    )
    parser.add_argument(
        "--retry-sleep-sec",
        type=float,
        default=10.0,
        help="Base sleep between retry attempts; multiplied by attempt number.",
    )
    parser.add_argument(
        "--user-agent",
        default=os.environ.get("LLM_USER_AGENT", "PriVTE-LLM-Baseline/0.1"),
        help="HTTP User-Agent header sent to the OpenAI-compatible endpoint.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-json-mode", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.evidence_jsonl.exists():
        raise SystemExit(f"Evidence JSONL not found: {args.evidence_jsonl}")
    if not args.model and not args.dry_run:
        raise SystemExit("Missing --model or LLM_MODEL.")
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key and not args.dry_run:
        raise SystemExit(f"Missing API key env var: {args.api_key_env}")

    output_dir, evidence_tag, model_tag = resolve_output_dir(args)
    token_budget_param = select_token_budget_param(args.model or "", args.token_budget_param)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    prompts_path = output_dir / "prompts.jsonl"
    metrics_path = output_dir / "metrics.json"
    config_path = output_dir / "run_config.json"
    if args.overwrite:
        for path in (predictions_path, prompts_path, metrics_path):
            if path.exists():
                path.unlink()

    records = read_jsonl(args.evidence_jsonl)
    if args.limit is not None:
        records = records[: args.limit]
    seen = read_existing_predictions(predictions_path)

    write_json(
        config_path,
        {
            "evidence_jsonl": args.evidence_jsonl.as_posix(),
            "output_dir": output_dir.as_posix(),
            "evidence_tag": evidence_tag,
            "model_tag": model_tag,
            "model": args.model,
            "base_url": args.base_url,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "token_budget_param": token_budget_param,
            "requested_token_budget_param": args.token_budget_param,
            "max_retries": args.max_retries,
            "retry_sleep_sec": args.retry_sleep_sec,
            "json_mode": not args.no_json_mode,
            "user_agent": args.user_agent,
            "dry_run": args.dry_run,
            "note": "target_label is used only for evaluation and is never inserted into messages.",
        },
    )

    for record in records:
        sample_id = record["sample_id"]
        messages = build_messages(record)
        append_jsonl(
            prompts_path,
            {
                "sample_id": sample_id,
                "messages": messages,
            },
        )
        if args.dry_run or sample_id in seen:
            continue
        output_item = {
            "sample_id": sample_id,
            "target_label": label_from_record(record),
            "prediction": None,
            "raw_response": None,
            "error": None,
        }
        for attempt in range(args.max_retries + 1):
            try:
                response = chat_completion(
                    base_url=args.base_url,
                    api_key=api_key,
                    model=args.model,
                    messages=messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    token_budget_param=token_budget_param,
                    timeout_sec=args.timeout_sec,
                    json_mode=not args.no_json_mode,
                    user_agent=args.user_agent,
                )
                content = response["choices"][0]["message"]["content"]
                prediction, parse_error = parse_json_response(content)
                output_item["prediction"] = prediction
                output_item["raw_response"] = content
                output_item["error"] = parse_error
                output_item["usage"] = response.get("usage")
                output_item["attempts"] = attempt + 1
                break
            except Exception as exc:
                output_item["error"] = str(exc)
                output_item["attempts"] = attempt + 1
                if attempt >= args.max_retries or not is_retryable_error(exc):
                    break
                time.sleep(args.retry_sleep_sec * (attempt + 1))
        append_jsonl(predictions_path, output_item)
        if args.sleep_sec:
            time.sleep(args.sleep_sec)

    if args.dry_run:
        print(f"wrote_prompts={prompts_path}")
        print("dry_run=true")
        return 0

    predictions = read_jsonl(predictions_path)
    report = metric_report(predictions)
    write_json(metrics_path, report)
    print(f"predictions={predictions_path}")
    print(f"metrics={metrics_path}")
    if report["valid_predictions"] == 0 and report["records"]:
        print("warning=all_predictions_invalid")
        if report["error_counts"]:
            print(f"errors={report['error_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
