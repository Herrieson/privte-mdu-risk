# LLM Baseline

This directory contains the direct text-only LLM baseline for MDU-RiskBench.

The runner reads PriVTE evidence JSONL, sends only `text_evidence` to an
OpenAI-compatible `/chat/completions` endpoint, parses the required JSON output,
and evaluates predictions against `target_label`. Labels are used only after
model inference and are never inserted into the prompt messages.

## Run

```bash
export LLM_BASE_URL=https://api.openai.com/v1
export LLM_API_KEY=...
export LLM_MODEL=gpt-model-name

uv run python experiments/llm/run_llm_baseline.py \
  --evidence-jsonl outputs/quickstart/<run>/evidence.<subset>.jsonl \
  --overwrite
```

By default this writes to:

```text
outputs/llm_baselines/<evidence_tag>/<model_tag>/
```

`model_tag` is inferred from `LLM_MODEL`. To control the directory name, pass
`--model-tag`, for example:

```bash
uv run python experiments/llm/run_llm_baseline.py \
  --evidence-jsonl outputs/quickstart/<run>/evidence.<subset>.jsonl \
  --model-tag gpt_main \
  --overwrite
```

For GPT-5/reasoning-style OpenAI models, the runner automatically uses
`max_completion_tokens` instead of `max_tokens`. If an endpoint still rejects
the token-budget parameter, set it explicitly:

```bash
uv run python experiments/llm/run_llm_baseline.py \
  --evidence-jsonl outputs/quickstart/<run>/evidence.<subset>.jsonl \
  --model-tag gpt_main \
  --token-budget-param max_completion_tokens \
  --overwrite
```

The runner omits `temperature` by default because some GPT-5-compatible
endpoints only accept the model default. For endpoints that support deterministic
sampling, pass `--temperature 0`.

## Run Claude Through an OpenAI-compatible Gateway

This runner calls `/chat/completions`, so Claude needs an OpenAI-compatible
gateway such as LiteLLM, OpenRouter, or another proxy.

```bash
export LLM_BASE_URL=http://localhost:4000/v1
export LLM_API_KEY=...
export LLM_MODEL=claude-model-name

uv run python experiments/llm/run_llm_baseline.py \
  --evidence-jsonl outputs/quickstart/<run>/evidence.<subset>.jsonl \
  --model-tag claude_main \
  --overwrite \
  --no-json-mode
```

For a local OpenAI-compatible server, set `LLM_BASE_URL`, for example:

```bash
export LLM_BASE_URL=http://localhost:8000/v1
export LLM_API_KEY=dummy
export LLM_MODEL=Qwen2.5-7B-Instruct
```

## Dry Run

Use this to inspect the exact prompt messages without calling a model:

```bash
uv run python experiments/llm/run_llm_baseline.py \
  --evidence-jsonl outputs/quickstart/<run>/evidence.<subset>.jsonl \
  --output-dir /tmp/privte_llm_dry_run \
  --dry-run \
  --limit 2
```

Outputs:

```text
prompts.jsonl       # sample_id + messages sent to the LLM
predictions.jsonl   # sample_id + parsed prediction + raw response + target label
metrics.json        # accuracy, macro-F1, per-label metrics, confusion matrix
run_config.json     # endpoint/model/run settings
```
