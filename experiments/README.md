# Experiments

This directory keeps experiment runners separate from the PriVTE evidence
generation pipeline.

Current mainline:

```text
PriVTE evidence JSONL -> direct text-only LLM call -> JSON prediction -> label-only evaluation
```

Use:

```bash
export LLM_MODEL=gpt-or-claude-model-name

uv run python experiments/llm/run_llm_baseline.py \
  --evidence-jsonl outputs/quickstart/<run>/evidence.<subset>.jsonl
```

By default, outputs are grouped as:

```text
outputs/llm_baselines/<evidence_tag>/<model_tag>/
```

The prompt receives only `text_evidence`. `target_label` is used only after
inference for metrics.

Historical evidence-label audit scripts and outputs were archived under
`archive/legacy_2026_06_10/`.
