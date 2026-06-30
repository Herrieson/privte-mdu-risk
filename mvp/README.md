# PriVTE MVP

This directory is reserved for the next minimal runnable demo on top of the
stable `../pipeline/` package.

The previous FlowLite, Behavior, Trace, and simple-video-quality MVP runners
have been archived under:

```text
archive/legacy_2026_06_10/methods/mvp/
```

The new MVP should be rebuilt after the evidence schema is frozen. It should use
the active pipeline boundary:

```text
pipeline/privte_pipeline/core/          # stable extractor interface
pipeline/privte_pipeline/algorithms/    # replaceable algorithms
configs/algorithms/                     # algorithm configs
outputs/quickstart/                     # generated quickstart outputs
```

Expected next MVP shape:

```text
raw video
  -> local proxy extraction
  -> global_features
  -> event_windows from multiple listeners
  -> privacy filtering
  -> LLM-ready evidence JSONL and text
```

## PriVTE Preprocessor v0

`privte_preprocessor_v0` is the current runnable MVP for the new method line. It
performs local frame sampling, coarse relative-time window construction,
lightweight screen/device-region heuristics, face-observability proxies,
MediaPipe Tasks hand-landmark visibility, FaceMesh visibility, sparse-frame eye
openness proxies, selected facial geometry action proxies, frame-difference
motion proxies, event-window selection, schema validation, privacy flags, LLM
package generation, and deterministic text rendering.

This version still treats hand interaction, gaze, blink rate, and facial action
units as proxy evidence. It does not emit raw frames, crops, coordinates, or
landmark sequences, and it does not claim exact eye tracking, exact blink
counting, validated AU classification, emotion recognition, or diagnosis.

Run:

```bash
uv run python mvp/run_preprocessor_v0_mvp.py --overwrite
```

Outputs:

```text
outputs/quickstart/6_1data_privte_preprocessor_v0/preprocessor_v0_evidence.6_1data.jsonl
outputs/quickstart/6_1data_privte_preprocessor_v0/preprocessor_v0_report.6_1data.json
outputs/quickstart/6_1data_privte_preprocessor_v0/evidence_text/*.txt
```

## PriVTE Preprocessor v1

`privte_preprocessor_v1` keeps the v0 public evidence schema and reuses the same
local visual proxies, but calibrates the demo outputs for easier tuning:

- event windows must pass trigger-specific evidence-strength thresholds;
- device-region motion without visible hands is weak proxy evidence, not
  confirmed interaction;
- repetitive-operation and motion-confounding bins are aggregated more
  conservatively;
- global features include negative-evidence summaries for text-only screening.

Run:

```bash
uv run python mvp/run_preprocessor_v1_mvp.py --overwrite
```

## PriVTE Preprocessor v2

`privte_preprocessor_v2` keeps the V1 calibrated proxy extractor and adds a
stateful evidence layer. It emits privacy-filtered symbolic per-window states,
behavior episodes, temporal trajectory summaries, and an evidence graph with
positive evidence, weak proxy evidence, counter-evidence, and quality
limitations.

Run:

```bash
uv run python mvp/run_preprocessor_v2_mvp.py --overwrite
```
