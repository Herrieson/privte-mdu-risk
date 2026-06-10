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
