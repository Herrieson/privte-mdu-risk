# PriVTE Quickstart MVPs

This directory contains convenience scripts for quick local runs on top of the
stable `../pipeline/` package. It should not contain core algorithm
implementations.

Current source-of-truth locations:

```text
pipeline/privte_pipeline/core/          # stable extractor interface
pipeline/privte_pipeline/algorithms/    # replaceable algorithms
configs/algorithms/                     # algorithm configs
outputs/quickstart/                     # default outputs from these scripts
```

## Simple Video Quality

The first MVP uses the `simple_video_quality` extractor. It is deliberately
simple but not a placeholder: it reads video file availability and coarse video
container metadata to produce privacy-filtered text evidence. It uses OpenCV if
available and otherwise falls back to a pure Python MP4 metadata parser. It does
not output frames, images, audio, OCR/ASR, app names, questionnaire answers, or
exact heart-rate values.

## Run

```bash
uv run python mvp/run_simple_mvp.py
```

Equivalent explicit command:

```bash
uv run python mvp/run_simple_mvp.py \
  --person-manifest data/manifests/internal_person_manifest.6_1data.v0.labeled.jsonl \
  --output-dir outputs/quickstart/6_1data_simple_video_quality \
  --subset-name 6.1data \
  --max-metadata-clips 16
```

Use `--metadata-all` if you want to probe every video file instead of a
deterministic sample per participant.

## Outputs

```text
outputs/quickstart/6_1data_simple_video_quality/simple_mvp_evidence.6_1data.jsonl
outputs/quickstart/6_1data_simple_video_quality/simple_mvp_report.6_1data.json
outputs/quickstart/6_1data_simple_video_quality/evidence_text/*.txt
```

## PriVTE-FlowLite v0

`PriVTE-FlowLite v0` is the first frame-level MVP. It follows the advisor brief
flow more closely:

```text
raw video
  -> local frame sampling
  -> key-window aggregation
  -> ROI focusing
  -> proxy feature extraction
  -> within-person reference normalization
  -> quality estimation
  -> LLM-ready privacy-filtered behavior evidence
```

Run:

```bash
uv run python mvp/run_flowlite_mvp.py
```

Useful options:

```bash
uv run python mvp/run_flowlite_mvp.py \
  --max-video-clips 12 \
  --frames-per-clip 8
```

Use `--process-all-clips` for a full internal run. The default config requires
OpenCV, so the command fails instead of silently falling back when frame-level
analysis dependencies are missing. Use `--allow-metadata-fallback` only for
dependency-free pipeline checks.

The default config is:

```text
configs/algorithms/privte_flowlite.v0.json
```

Outputs:

```text
outputs/quickstart/6_1data_privte_flowlite/flowlite_mvp_evidence.6_1data.jsonl
outputs/quickstart/6_1data_privte_flowlite/flowlite_mvp_report.6_1data.json
outputs/quickstart/6_1data_privte_flowlite/evidence_text/*.txt
```

Older generated files may still exist under `mvp/outputs/`, but new runs should
write to `outputs/quickstart/`.

The JSONL keeps two layers:

- `evidence`: auditable extractor output, including coarse ratios, bins, event
  windows, quality summaries, and privacy flags.
- `llm_evidence_package`: label-free model input with behavior proxy
  observations, risk-screening rubric, supporting evidence, uncertainty,
  missing information, privacy constraints, and the required JSON model output
  schema.

The rendered `evidence_text/*.txt` files are built from
`llm_evidence_package`, not from raw videos, frames, OCR/ASR, questionnaire
answers, exact heart-rate values, app names, or labels.

## What This MVP Proves

- The project now has a stable pipeline/algorithm boundary.
- FlowLite performs local frame-level proxy feature extraction over video files.
- The LLM input is a behavior-oriented evidence package, not a video-quality
  report.
- The text-only input can be generated without exposing raw videos or frames.
- Later work only needs to replace the extractor with a stronger PriVTE visual
  proxy algorithm.

## PriVTE-Behavior v1

`PriVTE-Behavior v1` is the practical stronger algorithm profile. It uses
off-the-shelf local CV models when installed:

```text
raw video
  -> local frame sampling
  -> YOLO device detection with heuristic fallback
  -> MediaPipe hand / face / pose analysis
  -> short-interval local motion around sampled frames
  -> hand-device interaction proxy
  -> device-region activity proxy
  -> stable screen engagement proxy
  -> posture/context change proxy
  -> privacy-filtered LLM evidence package
```

Run:

```bash
uv run python mvp/run_behavior_v1_mvp.py
```

Useful options:

```bash
uv run python mvp/run_behavior_v1_mvp.py \
  --max-video-clips 16 \
  --frames-per-clip 12
```

Use `--disable-yolo` to use MediaPipe plus heuristic device/screen detection.
Real Behavior v1 output requires the configured MediaPipe `.task` model files
under `models/mediapipe/`. Use `--allow-metadata-fallback` only for
dependency-free plumbing checks; that mode intentionally emits
insufficient-evidence text instead of behavior proxy features.

The default config is:

```text
configs/algorithms/privte_behavior.v1.json
```

Outputs:

```text
outputs/quickstart/6_1data_privte_behavior_v1/behavior_v1_mvp_evidence.6_1data.jsonl
outputs/quickstart/6_1data_privte_behavior_v1/behavior_v1_mvp_report.6_1data.json
outputs/quickstart/6_1data_privte_behavior_v1/evidence_text/*.txt
```

This version still does not output frames, crops, coordinates, masks, OCR/ASR,
face embeddings, landmark sequences, questionnaire answers, exact heart-rate
values, app names, raw paths, or exact timestamps.
