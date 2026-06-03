# PriVTE Evidence Pipeline

This directory is the stable engineering base for PriVTE evidence generation.
It is not the MVP experiment itself.

The pipeline turns an internal labeled person manifest into person-level
evidence records and templated text:

```text
internal labeled person manifest
  -> replaceable algorithm module
  -> person-level evidence JSONL
  -> LLM-ready evidence package
  -> rendered text evidence
  -> later text-only model / LLM benchmark
```

The main design goal is algorithm replacement:

```text
manifest reader / writer / text renderer / report
  stay stable

evidence extractor
  can be replaced independently
```

## Layout

```text
pipeline/
  build_evidence.py              # generic CLI entrypoint
  privte_pipeline/
    core/                        # stable extractor interface
    algorithms/                  # replaceable algorithm modules + registry
    io.py                        # JSON/JSONL/text output helpers
    run.py                       # shared read/build/write orchestration
    extractors.py                # compatibility exports, not new algorithm code
    evidence.py                  # final record assembly and report
    llm_package.py               # label-free LLM evidence package builder
    renderers.py                 # text evidence rendering
```

## Run

Schema/data-flow baseline:

```bash
uv run python pipeline/build_evidence.py \
  --person-manifest data/manifests/internal_person_manifest.6_1data.v0.labeled.jsonl \
  --output-dir outputs/pipeline/6_1data \
  --subset-name 6.1data \
  --extractor manifest_only
```

Simple video-quality extractor:

```bash
uv run python pipeline/build_evidence.py \
  --person-manifest data/manifests/internal_person_manifest.6_1data.v0.labeled.jsonl \
  --output-dir outputs/pipeline/6_1data_simple_video_quality \
  --subset-name 6.1data \
  --extractor simple_video_quality
```

PriVTE-FlowLite extractor:

```bash
uv run python pipeline/build_evidence.py \
  --person-manifest data/manifests/internal_person_manifest.6_1data.v0.labeled.jsonl \
  --output-dir outputs/pipeline/6_1data_privte_flowlite \
  --subset-name 6.1data \
  --extractor privte_flowlite \
  --extractor-config configs/algorithms/privte_flowlite.v0.json
```

PriVTE-Behavior v1 extractor:

```bash
uv run python pipeline/build_evidence.py \
  --person-manifest data/manifests/internal_person_manifest.6_1data.v0.labeled.jsonl \
  --output-dir outputs/pipeline/6_1data_privte_behavior_v1 \
  --subset-name 6.1data \
  --extractor privte_behavior_v1 \
  --extractor-config configs/algorithms/privte_behavior.v1.json
```

## Outputs

```text
outputs/pipeline/<run>/evidence.<subset>.jsonl
outputs/pipeline/<run>/pipeline_report.<subset>.json
outputs/pipeline/<run>/evidence_text/*.txt
```

## Current Extractors

`manifest_only` in `privte_pipeline/algorithms/manifest_only.py`

- validates schema and data flow;
- reports modality availability and missing evidence;
- does not compute visual proxy features.

`simple_video_quality` in `privte_pipeline/algorithms/simple_video_quality.py`

- computes video availability and file-size distributions;
- optionally reads video container metadata through OpenCV when available;
- falls back to a pure Python MP4 box parser when OpenCV is unavailable;
- reports coarse duration, resolution, and FPS distributions;
- does not output frames, images, audio, OCR, ASR, questionnaire answers, app
  names, or exact heart-rate values.

`privte_flowlite` in `privte_pipeline/algorithms/flowlite.py`

- samples video frames locally with OpenCV;
- selects coverage/event/quality evidence through frame-level aggregation;
- uses face ROI, screen-like bright-rectangle ROI, and frame-difference motion;
- normalizes motion bursts against within-person median motion;
- emits only coarse ratios, bins, event types, behavior proxy observations, and
  quality summaries;
- builds LLM-ready evidence for device-use observability, sustained screen
  engagement proxy, active device interaction proxy, repetitive operation proxy,
  posture/context change proxy, and explicitly missing gaze/affect/fatigue
  evidence;
- does not output frames, crops, coordinates, OCR/ASR, face embeddings,
  high-dimensional landmarks, app names, questionnaire answers, or exact
  heart-rate values.

`privte_behavior_v1` in `privte_pipeline/algorithms/behavior_v1.py`

- uses OpenCV for local frame decoding and motion proxies;
- uses MediaPipe Hands, Face Mesh, and Pose when dependencies are installed;
- uses Ultralytics YOLO for device-like object detection when available, with a
  screen-like heuristic fallback;
- extracts device observability, short-interval device-region activity,
  hand-device interaction proxy, repetitive operation proxy, face-device context
  observability, stable engagement proxy, and posture/context change proxy;
- emits only coarse ratios, bins, event types, quality summaries, and
  LLM-ready behavior observations;
- requires the configured MediaPipe `.task` model files for real behavior
  output; metadata fallback is only for plumbing checks and should produce
  insufficient-evidence text;
- does not output frames, crops, coordinates, masks, OCR/ASR, face embeddings,
  high-dimensional landmarks, app names, questionnaire answers, or exact
  heart-rate values.

## Evidence Contract

Every extractor should return the same top-level contract:

```text
extractor
modality_availability
feature_blocks
privacy_processing_summary
missing_information
limitations
```

The assembled record also contains:

```text
target_label             # evaluation only, not rendered into model input
llm_evidence_package     # label-free model input contract
text_evidence            # human-readable rendering of llm_evidence_package
```

This lets later work replace only the extractor with a fuller PriVTE algorithm:

```text
simple_video_quality    # current runnable algorithm baseline
privte_flowlite         # OpenCV frame-level PriVTE protocol MVP
privte_behavior_v1      # MediaPipe + YOLO practical behavior proxy extractor
video_proxy_v0          # stronger face/hand/device visibility + interaction proxies
privte_v1               # key-window + ROI + proxy-feature extractor
```

## Adding Algorithms

Future algorithm work should not edit `pipeline/build_evidence.py`,
`evidence.py`, `renderers.py`, or the quickstart scripts.

Add or change:

```text
pipeline/privte_pipeline/algorithms/<algorithm_name>.py
pipeline/privte_pipeline/algorithms/registry.py
configs/algorithms/<algorithm_name>.<version>.json
```

The algorithm returns the extractor evidence contract. The shared pipeline then
separates labels, builds the LLM evidence package, renders text, and writes
outputs.
