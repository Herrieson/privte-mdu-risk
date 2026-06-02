# PriVTE Evidence Pipeline

This directory is the stable engineering base for PriVTE evidence generation.
It is not the MVP experiment itself.

The pipeline turns an internal labeled person manifest into person-level
evidence records and templated text:

```text
internal labeled person manifest
  -> replaceable evidence extractor
  -> person-level evidence JSONL
  -> templated text evidence
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
    io.py                        # JSON/JSONL/text output helpers
    extractors.py                # replaceable evidence extractors
    evidence.py                  # final record assembly and report
    renderers.py                 # text evidence rendering
```

## Run

Schema/data-flow baseline:

```bash
uv run python pipeline/build_evidence.py \
  --person-manifest data/manifests/internal_person_manifest.6_1data.v0.labeled.jsonl \
  --output-dir pipeline/outputs/6_1data \
  --subset-name 6.1data \
  --extractor manifest_only
```

Simple video-quality extractor:

```bash
uv run python pipeline/build_evidence.py \
  --person-manifest data/manifests/internal_person_manifest.6_1data.v0.labeled.jsonl \
  --output-dir pipeline/outputs/6_1data_simple_video_quality \
  --subset-name 6.1data \
  --extractor simple_video_quality
```

## Outputs

```text
pipeline/outputs/<run>/evidence.<subset>.jsonl
pipeline/outputs/<run>/pipeline_report.<subset>.json
pipeline/outputs/<run>/evidence_text/*.txt
```

## Current Extractors

`manifest_only`

- validates schema and data flow;
- reports modality availability and missing evidence;
- does not compute visual proxy features.

`simple_video_quality`

- computes video availability and file-size distributions;
- optionally reads video container metadata through OpenCV when available;
- falls back to a pure Python MP4 box parser when OpenCV is unavailable;
- reports coarse duration, resolution, and FPS distributions;
- does not output frames, images, audio, OCR, ASR, questionnaire answers, app
  names, or exact heart-rate values.

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

This lets later work replace only the extractor with a fuller PriVTE algorithm:

```text
simple_video_quality    # current runnable algorithm baseline
video_proxy_v0          # face/hand/device visibility + simple interaction proxies
privte_v1               # key-window + ROI + proxy-feature extractor
```
