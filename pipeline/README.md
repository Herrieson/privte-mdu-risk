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

`privte_preprocessor_v0` in
`privte_pipeline/algorithms/privte_preprocessor_v0.py`

- establishes the active `session_metadata + global_features + event_windows`
  evidence schema;
- runs local frame sampling from coverage windows;
- computes lightweight device/screen visibility, face observability,
  MediaPipe Tasks hand-landmark visibility, FaceMesh visibility, sparse-frame
  eye-openness proxies, selected facial geometry action proxies, frame-difference
  motion, interaction-proxy, posture-proxy, and quality fields;
- emits event windows with relative time periods, compact proxy-evidence fields,
  quality metrics, privacy flags, and deterministic text rendering;
- remains the integration point for stronger gaze, blink, AU validation,
  baseline-window, and class-level listener modules.

Old FlowLite, Behavior, Trace, and simple-video-quality modules were archived
under `archive/legacy_2026_06_10/methods/`. They are not active extractors.

The next active extractor should be rebuilt around the fixed schema:

```text
global_features
event_windows
quality_summary
limitations
privacy_processing_summary
```

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
manifest_only           # current schema/data-flow baseline
privte_preprocessor_v0  # active frame-proxy preprocessor MVP
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
