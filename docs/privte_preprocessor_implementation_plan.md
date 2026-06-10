# PriVTE Preprocessor Implementation Plan

## 1. Active Direction

This document is the active engineering plan for the PriVTE local video
preprocessor.

The new direction is:

```text
local / controlled-environment high-density visual analysis
  -> privacy fuse / granularity compression
  -> low-dimensional structured evidence facts
  -> text-only LLM risk screening
```

The active evidence structure is:

```text
session_metadata
global_features
event_windows
quality_summary
limitations
privacy_processing_summary
```

The project should no longer continue the archived FlowLite / Behavior / Trace
line. Those versions are preserved under `archive/legacy_2026_06_10/` only for
reference.

## 2. Design Goals

### Goals

- Process raw videos in a controlled local preprocessing environment.
- Avoid sending raw video, frames, audio, OCR, ASR, raw coordinates, face
  embeddings, app names, questionnaire answers, or exact heart-rate values to
  the center side.
- Convert long videos into auditable, low-dimensional, schema-first evidence.
- Separate global session statistics from selected event windows.
- Make each event detector independently replaceable and tunable.
- Keep the center-side LLM as an evidence synthesizer, not a diagnostic model.

### Non-goals

- Do not build a video-based addiction diagnosis model.
- Do not release raw video or image evidence.
- Do not depend on questionnaire, heart-rate, or app logs as model input.
- Do not use free-form video captioning as the primary evidence pipeline.
- Do not expose exact timestamps or high-frequency landmark trajectories in the
  public-lite evidence format.
- Do not make edge deployment or memory-only processing a required claim for the
  current benchmark paper.

## 3. Future Deployment Targets

The current paper line is the privacy-filtered benchmark:

```text
raw sensitive videos
  -> controlled local preprocessing
  -> public-lite / controlled textual evidence
  -> text-only benchmark
```

Edge deployment is a future application target, not the current main
contribution. If time and engineering capacity allow, later versions can pursue:

1. **Edge-side execution**: move the preprocessor from a controlled local
   workstation/server to participant-side or institution-side devices.
2. **Memory-only processing**: process video streams without writing raw video
   to disk, then retain only privacy-filtered evidence.
3. **Zero-retention raw media mode**: delete or never persist raw video after
   local evidence extraction.

These targets can strengthen the application story, but the current benchmark
should not depend on achieving them.

## 4. End-to-End Architecture

```text
Raw local video
  -> Spatio-Temporal Trimming
  -> ROI Focusing
  -> Proxy Feature Extraction
  -> Quality Estimation Gateway
  -> Privacy Filtering & Granularity Compression
  -> Schema-first Evidence Assembly
  -> Text-only LLM screening
```

## 5. Pipeline Layers

### 5.1 Spatio-Temporal Trimming

Purpose: reduce a long raw video into a small set of useful analysis windows.

Responsibilities:

- Build coarse coverage windows across the whole session.
- Detect candidate event windows using lightweight signals such as frame
  difference, local motion bursts, posture shifts, and quality changes.
- Select quality windows where face, hands, and device regions are visible.
- Build baseline windows for within-session normalization.

Expected output:

```text
coverage_windows
candidate_event_windows
quality_windows
baseline_windows
```

Implementation notes:

- Start with deterministic sampling plus frame-difference motion scores.
- Keep exact frame indices and timestamps internal only.
- Public evidence should expose only relative order, coarse position, and
  duration bins.

### 5.2 ROI Focusing

Purpose: limit computation to behavior-relevant regions and avoid unnecessary
background processing.

Candidate ROIs:

- face region;
- eye/head region;
- hand region;
- body/posture region;
- device or screen-like region.

Privacy rule:

- ROI crops may be used in memory for local computation.
- ROI images, crops, masks, exact coordinates, and high-dimensional landmark
  sequences must not be written to public evidence.

### 5.3 Proxy Feature Extraction

Purpose: compute objective observable proxy facts, not risk labels.

Initial feature groups:

- screen-oriented gaze / head proxy;
- hand-device interaction proxy;
- repetitive operation proxy;
- posture state and posture trend;
- blink-rate level and blink-rate change;
- facial action unit trend;
- device visibility and screen-like region visibility;
- motion confounding indicators.

The output of this layer is still internal. It should be compressed before
leaving the local pipeline.

### 5.4 Quality Estimation Gateway

Purpose: prevent low-quality visual predictions from becoming overconfident LLM
evidence.

Quality metrics:

- face visibility level;
- hand visibility level;
- device visibility level;
- lighting quality;
- occlusion level;
- gaze estimation quality;
- multi-person interference;
- motion confounding level;
- event confidence.

Quality should be attached to both:

- global session-level evidence;
- each event window.

### 5.5 Privacy Filtering And Granularity Compression

Purpose: enforce a whitelist output policy before any evidence leaves the local
device.

Allowed public-lite evidence:

- anonymous sample/session IDs;
- relative order and coarse time position;
- duration bins;
- categorical or binned global features;
- categorical event-window proxy evidence;
- quality levels;
- limitations;
- privacy processing flags.

Blocked public-lite evidence:

- raw video;
- frames, crops, masks, or keyframes;
- audio;
- OCR or ASR text;
- exact timestamps;
- exact raw paths;
- exact heart-rate values;
- app names;
- questionnaire answers;
- face embeddings;
- high-frequency coordinate sequences;
- high-dimensional face mesh, pose, or hand landmark sequences;
- free-form appearance, identity, home, school, class, or background
  descriptions.

### 5.6 Schema-first Evidence Assembly

Purpose: assemble the compressed facts into a stable machine-readable JSON
object and a deterministic text rendering.

The JSON is the source of truth. Text evidence should be a deterministic
rendering of the JSON, not an independent free-form caption.

## 6. Evidence Schema v0

The first active implementation should output this structure.

```json
{
  "schema_version": "privte_preprocessor_evidence.v0",
  "session_metadata": {
    "session_id": "ANON_8A9B2C",
    "duration_bin": "25-30min",
    "valid_observation_duration_bin": "25-30min",
    "analyzed_window_count": 12,
    "event_window_count": 2
  },
  "global_features": {
    "screen_gaze_ratio_bin": "high",
    "max_continuous_gaze_duration_bin": "long",
    "blink_rate_level": "typical",
    "blink_rate_trend": "decreased_during_high_engagement_windows",
    "overall_posture_trend": "progressive_forward_lean",
    "interaction_intensity": "high",
    "repetitive_operation_level": "medium",
    "motion_confounding_level": "low"
  },
  "event_windows": [
    {
      "window_id": "event_01",
      "window_order": 1,
      "relative_position": "early",
      "duration_bin": "medium",
      "trigger_type": "high_interaction_intensity",
      "trigger_source": "interaction_burst_listener",
      "proxy_evidence": {
        "interaction_intensity": "high",
        "interaction_pattern": "rapid_tap_or_swipe_proxy",
        "posture_state": "static_close_to_screen",
        "gaze_state": "sustained_screen_oriented_proxy",
        "blink_rate_change": "decreased",
        "facial_au_codes": [
          {
            "au_code": "AU4",
            "label": "brow_lowerer",
            "intensity": "low"
          }
        ]
      },
      "quality_metrics": {
        "face_visibility_level": "very_high",
        "hand_visibility_level": "high",
        "device_visibility_level": "high",
        "lighting_quality": "sufficient",
        "occlusion_level": "none",
        "event_confidence": "high"
      }
    }
  ],
  "quality_summary": {
    "overall_data_sufficiency": "adequate",
    "face_observability": "high",
    "hand_observability": "medium",
    "device_observability": "high",
    "gaze_estimation_quality": "medium",
    "multi_person_interference": "none_detected",
    "motion_confounding_level": "low"
  },
  "limitations": [
    "hand visibility was low in some late windows",
    "gaze evidence is proxy-based and should not be interpreted as exact eye tracking",
    "facial action units are behavioral proxies and not emotion diagnosis",
    "questionnaire, exact heart-rate values, app names, OCR, ASR, raw video, and images are not included"
  ],
  "privacy_processing_summary": {
    "raw_video_included": false,
    "raw_images_included": false,
    "raw_audio_included": false,
    "ocr_text_included": false,
    "asr_text_included": false,
    "exact_timestamps_included": false,
    "high_frequency_coordinates_included": false,
    "face_embeddings_included": false,
    "screen_content_included": false,
    "questionnaire_answers_included": false,
    "exact_heart_rate_values_included": false,
    "app_names_included": false,
    "processing_steps": [
      "raw video processed in a controlled local preprocessing environment",
      "event windows represented by relative order and coarse position only",
      "visual measurements compressed into bins and categorical proxy evidence",
      "background, appearance, OCR, ASR, and app content suppressed"
    ]
  }
}
```

## 7. Event Listener Architecture

Each listener should be independent. A listener consumes local frame/window
features and emits zero or more event windows.

```text
listener.detect(window_features, global_context) -> list[EventWindow]
```

Initial listeners:

| Listener | Purpose | Output trigger_type | Status |
| --- | --- | --- | --- |
| `InteractionBurstListener` | Detect high hand/device activity or rapid interaction proxy | `high_interaction_intensity` | Basic v0 proxy implemented |
| `SustainedGazeListener` | Detect long screen-oriented gaze/head proxy windows | `sustained_screen_oriented_proxy` | Basic v0 proxy implemented; not exact eye tracking |
| `PostureShiftListener` | Detect forward lean, backward shift, head drop, or posture transitions | `posture_shift` | Partial close-to-screen proxy implemented |
| `BlinkChangeListener` | Detect blink-rate drop or spike relative to baseline | `blink_rate_change` | Not implemented |
| `FacialAUTrendListener` | Detect selected AU trends such as AU4 / AU15 | `facial_au_trend` | Not implemented |
| `QualityDropListener` | Detect windows where quality limits evidence reliability | `quality_drop` | Basic v0 quality proxy implemented |
| `MotionConfounderListener` | Detect camera/global motion that may confound interaction evidence | `motion_confounding` | Basic v0 proxy implemented |

Event selection should keep a small number of high-value windows and preserve
relative order.

## 8. Current Repository Status

| Component | Status | Notes |
| --- | --- | --- |
| Advisor-facing project brief | Active | `docs/advisor_brief.md` |
| Historical docs | Archived | `archive/legacy_2026_06_10/docs/` |
| Historical FlowLite / Behavior / Trace methods | Archived | `archive/legacy_2026_06_10/methods/` |
| Historical generated results | Archived | `archive/legacy_2026_06_10/results/` |
| Data manifest scripts | Implemented | `scripts/build_dataset_manifest.py`, `scripts/build_combined_manifest.py`, `scripts/join_labels_to_manifest.py` |
| Generic pipeline runner | Implemented | `pipeline/build_evidence.py` |
| Extractor interface | Implemented | `pipeline/privte_pipeline/core/base.py` |
| Active extractor registry | Implemented | exposes `manifest_only` and `privte_preprocessor_v0` |
| Generic LLM evidence package builder | Implemented | supports schema-first `global_features + event_windows` style fields |
| Generic text renderer | Implemented | deterministic rendering from JSON evidence |
| LLM baseline runner | Implemented | OpenAI-compatible text-only runner |
| New video preprocessor extractor | Implemented v0 | emits runnable frame-proxy schema |
| Key-window selector | Partial | clip-order coverage windows plus event-window selection implemented |
| ROI focusing module | Partial | screen/device-like region and face-observability proxies implemented |
| Proxy feature extractors | Partial | frame difference, screen/device heuristic, face visibility, posture proxy implemented |
| Quality gateway | Partial | lighting, blur, face/device observability, motion confounding, event confidence implemented |
| Privacy compressor | Partial | schema whitelist and privacy flags implemented; raw media/path suppression audited |
| Event listeners | Partial | basic listener behavior implemented inside v0 extractor; class-level listener interface pending |
| Schema validator | Implemented | `validate_preprocessor_evidence` |
| Privacy audit script | Implemented | `scripts/audit_privacy_evidence.py` |

## 9. Development Todo List

### Phase 0: Freeze Schema And Enums

- [x] Create a schema constants module for allowed values.
- [x] Freeze top-level schema fields.
- [ ] Freeze event window enum values.
- [x] Freeze global feature names.
- [x] Freeze quality metric names.
- [x] Freeze privacy flags.
- [ ] Decide internal-only vs public-lite fields.

Deliverable:

```text
pipeline/privte_pipeline/schemas/preprocessor_v0.py
```

### Phase 1: Add Extractor Skeleton

- [x] Add `pipeline/privte_pipeline/algorithms/privte_preprocessor_v0.py`.
- [x] Register extractor name `privte_preprocessor_v0`.
- [x] Add config file `configs/algorithms/privte_preprocessor.v0.json`.
- [x] Add MVP runner `mvp/run_preprocessor_v0_mvp.py`.
- [x] Make the first version output valid `global_features` and `event_windows`
  with real session metadata and privacy flags.

Deliverable:

```text
uv run python mvp/run_preprocessor_v0_mvp.py
```

### Phase 2: Implement Video Sampling And Key Windows

- [x] Read videos locally with OpenCV.
- [x] Deterministically sample coverage windows by clip order.
- [x] Probe MP4 container metadata without exporting frames.
- [x] Compute lightweight frame-difference motion scores.
- [x] Select candidate event windows.
- [ ] Select dedicated quality windows.
- [ ] Build baseline windows.
- [x] Keep exact frame indices internal only.

Tunable parameters:

- max videos per person;
- frames per window;
- coverage window count;
- event candidate count;
- motion threshold;
- min quality threshold.

### Phase 3: Implement Quality Gateway

- [x] Estimate readable frame ratio.
- [x] Estimate face visibility level.
- [ ] Estimate hand visibility level.
- [x] Estimate device visibility level.
- [x] Estimate lighting quality.
- [x] Estimate occlusion level from face visibility proxy.
- [x] Estimate multi-person interference from face-count proxy.
- [x] Estimate motion confounding level.
- [x] Attach event-level confidence.

Acceptance criterion:

- Low-quality windows must not produce high-confidence evidence.

### Phase 4: Implement Proxy Feature Extractors

- [x] Device/screen-like region detector.
- [ ] Hand detector or hand landmark availability estimator.
- [x] Head/gaze orientation proxy.
- [x] Posture proxy.
- [ ] Blink proxy.
- [ ] Selected facial AU proxy.
- [x] Local motion and interaction proxy.
- [x] Repetition proxy.

Notes:

- Features should remain internal until compression.
- Public evidence should use bins, categories, and relative order.
- Exact coordinates and high-frequency sequences must not be emitted.

### Phase 5: Implement Event Listeners

- [ ] Define listener base class.
- [x] Implement basic `InteractionBurstListener` behavior.
- [x] Implement basic `SustainedGazeListener` behavior.
- [x] Implement basic `PostureShiftListener` behavior.
- [ ] Implement `BlinkChangeListener`.
- [ ] Implement `FacialAUTrendListener`.
- [x] Implement basic `QualityDropListener` behavior.
- [x] Implement basic `MotionConfounderListener` behavior.
- [x] Merge and rank listener outputs.
- [x] Deduplicate overlapping windows.
- [x] Preserve relative order.

Acceptance criterion:

- `event_windows` should be useful enough that a text-only LLM can cite concrete
  evidence rather than only abstract global statistics.

### Phase 6: Implement Global Aggregation

- [x] Aggregate screen gaze proxy ratio bin.
- [x] Aggregate maximum continuous gaze proxy duration bin.
- [ ] Aggregate blink-rate level and trend.
- [x] Aggregate posture trend proxy.
- [x] Aggregate interaction intensity.
- [x] Aggregate repetitive operation level.
- [x] Aggregate motion confounding level.

Acceptance criterion:

- `global_features` should summarize the whole session without hiding important
  temporal evidence that belongs in `event_windows`.

### Phase 7: Implement Privacy Compressor And Validator

- [x] Convert exact internal values to bins/ranges/categories.
- [x] Drop raw paths from evidence.
- [x] Drop exact absolute timestamps and frame indices.
- [x] Drop coordinates and landmark arrays.
- [x] Drop OCR/ASR/screen content.
- [x] Drop app names and questionnaire answers.
- [x] Drop exact heart-rate values.
- [x] Add schema validation.
- [x] Add privacy audit script.

Deliverables:

```text
experiments/privacy/audit_preprocessor_evidence.py
```

or:

```text
scripts/audit_privacy_evidence.py
```

### Phase 8: Run MVP And LLM Baseline

- [x] Run `privte_preprocessor_v0` on `6.1data`.
- [x] Run `privte_preprocessor_v0` on `all_current`.
- [x] Inspect rendered text evidence manually.
- [ ] Run text-only LLM baseline.
- [ ] Compare against labels.
- [ ] Audit false positives and false negatives.
- [ ] Tune listener thresholds and quality gating.

Expected output layout:

```text
outputs/quickstart/<subset>_privte_preprocessor_v0/
  preprocessor_v0_evidence.<subset>.jsonl
  preprocessor_v0_report.<subset>.json
  evidence_text/*.txt
```

## 10. Tuning Strategy

Future tuning should happen inside specific modules, not by changing the whole
schema.

Primary tuning knobs:

- window sampling density;
- event candidate threshold;
- quality threshold;
- listener-specific thresholds;
- event ranking and deduplication;
- bin boundaries for global features;
- text rendering phrasing;
- LLM prompt after evidence schema stabilizes.

Tuning should preserve:

- top-level schema;
- allowed enum values;
- privacy flags;
- no raw media or exact identifiers;
- separation of `global_features` and `event_windows`.

## 11. Evaluation Plan

Minimum checks before using outputs for LLM evaluation:

- schema validity;
- privacy audit pass;
- manual inspection of representative evidence text;
- label distribution check;
- LLM JSON validity;
- accuracy and macro-F1;
- high-risk / moderate-risk recall;
- confusion matrix;
- confidence calibration;
- human-review trigger rate;
- qualitative error analysis.

Important comparison settings:

- `manifest_only` baseline;
- `global_features` only;
- `event_windows` only;
- `global_features + event_windows`;
- with and without quality gating;
- with and without selected listener groups.

## 12. Immediate Next Steps

1. Run the text-only LLM baseline on the v0 schema.
2. Inspect false positives and false negatives against labels.
3. Decide whether the next module should be hand landmarks, stronger device
   detection, exact gaze/blink proxy, or AU extraction.
4. If the schema stays stable, move tuning into listener thresholds rather than
   changing the top-level evidence contract.
