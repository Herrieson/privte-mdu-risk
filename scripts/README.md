# Scripts

This directory contains internal data-preparation scripts for PriVTE / MDU-RiskText.

All generated manifests are internal by default. They may contain raw paths, raw
participant IDs, exact timestamps, and label summaries. Do not treat them as
public release artifacts.

## `build_dataset_manifest.py`

Builds dataset-level, person-level, and clip-level manifests from a flat raw data
directory.

Expected input layout:

```text
data_root/
  210040_VID_2025_09_12_10_45_29_000/
    clip_000.mp4
    questionnaire.json
    questionnaire_list.png
    usage.json
    heartRate.json
    heartRate.png
```

Example:

```bash
uv run python scripts/build_dataset_manifest.py \
  --data-root data/6.1data \
  --output-dir data/manifests \
  --subset-name 6.1data
```

Outputs:

```text
data/manifests/dataset_manifest.<subset>.v0.json
data/manifests/internal_person_manifest.<subset>.v0.jsonl
data/manifests/internal_clip_manifest.<subset>.v0.jsonl
```

Notes:

- person manifest is one row per participant;
- clip manifest is one row per clip and is mainly for processing pipelines;
- split must be subject-wise, not clip-wise;
- raw questionnaire answers, app names, heart-rate values, images, and video
  contents are not inlined.

## `join_labels_to_manifest.py`

Joins `data/label.xlsx` into existing person and clip manifests.

Matching strategy:

```text
clip label:
  exact segment directory name matched inside label path columns

person label:
  aggregate exact clip-level matches;
  if no clip match exists, fall back to participant raw id
```

Example:

```bash
uv run python scripts/join_labels_to_manifest.py \
  --person-manifest data/manifests/internal_person_manifest.6_1data.v0.jsonl \
  --clip-manifest data/manifests/internal_clip_manifest.6_1data.v0.jsonl \
  --label-xlsx data/label.xlsx
```

Outputs:

```text
data/manifests/internal_person_manifest.6_1data.v0.labeled.jsonl
data/manifests/internal_clip_manifest.6_1data.v0.labeled.jsonl
data/manifests/label_join_report.6_1data.v0.labeled.json
```

The joined label field contains risk-label summaries, raw label count summaries,
matched label row indices, and dimension confidence summaries. It does not inline
raw path columns from the Excel file.

## `build_combined_manifest.py`

Builds one internal manifest from multiple flat raw data roots. Use this when a
comparison should run on the current total dataset instead of one uploaded batch.
The script reassigns `person_uid` globally across all roots, so IDs do not
collide across per-batch manifests.

Example:

```bash
uv run python scripts/build_combined_manifest.py \
  --data-root data/6.1data \
  --data-root data/数据测试2 \
  --data-root data/6.5数据测试 \
  --output-dir data/manifests \
  --subset-name all_current
```

Then join labels:

```bash
uv run python scripts/join_labels_to_manifest.py \
  --person-manifest data/manifests/internal_person_manifest.all_current.v0.jsonl \
  --clip-manifest data/manifests/internal_clip_manifest.all_current.v0.jsonl \
  --label-xlsx data/label.xlsx
```

Historical Trace/V3 refresh and conversion scripts were archived under
`archive/legacy_2026_06_10/methods/scripts/`.

## `audit_privacy_evidence.py`

Audits generated evidence JSONL public-input fields for obvious privacy leaks,
including raw paths, raw video directory IDs, raw clip filenames, image
filenames, and privacy flags that should remain false.

Example:

```bash
uv run python scripts/audit_privacy_evidence.py \
  outputs/quickstart/6_1data_privte_preprocessor_v0/preprocessor_v0_evidence.6_1data.jsonl
```

The audit targets `llm_evidence_package`, `text_evidence`, and
`feature_blocks.preprocessor_evidence`. It does not treat internal manifests as
public release artifacts.
