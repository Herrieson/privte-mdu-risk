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
