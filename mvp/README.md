# PriVTE Simple Algorithm MVP

This directory contains a runnable MVP profile on top of the stable
`../pipeline/` base.

The current MVP uses the `simple_video_quality` extractor. It is deliberately
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
  --output-dir mvp/outputs/6_1data_simple_video_quality \
  --subset-name 6.1data \
  --max-metadata-clips 16
```

Use `--metadata-all` if you want to probe every video file instead of a
deterministic sample per participant.

## Outputs

```text
mvp/outputs/6_1data_simple_video_quality/simple_mvp_evidence.6_1data.jsonl
mvp/outputs/6_1data_simple_video_quality/simple_mvp_report.6_1data.json
mvp/outputs/6_1data_simple_video_quality/evidence_text/*.txt
```

## What This MVP Proves

- The project now has a stable pipeline/algorithm boundary.
- The MVP performs a real computation over video files.
- The text-only input can be generated without exposing raw videos or frames.
- Later work only needs to replace the extractor with a stronger PriVTE visual
  proxy algorithm.
