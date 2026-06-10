# Video-to-Text Preprocessor Design

This document is kept as the stable entry point for the PriVTE video-to-text
preprocessor design.

The current paper-line technical plan has converged to PriVTE-Trace:

```text
local video proxy extraction
  -> privacy-filtered phase trace
  -> templated text evidence
  -> text-only LLM screening
```

See the current full technical plan:

- [PriVTE current technical plan](technical_plan.md)

Core design constraints remain:

- no raw video in public artifacts;
- no images or keyframes;
- no audio, OCR, ASR, app names, or exact timestamps;
- no face embeddings or high-dimensional pose / face mesh sequences;
- deterministic schema-first evidence generation;
- explicit missing information and human-review flags.

