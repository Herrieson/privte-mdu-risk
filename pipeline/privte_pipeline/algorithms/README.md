# PriVTE Algorithms

This directory contains replaceable evidence extraction algorithms.

Algorithm modules should be self-contained and implement:

```python
class MyExtractor(EvidenceExtractor):
    name = "my_algorithm"
    version = "v0"
    feature_schema_version = "my_algorithm_feature_schema.v0"

    def extract(self, person_record: dict) -> dict:
        ...
```

The extractor should return the standard PriVTE evidence contract:

```text
extractor
modality_availability
feature_blocks
privacy_processing_summary
missing_information
limitations
```

The pipeline will then handle:

- target-label separation for evaluation;
- LLM evidence package assembly;
- text rendering;
- JSONL/report/text writing.

To add a new algorithm:

1. Add `algorithms/<algorithm_name>.py`.
2. Implement `EvidenceExtractor`.
3. Add the class to `algorithms/registry.py`.
4. Add a config under `configs/algorithms/` if needed.
5. Run `pipeline/build_evidence.py --extractor <algorithm_name>`.

Do not output raw videos, frames, crops, coordinates, OCR/ASR, face embeddings,
high-dimensional landmarks, app names, questionnaire answers, exact heart-rate
values, raw paths, or exact timestamps.

Current algorithm modules:

```text
manifest_only.py          # schema/data-flow baseline
privte_preprocessor_v0.py # active frame-proxy preprocessor MVP
```

Archived historical algorithms are under
`archive/legacy_2026_06_10/methods/pipeline_algorithms/`. New work should not
extend those versions; add a fresh extractor for the listener-based
`global_features + event_windows` schema.
