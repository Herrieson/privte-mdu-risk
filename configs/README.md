# Configs

Central configuration files live here instead of inside `mvp/`.

```text
configs/
  algorithms/      # extractor-specific configs
```

Use `configs/algorithms/<algorithm>.<version>.json` for algorithm parameters.
Run scripts and CLIs should accept `--config` or `--extractor-config` so a run
can swap algorithms/configs without editing code.

Historical FlowLite, Behavior, and Trace configs were archived under
`archive/legacy_2026_06_10/methods/configs/algorithms/`.

Current active configs:

```text
configs/algorithms/privte_preprocessor.v0.json
configs/algorithms/privte_preprocessor.v1.json
configs/algorithms/privte_preprocessor.v2.json
```

`privte_preprocessor.v0.json` controls the current frame-proxy MVP:

- coverage window count;
- sampled frames per coverage window;
- analysis resize width;
- local/global motion thresholds;
- maximum emitted event windows.

`privte_preprocessor.v1.json` keeps the same public evidence schema but adds
calibration knobs for thresholded event selection, conservative repetitive
operation bins, and motion-confounding aggregation.

`privte_preprocessor.v2.json` keeps the V1 calibrated visual proxies and adds
stateful evidence settings for per-window symbolic states, behavior episodes,
temporal trends, and separated positive / weak / counter / quality evidence
facts.
