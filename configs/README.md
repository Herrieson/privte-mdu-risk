# Configs

Central configuration files live here instead of inside `mvp/`.

```text
configs/
  algorithms/      # extractor-specific configs
```

Use `configs/algorithms/<algorithm>.<version>.json` for algorithm parameters.
Run scripts and CLIs should accept `--config` or `--extractor-config` so a run
can swap algorithms/configs without editing code.
