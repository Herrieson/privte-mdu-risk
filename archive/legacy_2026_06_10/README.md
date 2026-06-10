# Legacy Archive 2026-06-10

This archive contains historical PriVTE work that is no longer the active
method line.

Archived content:

```text
docs/       # old paper plans, technical plans, and future-work drafts
methods/    # old FlowLite, Behavior, Trace, MVP, config, and helper scripts
            # also includes old LLM package/rendering support for those schemas
results/    # generated quickstart, LLM baseline, and analysis outputs
misc/       # temporary local files
```

The active docs directory now keeps only:

```text
docs/advisor_brief.md
```

New implementation work should rebuild around the clean evidence schema:

```text
global_features
event_windows
quality_summary
limitations
privacy_processing_summary
```
