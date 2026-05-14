# AGENTS.md - PriVTE Project

This repository is the working project for PriVTE:

```text
Privacy-preserving Video-to-Text Evidence Encoding for Minor Digital Use Risk Screening
```

The project is intended for an AAAI AI for Social Impact style submission.

## Read First

At the start of a session in this repo, read:

1. `README.md`
2. `docs/aaai_aisi_submission_plan.md`
3. `docs/video_to_text_preprocessor_design.md`

These files contain the project framing, terminology, ethical boundaries, and current paper plan.

## Core Names

- Project / method: `PriVTE`
- Full method name: `Privacy-preserving Video-to-Text Evidence Encoding`
- Dataset: `MDU-RiskText`
- Benchmark: `MDU-RiskBench`
- Task: `Text-only Minor Digital Use Risk Screening`

## Core Framing

The project should be framed as:

```text
video-only, privacy-preserving, text-only risk screening based on observable proxy evidence
```

Do not frame it as:

```text
video-based addiction diagnosis
```

LLMs are evidence synthesizers, not diagnostic authorities. They should receive structured textual evidence only, and output risk level, evidence, confidence, missing information, and a human-review flag.

## Dataset Context

The source data were collected through an on-site field study in Ordos, Inner Mongolia. This is a key project highlight because the data are real-world, sensitive, difficult to collect, and socially motivated.

Use careful wording:

```text
a field-collected cohort from an on-site study in Ordos, Inner Mongolia
```

Avoid claiming representativeness beyond the collected cohort. Do not imply that the dataset represents all minors, all regions, Inner Mongolia, or any ethnic/cultural group.

## Privacy Rules

The public dataset should not contain:

- raw videos;
- images or keyframes;
- audio;
- ASR transcripts;
- OCR text;
- exact heart rate or HRV;
- face embeddings;
- high-dimensional skeleton or face mesh sequences;
- school, address, class, district, precise date, or other identity-bearing fields;
- free-form appearance or scene descriptions.

Treat textual evidence as sensitive derived data. Text-only does not mean anonymous by default.

## Preferred Technical Approach

Use schema-first evidence generation:

```text
structured visual feature extraction -> privacy filtering -> deterministic textual evidence
```

Avoid using free-form video captioning as the primary pipeline.

PriVTE modules:

1. Key-window extractor
2. Region-of-interest focuser
3. Proxy feature extractor
4. Reference-sequence normalizer
5. Quality estimator
6. Privacy-aware text generator

## Recommended Labels

Use:

```text
no_observed_risk
mild_risk
moderate_risk
high_risk
insufficient_evidence
```

Avoid:

```text
no_addiction
mild_addiction
severe_addiction
addict
diagnosis
detector
```

## AAAI AISI Story

The paper should emphasize:

```text
social problem + privacy-preserving data release + text-only LLM benchmark + responsible AI workflow
```

The main contributions should be:

1. Problem formulation for privacy-preserving text-only screening from sensitive minor videos.
2. PriVTE as a video-to-text evidence encoding protocol.
3. MDU-RiskText as a field-derived textual evidence benchmark.
4. MDU-RiskBench as a text-only LLM and classifier evaluation benchmark.

## Immediate Work Items

- Freeze the evidence schema.
- Draft a dataset card.
- Draft an ethics and release policy.
- Define public-lite, controlled-research, and internal-raw data tiers.
- Implement the minimal PriVTE pipeline.
- Add baseline model scripts.
- Add privacy audit scripts.
- Add paper outline and experiment table templates.
