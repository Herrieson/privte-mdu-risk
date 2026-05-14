# PriVTE

Privacy-preserving Video-to-Text Evidence Encoding for Minor Digital Use Risk Screening.

PriVTE is a research project for AAAI AI for Social Impact style work. The project studies how to transform sensitive long-form videos of minors' digital device use into structured, privacy-filtered textual evidence, so text-only LLMs and supervised baselines can perform risk screening without accessing raw videos, images, or audio.

## Core Positioning

This project is not a medical diagnostic system and does not claim to detect addiction from video. The safer framing is:

> video-only, privacy-preserving, text-only risk screening based on observable proxy evidence.

The intended contribution is:

1. A privacy-preserving video-to-text evidence encoder: `PriVTE`.
2. A field-derived textual evidence dataset: `MDU-RiskText`.
3. A text-only evaluation benchmark: `MDU-RiskBench`.

## Source Data

The source data come from an on-site field study in Ordos, Inner Mongolia. The original collected data include:

- approximately 30-minute videos of minors using digital devices;
- questionnaire responses about digital device use;
- physiological and affective cues, such as heart-rate-related signals and facial-expression observations;
- human-assigned risk labels based on questionnaire, physiological, affective, and behavioral evidence.

The Ordos field collection is a project highlight because it is real-world, sensitive, difficult to collect, and socially motivated. It should be described as a field-collected cohort, not as a population-representative survey.

Recommended phrasing:

> a field-collected cohort from an on-site study in Ordos, Inner Mongolia

Avoid framing the dataset as representative of all minors, all regions, Inner Mongolia, or any ethnic/cultural group.

## Method Overview

PriVTE converts sensitive videos into schema-first textual evidence through a local preprocessing pipeline:

```text
Raw Sensitive Video
  -> Key-window Extraction
  -> Region-of-Interest Focusing
  -> Proxy Feature Extraction
  -> Reference-sequence Normalization
  -> Quality Estimation
  -> Privacy-aware Text Generation
  -> Text-only LLM / Text Classifier
```

The method should prefer deterministic structured evidence over free-form video captions:

```text
structured visual feature extraction -> privacy filtering -> deterministic textual evidence
```

Do not use the primary framing:

```text
video -> free-form caption -> LLM diagnosis
```

## Dataset Release Principle

The public benchmark should release text, not raw videos.

Public release may include:

- anonymous sample IDs;
- risk labels;
- train/validation/test splits;
- coarse duration bins;
- global behavioral summaries;
- key event descriptions;
- data quality indicators;
- video-only limitations;
- privacy processing summaries.

Public release should not include:

- raw video;
- images or keyframes;
- audio;
- ASR transcripts;
- OCR text;
- exact heart rate or HRV;
- face embeddings;
- high-dimensional skeleton or face mesh sequences;
- school, address, class, district, precise date, or other identity-bearing fields;
- free-form appearance or scene descriptions.

## Recommended Paper Framing

Suggested title:

```text
PriVTE: Privacy-Preserving Video-to-Text Evidence Encoding for Minor Digital Use Risk Screening
```

One-sentence summary:

```text
We introduce PriVTE, a privacy-preserving video-to-text evidence encoding framework that transforms sensitive long-form videos of minors' digital device use into structured textual evidence, enabling text-only LLM-based risk screening and reproducible research without releasing raw videos.
```

Recommended AAAI AISI story:

```text
Social problem + privacy-preserving data release + text-only LLM benchmark + responsible AI workflow
```

Avoid making it sound like a generic video classifier.

## Current Planning Documents

- [AAAI AISI submission plan](docs/aaai_aisi_submission_plan.md)
- [Video-to-text preprocessor design](docs/video_to_text_preprocessor_design.md)

## Immediate Next Steps

1. Freeze the task definition, label names, and text schema.
2. Write the dataset card and ethics/release policy.
3. Implement a minimal PriVTE pipeline that can generate JSON and templated text from extracted features.
4. Build `MDU-RiskText` public-lite format.
5. Run baseline experiments: traditional classifiers, small LLM, strong LLM.
6. Run ablations: event windows, quality report, reference normalization, schema-first text vs free caption.
7. Run privacy checks: PII scan, manual audit, field uniqueness analysis.

## Ethical Boundaries

Use risk-screening language:

```text
no_observed_risk
mild_risk
moderate_risk
high_risk
insufficient_evidence
```

Avoid diagnostic or stigmatizing language:

```text
no_addiction
mild_addiction
severe_addiction
addict
detector
diagnosis
```

The system should output:

- risk level;
- confidence;
- supporting evidence;
- missing information;
- human review flag.

It should not be used for automatic diagnosis, punishment, student profiling, or high-stakes individual decisions.
