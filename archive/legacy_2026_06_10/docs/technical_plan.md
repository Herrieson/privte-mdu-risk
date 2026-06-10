# PriVTE 当前论文主线技术方案

## 0. 当前结论

当前 AAAI AISI 主线建议正式收敛为：

```text
PriVTE-Trace:
Privacy-preserving phase-level behavioral trace encoding
for text-only minor digital use risk screening.
```

中文理解：

```text
面向 text-only 风险筛查的隐私保护行为轨迹证据编码。
```

这不是“视频分类器”，也不是“视频成瘾诊断模型”，而是：

```text
local video proxy extraction
  -> privacy-filtered phase trace
  -> text-only LLM evidence synthesis
```

## 1. 为什么最终选择 PriVTE-Trace

### 1.1 隐私约束决定不能直接公开视频或高维视觉特征

原始视频包含未成年人面部、身体、场景、家庭/学校环境、设备内容和潜在身份线索。即使不发布原图，高维视觉 embedding、pose sequence、face mesh、latent video token 也可能泄露身份或敏感属性。

因此，公开层应坚持：

```text
no raw video
no keyframes
no audio
no OCR / ASR
no face embedding
no high-dimensional pose or face mesh sequence
no app names
no exact timestamps
```

结论：

```text
公开 benchmark 的主输入必须是结构化文本证据，而不是视觉数据或视觉 embedding。
```

### 1.2 标签来源决定不能声称视频诊断

现有标签似乎来自视频、心率、问卷、应用记录等多维综合判断。部署时若只用视频，就天然缺少问卷、长期损害、主观渴求、真实 app 行为和精确生理信号。

所以系统只能做：

```text
video-only proxy evidence screening
```

不能写成：

```text
video-based addiction diagnosis
```

结论：

```text
论文和数据卡必须说明 video-only proxy evidence 的边界；
但 LLM 输入应把 evidence package 视为该评测样本的完整输入，
不能反复提示问卷、心率或应用记录未作为输入。
```

因此，`missing_information` 和 `needs_human_review` 应用于表达证据包内部的质量、
可见性或行为证据不足，而不是提醒模型存在未输入的额外模态。

### 1.3 长视频约束决定不能只输出全局统计

当前 `v3_temporal` 的实验现象是：LLM 已经能读懂更丰富的时序证据，但仍倾向把样本压到 `mild_risk`。这说明单纯 episode 计数或全局比例不足以支撑更细的风险区分。

长视频中的关键信息不只是“出现过什么”，还包括：

- 什么时候开始出现；
- 是否持续；
- 是否反复；
- 是否从被动参与转向主动交互；
- 是否被姿态、镜头或场景运动混淆；
- 是否在后段消退或再次出现。

结论：

```text
证据单位应从 global stats / selected episodes
升级为 phase-level behavioral trace。
```

### 1.4 小数据和敏感场景决定不宜训练黑盒行为分类器

当前样本规模仍小，且标注本身来自多模态综合判断。直接训练视频分类模型会有三个问题：

- 容易过拟合；
- 难以解释；
- 论文贡献会偏离 privacy-preserving benchmark。

结论：

```text
主线不训练端到端视频分类器；
主线做 deterministic, auditable evidence encoder。
```

## 2. 最终系统架构

推荐最终流程：

```text
Raw sensitive video
  -> local frame / clip sampling
  -> ROI detection and focusing
  -> proxy feature extraction
  -> quality estimation
  -> phase segmentation
  -> transition scoring
  -> privacy filtering
  -> structured trace JSON
  -> templated text evidence
  -> text-only LLM screening
```

其中 LLM 只负责：

```text
evidence synthesis and risk screening
```

不负责：

```text
raw video understanding
diagnosis
identity inference
emotion diagnosis
```

## 3. 底层技术选择

### 3.1 视频读取和抽帧

当前实现：

```text
OpenCV VideoCapture
```

建议：

- 短期继续使用 OpenCV，因为仓库已跑通；
- 中期补 `FFmpeg / PyAV`，用于更稳定的长视频解码、精确采样和后续 streaming / zero-retention。

### 3.2 ROI 检测

当前建议：

```text
YOLO11n + bright-rectangle screen heuristic
```

职责划分：

- `YOLO11n`：检测手机、电脑、电视、键盘、遥控器等常见设备；
- bright-rectangle heuristic：补充平板、屏幕样区域和 YOLO 漏检场景；
- 后续如果漏检严重，可做小规模内部设备框标注和 YOLO fine-tuning。

### 3.3 手部、人脸和姿态代理

当前建议：

```text
MediaPipe Tasks
```

只用于本地生成粗粒度代理：

- 手部可见；
- 手-设备接近；
- 人脸/设备共现；
- 头部/设备上下文；
- 姿态中心变化；
- 质量和遮挡信息。

不输出：

- 手部关键点序列；
- face mesh；
- face embedding；
- 高维 pose sequence；
- 身份相关描述。

### 3.4 运动和交互代理

当前实现可继续保留：

```text
frame difference
```

后续 paper-grade 建议升级：

```text
ROI-level optical flow
```

目的不是识别具体点击或滑动，而是更好地区分：

- 设备区域局部活动；
- 手部接近设备后的局部运动；
- 全局镜头运动；
- 姿态/场景混淆。

### 3.5 phase trace 构建

主线建议：

```text
deterministic finite-state machine
  + hysteresis smoothing
  + transition scoring
  + phase compression
```

不建议一开始使用：

```text
black-box sequence classifier
```

原因：

- 样本规模不足；
- 风险筛查需要可解释；
- 输出要能被 privacy audit；
- 论文主线是 benchmark / evidence protocol，不是训练新视频模型。

## 4. 最终 evidence schema

PriVTE-Trace 的核心输出建议固定为：

```text
trace_header
trace_signature
trace_phases[]
transition_events[]
risk_cues[]
counterevidence[]
missing_information[]
privacy_processing_summary
needs_human_review
```

### 4.1 `trace_signature`

一行行为轨迹签名，例如：

```text
visible_only -> brief_passive_contact -> sustained_passive_engagement -> interaction_burst -> confounded_motion
```

### 4.2 `trace_phases[]`

每个 phase 包含：

- `phase_id`
- `relative_position`
- `phase_type`
- `duration_bin`
- `confidence`
- `support_level`
- `supporting_cues`
- `counterevidence`
- `risk_relevance`
- `privacy_filter`

推荐 phase 词表：

- `no_device_or_visible_only`
- `brief_passive_contact`
- `sustained_passive_engagement`
- `active_interaction_burst`
- `repetitive_operation_run`
- `motion_confounded_activity`
- `disengagement_or_gap`
- `insufficient_quality`

## 5. LLM 输入策略

论文、数据卡和伦理说明必须披露 video-only proxy evidence 的边界；但交给 LLM 的
evidence package 不应反复强调这些边界，否则会诱导模型过度保守。

LLM 输入遵循以下规则：

- 把 evidence package 视为该评测样本的完整输入；
- 不列出问卷、心率、应用记录等未作为输入的额外模态；
- 不把 `questionnaire_input`、`exact_heart_rate_input`、`app_name_input` 等内部缺失项渲染给 LLM；
- 不把 `proxy/代理` 作为主要表述，渲染文本中使用更中性的 `indicator/指标` 或“行为证据”；
- 不重复渲染“不能诊断”“不输出 OCR/ASR”“不输出图像”等隐私排除清单；
- `missing_information` 只用于 evidence package 内部明确显示的质量、可见性或行为证据不足；
- 若 evidence package 内部质量可用，则不要因为未呈现额外模态而选择 `insufficient_evidence`。

换句话说：

```text
privacy / ethics boundary:
  written in paper, dataset card, release policy

LLM evidence input:
  concise complete evidence package for this benchmark sample
```

## 6. 不采用的主线方案

### 6.1 不采用直接 Video-LLM / video captioning 作为主线

原因：

- 原始视频不能直接给通用模型；
- 自由 caption 可能生成场景、外貌、身份线索；
- 输出难以审计；
- 与 text-only benchmark 的贡献冲突。

可以作为内部 sanity check，但不作为论文主线。

### 6.2 不采用端到端视频分类器作为主线

原因：

- 数据量小；
- 标签来自多模态综合，不是纯视频标签；
- 易过拟合；
- 解释性弱；
- 不利于公开 benchmark。

### 6.3 不把 Zero-Retention 合入当前主线

Zero-Retention PriVTE 很有价值，但更像下一篇方法论文。

当前 AAAI AISI 主线应聚焦：

```text
PriVTE + MDU-RiskText + MDU-RiskBench
```

Zero-Retention 可以作为：

```text
future work / second paper / CVPR or ACM MM style method extension
```

## 7. 评测设计

最终实验至少包括：

1. **版本比较**

   ```text
   flowlite
   behavior_v1
   behavior_v2
   behavior_v3_temporal
   PriVTE-Trace
   ```

   当前工程实现中，PriVTE-Trace 的第一版入口为：

   ```text
   extractor: privte_trace_v1
   config: configs/algorithms/privte_trace.v1.json
   runner: mvp/run_trace_v1_mvp.py
   fast conversion: scripts/build_trace_v1_from_temporal_evidence.py
   ```

   `privte_trace_v1` 复用 Behavior v3 的本地视觉分析结果，但将 LLM 主证据从
   selected episodes 改为 normal-use-aware behavior trace，显式区分普通设备使用、
   低强度使用、风险性使用轨迹和证据内部质量不足。

2. **LLM baseline**

   ```text
   GPT-family
   Claude-family
   selected open-source LLMs
   ```

3. **核心指标**

   - accuracy；
   - macro F1；
   - per-label precision / recall / F1；
   - `no_observed_risk` recall；
   - `moderate_risk` recall；
   - invalid JSON rate；
   - abstention / insufficient evidence behavior；
   - human review trigger rate。

4. **隐私评估**

   - PII scan；
   - exact timestamp scan；
   - OCR/ASR leakage scan；
   - app name leakage scan；
   - appearance / scene description scan；
   - uniqueness / rare-combination audit。

5. **消融**

   - no transition events；
   - no counterevidence；
   - no phase compression；
   - no ROI motion；
   - global statistics only；
   - phase trace only。

## 8. 相关工作支撑点

当前方案被三类文献支撑：

### 8.1 长视频和时序理解

近年 long-video / Video-LLM benchmark 显示，长视频理解需要时序推理、关键片段选择和多步事件证据，不能只依赖单帧或全局统计。

相关工作包括：

- [MVBench](https://arxiv.org/abs/2311.17005)；
- [Video-MME](https://arxiv.org/abs/2405.21075)；
- [TempCompass](https://arxiv.org/abs/2403.00476)；
- [LongVideoBench](https://arxiv.org/abs/2407.15754)；
- [Adaptive Keyframe Sampling](https://arxiv.org/abs/2502.21271)；
- [VRBench](https://arxiv.org/abs/2506.10857)；
- [PerceptionComp](https://arxiv.org/abs/2603.26653)；
- [LVSum](https://arxiv.org/abs/2604.10024)。

这支撑 PriVTE-Trace 的 key-window、phase trace 和 transition event 设计。

### 8.2 隐私保护视频理解

近年隐私视频工作显示，像素匿名化并不充分，高维视觉特征和 latent token 也可能泄露身份、外貌或敏感属性。

相关工作包括：

- [Privacy Beyond Pixels](https://arxiv.org/abs/2511.08666)；
- [From Pixels to Privacy](https://arxiv.org/abs/2603.26336)；
- [PrivHAR-Bench](https://arxiv.org/abs/2604.00761)。

这支撑 PriVTE 不公开视觉 embedding、不公开 pose sequence、只公开隐私过滤文本证据的选择。

### 8.3 LLM 拒答和人审

近年 LLM abstention / deferral 研究显示，在不完整或高风险任务中，让模型知道何时不回答、何时交给人审，是可靠系统的重要组成部分。

相关工作包括：

- [Do LLMs Know When to NOT Answer?](https://arxiv.org/abs/2407.16221)；
- [Know Your Limits](https://arxiv.org/abs/2407.18418)；
- [AbstentionBench](https://arxiv.org/abs/2506.09038)；
- [Guided Deferral Systems with LLMs](https://arxiv.org/abs/2406.07212)。

这支撑 `insufficient_evidence`、`missing_information` 和 `needs_human_review` 成为正式输出字段。

## 9. 最终一句话方案

最终当前主线建议写成：

```text
We propose PriVTE-Trace, a privacy-preserving, phase-level video-to-text evidence encoding framework that converts sensitive long-form minor digital-use videos into structured behavioral trace evidence for text-only LLM-based risk screening, without releasing raw video, images, audio, OCR/ASR, high-dimensional visual features, or identity-bearing descriptions.
```

中文：

```text
我们提出 PriVTE-Trace，一种隐私保护的阶段级视频到文本证据编码框架，
将未成年人数字设备使用长视频转换成结构化行为轨迹证据，
使 text-only LLM 能够在不接触原始视频、图像、音频、OCR/ASR、
高维视觉特征或身份线索的情况下进行风险筛查。
```
