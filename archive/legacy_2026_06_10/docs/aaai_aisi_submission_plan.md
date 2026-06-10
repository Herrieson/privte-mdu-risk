# AAAI AISI Track 投稿方案：未成年人电子产品使用风险筛查的隐私保护视频文本证据编码

## 1. 投稿定位

本项目更适合按照 AAAI AI for Social Impact, AISI track 的逻辑组织，而不是按照纯计算机视觉方法论文组织。

推荐论文定位：

> 在未成年人敏感视频不能公开、也不应直接交给通用多模态 LLM 的条件下，构建一个可复现、可审计、隐私保护的 text-only AI 风险筛查框架与评测基准。

核心问题不是“如何用视频诊断成瘾”，而是：

> 在 video-only、privacy-preserving、text-only LLM 的约束下，AI 系统能否基于可观察代理证据进行负责任的风险筛查？

推荐英文标题方向：

```text
Privacy-Preserving Video-to-Text Evidence Encoding for LLM-based Risk Screening of Minor Digital Device Use
```

更短的标题方向：

```text
PriVTE: Privacy-Preserving Video-to-Text Evidence Encoding for Minor Digital Use Risk Screening
```

中文理解：

```text
面向未成年人电子产品使用风险筛查的隐私保护视频到文本证据编码
```

## 2. 一句话摘要

建议将论文的一句话摘要写成：

```text
We introduce PriVTE, a privacy-preserving video-to-text evidence encoding framework that transforms sensitive long-form videos of minors' digital device use into structured textual evidence, enabling text-only LLM-based risk screening and reproducible research without releasing raw videos.
```

对应中文：

```text
我们提出 PriVTE，一个隐私保护的视频到文本证据编码框架，将未成年人电子产品使用的敏感长视频转换为结构化文本证据，使 text-only LLM 能够在不接触原始视频的条件下进行风险筛查，并支持可复现研究。
```

## 3. AISI 适配性

AAAI AISI track 关注的是 AI 对社会问题的建模、方法、评估和实际影响，而不仅是算法指标提升。本项目与 AISI 的契合点包括：

- 未成年人电子产品过度使用风险是现实社会问题。
- 原始视频、声音、表情、生理代理特征和风险标签高度敏感。
- 直接公开原始视频不现实，也不合伦理。
- 直接把未成年人视频输入通用多模态 LLM 存在隐私和治理风险。
- 项目提出了一种在隐私保护约束下支持后续研究的数据发布和模型评测范式。
- 系统输出定位为风险筛查，不是医学或心理诊断。

因此，文章应突出：

```text
Social problem + Privacy-preserving data release + Text-only LLM benchmark + Responsible AI workflow
```

而不是突出：

```text
一个普通的视频分类任务
```

## 4. 核心论点

论文的核心论点建议写成：

> 未成年人电子产品使用风险筛查需要利用视频行为线索，但原始视频高度敏感，无法直接公开或输入通用多模态 LLM。本文提出一种隐私保护的视频到文本证据编码框架，将敏感长视频转换为结构化文本证据，并构建一个 text-only LLM 风险筛查基准，在保护隐私的同时支持可复现研究。

需要明确三个边界：

1. 系统不是医学诊断器。
2. 视频只能提供可观察代理证据，不能完整还原问卷、长期行为损害或主观渴求。
3. 文本化数据仍可能是敏感派生数据，需要隐私审查和分级发布。

## 5. 推荐贡献

建议写成四个贡献。

### 5.1 问题建模贡献

提出一个新的社会影响 AI 问题：

```text
Privacy-preserving text-only risk screening from sensitive minor videos.
```

该问题具有以下挑战：

- 数据主体是未成年人。
- 原始视频、声音、面部、姿态、生理代理特征和风险标签均高度敏感。
- 传统公开视频数据集的方式不可行。
- 通用 LLM 或多模态 LLM 直接处理原始视频存在隐私暴露风险。
- 标签来自问卷、心率、表情和人工综合判断，但部署输入只有视频，存在天然信息缺口。

### 5.2 方法贡献

提出多管线前置组件：

```text
PriVTE: Privacy-preserving Video-to-Text Evidence Encoder
```

组件将原始视频转换成结构化文本证据，供 text-only LLM 使用。

核心模块包括：

- 关键时间窗提取器。
- 关键区域聚焦器。
- 参考序列特征提取器。
- 质量评估器。
- 隐私过滤器。
- schema-first 文本证据生成器。

重点强调：

```text
structured visual feature extraction -> privacy filtering -> deterministic textual evidence
```

而不是：

```text
free-form video captioning
```

### 5.3 数据集贡献

发布一个文本化行为证据数据集，可命名为：

```text
MDU-RiskText
```

含义：

```text
Minor Digital Use Risk Text Evidence Benchmark
```

数据集包含：

- 匿名样本 ID。
- 风险标签。
- 结构化文本证据。
- 关键事件文本。
- 数据质量指标。
- video-only 限制说明。
- train、validation、test split。
- 数据使用协议。

明确不包含：

- 原始视频。
- 图像。
- 音频。
- 原始 ASR。
- 原始 OCR。
- 精确心率。
- face embedding。
- 高维骨架序列。
- 可识别场景描述。

### 5.4 评测贡献

建立 text-only LLM 风险分级 benchmark。

评测内容包括：

- 不同 LLM 的风险分级能力。
- 不同 LLM、prompt 和证据粒度设置的比较。
- 不同文本证据粒度的影响。
- 前置组件消融。
- 隐私保护和筛查性能之间的权衡。
- 证据不足、低置信度和人工复核触发机制。

## 6. 方法设计

## 6.1 总体流程

推荐方法图：

```text
Raw Sensitive Video
        |
        v
Local Visual Analysis
  - key-window selection
  - ROI focusing
  - behavior/event extraction
  - reference-sequence normalization
        |
        v
Privacy Filtering
  - remove identity cues
  - remove OCR/ASR PII
  - aggregate and discretize features
        |
        v
Text Evidence Generation
  - JSON evidence
  - templated natural language evidence
        |
        v
Text-only Risk Screening
  - text-only LLM
  - risk level
  - confidence
  - missing information
  - human review flag
```

## 6.2 PriVTE 模块

### M1. Key-window Extractor

从 30 分钟长视频中选择具有代表性和判断价值的时间窗。

窗口类型包括：

- 覆盖性窗口：保证全程覆盖。
- 事件窗口：捕捉交互强度突增、负向表情峰值、姿态突变、长时间注视等事件。
- 高质量窗口：选择人脸、手部、设备可见度较好的片段。
- 基线窗口：用于后续参考序列归一化。

### M2. ROI Focuser

定位并分析关键区域：

- 面部区域。
- 眼部和头部方向区域。
- 手部区域。
- 身体姿态区域。
- 电子设备区域。
- 屏幕区域。
- 背景敏感区域。
- 其他人区域。

注意：ROI 的目标不是把裁剪图交给 LLM，而是让本地 CV 模型只分析必要区域，并辅助隐私过滤。

### M3. Proxy Feature Extractor

提取可观察代理特征：

- 屏幕注视比例。
- 最大连续屏幕注视时长。
- 离屏次数。
- 点击和滑动频率。
- 重复操作比例。
- 操作强度趋势。
- 头部前倾和身体姿态。
- 眨眼频率。
- 面部动作单元 AUs。
- 负向表情趋势。
- 疑似挫败、烦躁、疲劳、过度沉浸片段。
- 可选 rPPG 心率趋势。

rPPG 只能作为带质量门控的代理特征，不能作为接触式心率真值。

### M4. Reference-sequence Normalizer

使用个体内基线进行归一化，减少个体差异。

参考序列可包括：

- 前 3 到 5 分钟稳定片段。
- 低交互片段。
- 全视频中位数。
- 同年龄组群体参考，但需谨慎使用。

输出应优先使用相对变化：

```text
touch_rate: +120% above baseline
blink_rate: 35% below baseline
negative_affect: +2.1 z-score above baseline
```

而不是只输出绝对值。

### M5. Quality Estimator

给每类特征输出质量信息，避免低质量数据误导 LLM。

质量指标包括：

- 人脸可见比例。
- 手部可见比例。
- 设备可见比例。
- 屏幕可见比例。
- 光照质量。
- 遮挡程度。
- 多人干扰程度。
- rPPG 可用比例。
- 有效观察时长。

### M6. Privacy-aware Text Generator

将结构化特征转换为可供 LLM 使用的文本证据。

原则：

- 不使用自由 caption 作为主要输出。
- 不输出外貌、服装、家庭、学校、地点等身份线索。
- 不公开 OCR 原文和 ASR 原文。
- 精确时间戳转换为时间段。
- 精确心率转换为趋势或区间。
- 精确坐标转换为姿态类别。
- 高频序列转换为聚合统计。
- 所有输出遵循固定 schema。

## 7. LLM 风险筛查设计

LLM 的角色是证据综合器，而不是诊断器。

输入：

```text
schema-first textual evidence
```

输出：

```text
risk level + confidence + supporting evidence + missing information + human review flag
```

推荐标签：

```text
no_observed_risk
mild_risk
moderate_risk
high_risk
insufficient_evidence
```

不建议使用：

```text
no_addiction
mild_addiction
severe_addiction
```

原因是“成瘾”具有医学和污名化含义，而单段视频更适合辅助筛查。

输出 schema 示例：

```json
{
  "risk_level": "mild_risk",
  "confidence": 0.68,
  "supporting_evidence": [
    "screen_attention_high",
    "interaction_intensity_increased_late",
    "frustration_related_event_detected"
  ],
  "missing_information": [
    "no_questionnaire",
    "no_long_term_usage_record",
    "no_contact_heart_rate"
  ],
  "needs_human_review": true
}
```

## 8. 数据集设计

## 8.1 数据集名称

建议使用：

```text
MDU-RiskText
```

全称：

```text
Minor Digital Use Risk Text Evidence Benchmark
```

## 8.2 数据集字段

公开版数据集建议包含：

- sample_id。
- label。
- split。
- duration_bin。
- quality_summary。
- global_text。
- event_texts。
- structured_json。
- limitations。
- privacy_processing_summary。

示例：

```json
{
  "sample_id": "anon_pub_0421",
  "label": "mild_risk",
  "split": "test",
  "duration_bin": "25-30min",
  "quality": "medium",
  "global_text": "该样本在有效观察时长内表现出高屏幕注视比例、中高操作强度和后半段交互增强趋势。",
  "event_texts": [
    "08-10分钟出现操作强度升高，并伴随挫败相关面部动作增加。",
    "17-19分钟出现持续前倾和低眨眼频率，提示高专注或疲劳相关状态。"
  ],
  "limitations": [
    "video_only",
    "no_questionnaire",
    "no_long_term_record"
  ],
  "privacy_processing_summary": [
    "no_raw_video",
    "no_image",
    "no_audio",
    "no_ocr_text",
    "no_asr_text",
    "coarse_time_bins"
  ]
}
```

## 8.3 数据集统计表

论文中建议至少报告：

- 参与者数量。
- 样本数量。
- 视频时长分布。
- 标签分布。
- train、validation、test 划分。
- 平均事件窗口数。
- 平均文本长度。
- 数据质量分布。
- 隐私过滤统计。
- PII 扫描通过率。

如果样本量不大，应诚实说明真实敏感数据收集困难，并强调公开文本化数据集的研究价值。

## 8.4 数据发布分级

### Public Lite

可公开版本。

包含：

- 粗粒度文本证据。
- 风险标签。
- 数据质量等级。
- 事件类别。
- 不可判断信息。
- 固定 split。

不包含：

- 原始视频。
- 图片。
- 音频。
- ASR 原文。
- OCR 原文。
- 精确心率。
- 精确坐标。
- face embedding。
- 高维骨架序列。
- 地域、学校、家庭描述。
- 自由 caption。

### Controlled Research

受控研究版本，仅提供给签署数据使用协议的研究者。

可以包含：

- 更细粒度窗口特征。
- 更详细事件序列。
- 相对归一化数值。
- 更多质量指标。

仍不包含原始视频、图像、音频、ASR/OCR 原文或可重识别高维生物特征。

### Raw/Internal

内部版本，不公开。

包含：

- 原始视频。
- 原始音频。
- 原始 OCR/ASR。
- 中间图像。
- 高维特征。
- 标注记录。

## 9. 实验设计

实验应围绕五个研究问题组织。

## 9.1 RQ1: 文本化证据能否支持风险筛查？

比较设置：

- 开源小型 LLM。
- 开源较强 LLM。
- 闭源强 LLM。
- Zero-shot prompt。
- Rubric-based prompt。
- JSON-constrained output。
- Self-consistency 或 majority vote，可选。

指标：

- Accuracy。
- Macro F1。
- Weighted F1。
- Per-class recall。
- High-risk recall。
- Calibration error。
- Confusion matrix。

重点不应只看 accuracy。高风险召回、宏平均 F1 和校准更重要。

## 9.2 RQ2: 哪些证据最有用？

消融设置：

- Global summary only。
- Global + fixed windows。
- Global + event windows。
- Global + event windows + quality report。
- Global + event windows + reference normalization。
- Full PriVTE。

目标：

- 证明关键事件、质量报告和参考序列归一化对性能或稳定性有贡献。

## 9.3 RQ3: Schema-first 文本是否优于自由 caption？

比较：

- Free caption。
- Rule-based templated text。
- Structured JSON。
- JSON + natural language template。

预期：

- schema-first 文本更稳定。
- schema-first 文本隐私泄露更少。
- schema-first 文本更容易复现和审计。

## 9.4 RQ4: 隐私保护和模型性能如何权衡？

设计四个证据粒度：

```text
Level 0: Full internal features，仅作内部上限，不公开
Level 1: Fine-grained text，较细时间窗和数值区间
Level 2: Public Lite text，粗时间窗、趋势和事件类别
Level 3: Minimal text，仅全局摘要和少量事件
```

评估：

- screening performance。
- PII leakage rate。
- uniqueness / rare combination risk。
- manual privacy audit pass rate。

建议绘制：

```text
privacy-utility curve
```

## 9.5 RQ5: video-only 的信息上限在哪里？

如果可以访问完整标签来源，建议比较：

- 问卷 + 心率 + 表情 + 视频。
- 视频 only 原始结构化特征。
- 视频 only 文本证据。
- 视频 only 文本证据 + LLM。

目的：

- 诚实量化 video-only 与完整评估之间的信息差。
- 说明系统适用于风险筛查，而不是替代完整心理或临床评估。

## 10. 隐私评估

AISI 版本必须认真做隐私评估，否则容易被认为只是把视频换成文本而已。

## 10.1 PII 自动扫描

检查文本中是否残留：

- 人名。
- 学校。
- 班级。
- 电话。
- 账号。
- App 用户名。
- 地点。
- 聊天内容。
- 家庭成员称谓。

## 10.2 人工隐私审查

重点抽检：

- 高风险样本。
- 长文本样本。
- 罕见事件样本。
- 包含屏幕事件的样本。
- 包含声音事件的样本。
- 自动 PII 检测置信度低的样本。

## 10.3 粒度压缩分析

证明公开版数据不包含：

- 精确心率。
- 精确坐标。
- 逐帧序列。
- 屏幕 OCR 原文。
- ASR 原文。
- 外貌描述。
- 场景身份线索。

## 10.4 重识别风险代理指标

可以做 uniqueness test。

例如计算以下公开字段组合下有多少样本唯一：

```text
label + duration_bin + quality + event_types
label + event_types + global_features
label + duration_bin + event_count + quality
```

如果唯一性过高，应进一步合并区间或减少公开字段。

## 11. LLM 评测方式

不要只做 zero-shot。

建议至少包括：

- Zero-shot。
- Few-shot。
- Rubric-based prompting。
- JSON-constrained output。
- Self-consistency 或 majority vote，可选。
- 小型 LLM 与强 LLM 的对比。
- rubric prompt 与 JSON-constrained output。

更好的主结论不是“某个闭源 LLM 最好”，而是：

> 结构化证据表达比单纯更换模型更关键。

因此建议比较：

```text
same model + different evidence
different model + same evidence
```

这样可以证明前置组件本身的价值。

## 12. 论文结构建议

推荐正文结构：

```text
1. Introduction
   - 未成年人敏感视频风险
   - LLM 直接处理原始视频的问题
   - 本文提出 text-only privacy-preserving benchmark

2. Related Work
   - digital device use / problematic internet use screening
   - behavioral video understanding
   - privacy-preserving ML / de-identification
   - LLMs for decision support / social impact AI

3. Problem Formulation
   - video-only risk screening
   - privacy constraints
   - text-only evidence setting
   - output labels and limitations

4. PriVTE Method
   - key-window extraction
   - ROI focusing
   - reference normalization
   - quality estimation
   - privacy-aware text generation

5. Dataset
   - data source
   - label construction
   - text evidence schema
   - privacy filtering and release levels
   - ethical safeguards

6. Experiments
   - models
   - metrics
   - main results
   - ablations
   - privacy-utility analysis
   - video-only upper-bound analysis

7. Discussion
   - social impact
   - deployment workflow
   - human review
   - limitations
   - risks and misuse prevention

8. Conclusion
```

## 13. AISI 审稿标准对齐

论文应尽量对齐 AISI track 常见关注点。

| AISI 关注点 | 本文对应内容 |
| --- | --- |
| Problem significance | 未成年人数字设备使用风险，敏感数据难以公开 |
| Literature engagement | 连接行为风险筛查、视频理解、隐私保护、LLM 决策支持 |
| Novelty of approach | 隐私保护 video-to-text evidence encoding protocol |
| Justification | 解释为什么不用原始视频、自由 caption 或多模态 LLM |
| Quality of evaluation | 真实数据、多模型评估、消融、隐私-效用分析 |
| Follow-up work | 公开 Public Lite 数据集、schema、prompt、代码、评测脚本 |
| Scope and promise | 可作为辅助筛查和研究基准，不替代诊断 |

## 14. 最小可投版本

如果时间紧，最低限度需要完成：

- PriVTE 跑通，能批量生成 JSON 和模板化文本。
- 至少一个 Public Lite 文本数据集。
- 至少三类 LLM 设置比较：小型 LLM、强 LLM、rubric/JSON 约束输出。
- 至少四个消融：无事件、无质量、无参考归一化、自由 caption vs schema。
- 至少一个隐私评估：PII 扫描 + 人工抽检 + 字段唯一性。
- 明确伦理和使用限制。

如果缺少这些，论文容易像 proposal，而不是完整研究。

## 15. 理想可投版本

更强版本应包含：

- 真实数据规模足够，类别分布清楚。
- 标签有复核机制或一致性指标。
- 完整 video-only 上限实验。
- privacy-utility curve。
- 开源 PriVTE 代码、文本 schema、prompt、评测脚本。
- Public Lite 数据集可下载，Controlled Research 数据有申请机制。
- 专家评估：专家是否认为文本证据可解释，是否适合人工复核。

## 16. 需要避免的写法

不要写：

```text
我们用视频判断未成年人是否成瘾。
```

建议写：

```text
我们基于视频中可观察代理证据进行风险筛查。
```

不要写：

```text
我们完全保护了隐私。
```

建议写：

```text
我们降低了原始身份信息暴露，并通过粒度压缩、PII 审查和重识别风险分析控制残余风险。
```

不要写：

```text
LLM 是最终诊断器。
```

建议写：

```text
LLM 是证据综合器，输出风险等级、依据、置信度和人工复核建议。
```

不要写：

```text
rPPG 提供心率真值。
```

建议写：

```text
rPPG 在质量满足条件时提供心率趋势代理特征，并带置信度和质量门控。
```

## 17. 时间线建议

如果目标是 AAAI-27，应以官方 CFP 为准。参考 AAAI-26 的节奏，摘要和全文截稿在 7 月底到 8 月初附近，因此 AAAI-27 也可能在相近时间段，实际日期需要等待官方发布或确认。

从 5 月中旬开始，建议按 10 周倒排：

### 第 1 周

- 冻结任务定义。
- 冻结标签名称。
- 冻结文本 schema。
- 冻结公开字段清单。

### 第 2-3 周

- 跑通 PriVTE。
- 生成第一版 JSON 和模板化文本。
- 初步检查输出是否包含隐私泄露。

### 第 4 周

- 完成 PII 清洗。
- 完成人工隐私抽检。
- 完成 train、validation、test split。

### 第 5-6 周

- 完成主实验。
- 比较传统模型、小型 LLM、强 LLM。
- 输出主要性能表格。

### 第 7 周

- 完成消融实验。
- 包括事件窗口、质量报告、参考归一化、文本粒度、schema vs caption。

### 第 8 周

- 完成隐私评估。
- 完成 privacy-utility curve。
- 完成 uniqueness test。

### 第 9 周

- 完成论文初稿。
- 完成图表。
- 完成伦理声明。
- 完成数据集说明。

### 第 10 周

- 补实验。
- 压缩页数。
- 内部审稿。
- 准备 supplementary、代码和数据。

## 18. 投稿材料包

建议投稿前准备：

- Main paper。
- Supplementary appendix。
- Dataset card。
- Model/prompt card。
- Ethics statement。
- Data use agreement。
- Code repository。
- Evaluation script。
- Public Lite dataset。

AISI track 通常重视后续研究便利性，因此公开 schema、prompt、split 和评测脚本会明显增强论文可信度。

## 19. 风险与应对

### 风险 1：被认为只是工程 pipeline

应对：

- 明确 PriVTE 是 evidence encoding protocol。
- 做消融证明每个模块的贡献。
- 做 schema-first vs free caption 对比。
- 做隐私-效用曲线。

### 风险 2：被质疑 video-only 无法预测标签

应对：

- 明确视频 only 的信息上限。
- 做完整模态与 video-only 对比。
- 将输出定位为风险筛查，不是诊断。
- 输出中包含 missing information 和 insufficient_evidence。

### 风险 3：被质疑公开文本仍泄露隐私

应对：

- 做 PII 扫描。
- 做人工隐私抽检。
- 做字段唯一性分析。
- 分级发布 Public Lite、Controlled Research、Raw/Internal。

### 风险 4：被质疑社会影响被夸大

应对：

- 强调辅助筛查、研究基准和人工复核。
- 不声称替代家庭、学校、心理咨询或临床评估。
- 写清楚误用风险和禁止用途。

## 20. 最终建议

AAAI AISI 版本应主打：

```text
社会影响问题 + 隐私保护数据发布范式 + text-only LLM 风险筛查基准
```

不要主打：

```text
一个很强的视频识别系统
```

最稳的论文叙事是：

> 原始未成年人视频具有重要行为信息，但高度敏感、不可直接公开、也不适合直接输入通用 LLM。我们提出 PriVTE，将敏感视频转换为结构化、隐私过滤、质量感知的文本证据，并发布文本化风险筛查 benchmark。实验表明，schema-first 文本证据可以在降低隐私暴露的同时支持 text-only LLM 进行可解释风险筛查，但系统仍应被用于辅助筛查和人工复核，而非自动诊断。

如果能把数据集、隐私评估、消融实验和使用限制做扎实，这个方向投 AAAI AISI 是合理且有竞争力的。

## 21. 参考链接

- AAAI-26 AI for Social Impact Track Call for Papers: https://aaai.org/conference/aaai/aaai-26/aisi-call/
- AAAI-26 Main Technical Track Call for Papers: https://aaai.org/conference/aaai/aaai-26/main-technical-track-call/
- AAAI-26 Submission Instructions: https://aaai.org/conference/aaai/aaai-26/submission-instructions/
- NIST De-identification: https://www.nist.gov/itl/iad/deidentification
- NIST AI Risk Management Framework: https://www.nist.gov/itl/ai-risk-management-framework
- 《中华人民共和国个人信息保护法》: https://www.npc.gov.cn/npc/c2/c30834/202108/t20210820_313088.html
- WHO Gaming disorder FAQ: https://www.who.int/standards/classifications/frequently-asked-questions/gaming-disorder
