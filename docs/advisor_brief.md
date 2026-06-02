# PriVTE 项目简要说明（导师版）

## 1. 论文定位

本项目建议按 AAAI AI for Social Impact 方向组织，定位为：

```text
面向未成年人数字设备使用风险筛查的隐私保护视频到文本证据编码与 text-only LLM 评测基准
```

核心不是做“视频诊断成瘾”，而是研究：

```text
在原始未成年人视频不能公开、也不应直接交给通用多模态大模型的条件下，
能否将视频中的可观察代理证据转化为结构化文本，
从而支持可复现、可审计、隐私保护的风险筛查评测。
```

推荐英文题目：

```text
PriVTE: Privacy-Preserving Video-to-Text Evidence Encoding for Minor Digital Use Risk Screening
```

## 2. 为什么这个问题成立

现有原始数据包含四类信息：

- 视频；
- 心率相关记录；
- 问卷；
- 应用/使用信息。

现有标签也似乎是综合这些信息得到的。因此，只用视频作为模型输入会存在天然信息缺口。

但只用视频作为部署输入仍然合理，因为：

- 问卷不一定总能获得，且主观性和隐私性较强；
- 心率需要额外设备，通用性差；
- 应用日志依赖设备权限或平台接口，跨场景不可保证；
- 视频虽然敏感，但可以在本地处理，不公开原始视频，只上传隐私过滤后的文本证据。

所以本文应明确写成：

```text
video-only proxy evidence screening
```

而不是：

```text
video-based addiction diagnosis
```

## 3. 核心想法

提出 PriVTE：

```text
Privacy-preserving Video-to-Text Evidence Encoding
```

即：在本地把敏感视频转成结构化、质量感知、隐私过滤的文本证据，再让中心端 text-only LLM 或分类器做风险筛查判断。

流程：

```text
Raw Sensitive Video
  -> key-window selection
  -> ROI focusing
  -> proxy feature extraction
  -> quality estimation
  -> privacy filtering
  -> schema-first textual evidence
  -> text-only LLM / classifier
```

重点不是自由视频 caption，而是：

```text
structured visual features -> privacy filtering -> deterministic textual evidence
```

## 4. 关键技术

PriVTE 的关键技术不是训练一个新的大模型，而是构建一套可复现、可审计的视频证据编码流程。

主要模块包括：

1. **关键时间窗选择**

   从约 30 分钟长视频中选择有代表性的片段：

   - coverage windows：覆盖全程；
   - event windows：捕捉交互强度升高、持续注视、姿态变化、负向表情线索等；
   - quality windows：优先选择人脸、手部、设备可见度较好的片段；
   - baseline windows：用于个体内参考序列。

2. **ROI 聚焦**

   本地识别人脸、眼部/头部、手部、身体姿态、设备和屏幕区域。

   目的不是把图片交给 LLM，而是限制本地视觉分析范围，并辅助隐私过滤。

3. **视频代理特征提取**

   提取视频中可观察的代理证据，例如：

   - 屏幕注视比例；
   - 最大连续注视时长；
   - 点击/滑动频率；
   - 重复操作；
   - 头部前倾和姿态变化；
   - 眨眼频率变化；
   - 面部动作单元和负向表情趋势；
   - 疑似挫败、疲劳或高沉浸片段。

   这些只能作为 proxy evidence，不能直接解释为心理状态或诊断结论。

4. **质量估计**

   每类证据都附带质量信息，例如：

   - 人脸可见比例；
   - 手部可见比例；
   - 设备可见比例；
   - 光照质量；
   - 遮挡程度；
   - 多人干扰；
   - 有效观察时长。

   质量低时，模型应更倾向输出 `insufficient_evidence` 或触发人工复核。

5. **隐私过滤与粒度压缩**

   在文本生成前删除或压缩敏感信息：

   - 不输出 raw video、图像、音频；
   - 不输出 OCR/ASR 原文；
   - 不输出学校、班级、地址、精确时间；
   - 不输出外貌、服装、家庭环境描述；
   - 精确时间戳改为时间段；
   - 精确坐标改为姿态类别；
   - 精确心率不进入公开输入；
   - 高频序列转为聚合统计。

6. **Schema-first 文本证据生成**

   输出固定结构的 JSON 和模板化文本，例如：

   ```text
   global_features
   event_windows
   quality_summary
   limitations
   privacy_processing_summary
   ```

   LLM 只看到这些文本证据，不看到原始视频。

7. **结构化模型输出与评测**

   要求 LLM / classifier 输出：

   ```text
   risk_level
   confidence
   supporting_evidence
   missing_information
   needs_human_review
   ```

   评测不仅看分类准确率，还看校准性、证据一致性、缺失信息意识、人工复核触发和隐私泄露风险。

## 5. 主要贡献建议

本文更适合写成 benchmark / dataset / responsible AI workflow 论文，而不是新模型论文。

建议贡献为四点：

1. **问题定义**

   提出一个新的 AISI 问题：

   ```text
   privacy-preserving text-only risk screening from sensitive minor videos
   ```

2. **PriVTE 协议**

   提出一个 schema-first 的视频到文本证据编码协议，把敏感视频转成可审计文本证据。

3. **MDU-RiskText 数据集**

   构建一个由真实 field-collected cohort 派生的隐私过滤文本证据数据集。

4. **MDU-RiskBench 评测基准**

   构建 text-only LLM / classifier 评测基准，评估风险等级、置信度、证据使用、缺失信息、人类复核和隐私-效用权衡。

## 6. MDU-RiskText 和 MDU-RiskBench 的区别

```text
PriVTE = 方法/协议
MDU-RiskText = 数据集
MDU-RiskBench = 评测基准
```

具体来说：

- `MDU-RiskText` 是 PriVTE 生成的文本证据数据集；
- `MDU-RiskBench` 是围绕这个数据集定义的任务、prompt、metric、baseline 和评测脚本。

## 7. 未来应用形态

项目可以进一步解释为一种：

```text
边缘计算 + 中心大模型判断 + 隐私保护
```

应用流程：

```text
本地端:
  采集视频、保存身份映射、提取特征、隐私过滤、生成文本证据

中心端:
  只接收无显式身份字段的 evidence package
  由 text-only LLM 输出风险等级、证据、置信度和人工复核建议

本地端:
  将结果映射回具体个体，并由人工复核
```

严谨表述应为：

```text
中心端不接收 raw video、图像、音频、问卷原文、精确心率或身份映射；
系统降低中心端对个体的可链接性，但不声称完全匿名。
```

## 8. 关键实验

最重要的不是追求单一 SOTA，而是验证信息边界和隐私-效用权衡。

建议实验：

1. **Full-modality upper bound**

   使用问卷、心率、应用、视频等内部信息，估计完整信息下的上限。

2. **Video-only feature upper bound**

   只用视频提取的结构化特征，判断 video-only 代理证据是否有信号。

3. **Text evidence benchmark**

   比较 JSON、模板文本、JSON + 文本、free caption 等证据形式。

4. **LLM / classifier 对比**

   比较传统分类器、小型 LLM、强 LLM、rubric prompt、JSON constrained output。

5. **隐私-效用分析**

   比较不同隐私粒度下的性能、PII 泄露、字段唯一性和人工隐私审查结果。

## 9. 易混淆点

避免：

```text
我们用视频判断未成年人是否成瘾。
```

建议：

```text
我们基于视频中可观察代理证据进行风险筛查。
```

避免：

```text
LLM 是诊断器。
```

建议：

```text
LLM 是证据综合器，输出风险等级、依据、置信度、缺失信息和人工复核建议。
```

避免：

```text
文本化后数据已经匿名。
```

建议：

```text
文本化降低了 raw identity exposure，但仍需 PII 扫描、人工审查和重识别风险分析。
```

## 10. 一句话总结

```text
PriVTE 将原本难以公开的未成年人敏感视频筛查问题，
转化为一个隐私过滤后的 text-only benchmark，
使研究者可以在不接触原始视频的情况下评测 LLM 和分类器，
同时量化 video-only 代理证据的能力边界和隐私-效用权衡。
```
