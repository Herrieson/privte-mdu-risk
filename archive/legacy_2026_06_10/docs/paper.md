# PriVTE 相关论文递进式调研记录

本文档用于记录 PriVTE / MDU-RiskText / MDU-RiskBench 的相关工作调研。

项目目标不是做“视频诊断成瘾”，而是研究：

```text
video-only, privacy-preserving, text-only risk screening based on observable proxy evidence
```

因此文献调研需要同时覆盖：

1. 敏感长视频如何转成可审计的文本或结构化证据。
2. 为什么不直接走 free-form video captioning 或 multimodal LLM。
3. 如何设计 text-only LLM benchmark。
4. 如何做隐私保护数据发布、数据卡和评测治理。
5. 行为风险筛查、情绪/生理代理特征和质量控制的基础工作。

## 0. 本项目与现有工作的关系

PriVTE 的论文空间可以放在以下交叉点：

```text
sensitive minor videos
  + privacy-preserving evidence encoding
  + text-only model benchmark
  + risk screening with missing-information reporting
```

现有工作大致分为几类：

- 视频字幕/视频问答：通常直接从视频生成自然语言或回答问题，但不以隐私过滤和可发布文本证据为核心。
- 多模态 LLM benchmark：评测模型看图/看视频能力，但通常默认模型可以访问原始视觉输入。
- LLM benchmark：重视任务集合、指标、校准和鲁棒性，但输入多为文本题目，不处理敏感视频的前置证据编码。
- 数据治理/数据卡：强调数据集文档化、伦理和使用限制，但不直接给出视频到文本证据编码协议。
- 行为风险筛查：提供标签和领域背景，但常依赖问卷、访谈或长期行为数据，不适合直接声称由单段视频诊断。

PriVTE 的差异化贡献应写成：

```text
We do not benchmark video understanding itself.
We benchmark text-only risk screening under a privacy-preserving evidence encoding protocol.
```

## 0.1 2024-2026 LLM-era 主线文献

第一版文献中有不少基础工作，适合支撑背景和方法源流；但论文投稿时，主线应显式切到 2024-2026 的 LLM / MLLM benchmark、长视频理解、隐私泄露评测和高影响场景评测。2017-2021 的视频字幕、视频问答和早期 LLM benchmark 应降级为背景，不应成为 Related Work 的主体。

### 0.1.1 Long-video / video LLM benchmark 新工作

| 工作 | 年份 / 来源 | 核心内容 | PriVTE 应借鉴什么 | PriVTE 需要区分什么 |
| --- | --- | --- | --- | --- |
| MVBench | CVPR 2024, [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Li_MVBench_A_Comprehensive_Multi-modal_Video_Understanding_Benchmark_CVPR_2024_paper.html), [arXiv](https://arxiv.org/abs/2311.17005) | 用 20 个视频任务系统评测多模态模型的视频理解能力，强调单帧无法解决的时间推理。 | 可作为 PriVTE 的反向对照：我们不让模型看视频，而是把视频证据压成 text-only evidence。 | MVBench 评测 MLLM 的视觉理解，PriVTE 评测隐私约束下的文本证据综合。 |
| Video-MME | 2024, [arXiv](https://arxiv.org/abs/2405.21075) | 面向多模态 LLM 的综合视频分析 benchmark，覆盖不同视频时长和字幕设置。 | 可借鉴长视频、多设置、多模型报告方式。 | Video-MME 默认模型可访问视频；PriVTE 避免把敏感视频交给 MLLM。 |
| LongVideoBench | NeurIPS 2024 Datasets and Benchmarks, [arXiv](https://arxiv.org/abs/2407.15754), [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/329ad516cf7a6ac306f29882e9c77558-Paper-Datasets_and_Benchmarks_Track.pdf) | 面向长上下文视频-语言理解，视频最长可到约 1 小时，强调 interleaved video-language inputs。 | 对 PriVTE 的 30 分钟长视频很相关，可借鉴长视频采样、时间定位和 long-context evaluation。 | 它仍然让模型处理视觉/字幕输入；PriVTE 是本地视频处理后只发布 text evidence。 |
| MLVU | 2024, [arXiv](https://arxiv.org/abs/2406.04264) | Multi-task Long Video Understanding Benchmark，覆盖多视频类型、多任务和长视频退化分析。 | 可借鉴“长视频任务分层”和模型随视频长度性能下降的分析。 | PriVTE 不评测通用 long-video VLM 能力，而评测风险证据综合能力。 |
| LVBench | ICCV 2025, [CVF PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_LVBench_An_Extreme_Long_Video_Understanding_Benchmark_ICCV_2025_paper.pdf), [arXiv](https://arxiv.org/abs/2406.08035) | Extreme long video understanding benchmark。 | 支撑 PriVTE 的 key-window / event-window 设计：全视频直接输入并不现实。 | LVBench 仍以视觉理解为目标，PriVTE 以隐私保护文本证据为目标。 |
| TempCompass | ACL Findings 2024, [ACL PDF](https://aclanthology.org/2024.findings-acl.517.pdf), [arXiv](https://arxiv.org/abs/2403.00476) | 专门评估 Video LLM 的 temporal perception，指出当前 VLM/Video LLM 时序理解仍弱。 | 很适合支撑“不能直接相信 video LLM 对长视频事件的自由描述”，PriVTE 需要显式事件窗口和时序证据。 | PriVTE 应使用 deterministic event extraction + quality report，而不是让模型自由解释时序。 |
| EgoSchema | NeurIPS 2023 Datasets and Benchmarks, [arXiv](https://arxiv.org/abs/2308.09126) | 从 Ego4D 派生 very long-form video-language QA。 | 虽然是 2023，但属于 LLM-era 长视频 benchmark，可借鉴从敏感长视频派生文本 QA/label 的思路。 | EgoSchema 仍以问答评测为主，PriVTE 是风险筛查证据编码。 |

### 0.1.1A 新近 Video-LLM 处理方法

这些工作不一定是直接 baseline，但说明长视频处理的主流问题已经从“能否 caption”转向“如何选择、压缩、保留关键时序信息”。

| 工作 | 年份 / 来源 | 核心内容 | 对 PriVTE 的启发 |
| --- | --- | --- | --- |
| KeyVideoLLM | 2024, [arXiv](https://arxiv.org/abs/2407.03104) | 面向 VideoLLM 的关键帧选择。 | 支撑 PriVTE 的 key-window / key-frame selection，但 PriVTE 输出文本证据而非视觉 token。 |
| SlowFast-LLaVA | 2024, [arXiv](https://arxiv.org/abs/2407.15841) | 用 slow/fast pathway 同时保留空间细节和长期时序上下文。 | 可借鉴“低频全局 + 高频事件”的双通路思想。 |
| LongVU | 2024, [arXiv](https://arxiv.org/abs/2410.17434) | 用 spatiotemporal adaptive compression 压缩长视频 token。 | PriVTE 的对应思想是“把长视频压缩为隐私过滤后的结构化文本证据”。 |
| Adaptive Keyframe Sampling for Long Video Understanding | 2025, [arXiv](https://arxiv.org/abs/2502.21271) | 用 relevance + coverage 优化长视频关键帧选择。 | 可直接借鉴到 key-window extractor：同时覆盖全局和保留判断相关窗口。 |
| Video-LLaVA | EMNLP 2024, [ACL PDF](https://aclanthology.org/2024.emnlp-main.342.pdf), [arXiv](https://arxiv.org/abs/2311.10122) | 统一图像和视频表征以接入 LLM。 | 可作为“直接视觉接入 LLM”的技术对照；PriVTE 反其道而行，限制 LLM 只读文本证据。 |

### 0.1.2 新近 LLM benchmark 方法

| 工作 | 年份 / 来源 | 核心内容 | PriVTE 应借鉴什么 |
| --- | --- | --- | --- |
| LiveBench | ICLR 2025, [arXiv](https://arxiv.org/abs/2406.19314), [ICLR PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/e4a46394ba5378b3f9a186a5b4c650d1-Paper-Conference.pdf) | 动态、contamination-limited benchmark，使用近期数据源并尽量客观评分。 | MDU-RiskBench 应记录数据冻结时间、版本号和 split；若后续更新，应有 versioned benchmark。 |
| MMLU-Pro | NeurIPS 2024 Datasets and Benchmarks, [paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/ad236edc564f3e3156e1b2feafb99a24-Paper-Datasets_and_Benchmarks_Track.pdf), [arXiv](https://arxiv.org/abs/2406.01574) | 对 MMLU 做更难、更稳健的版本，扩展选项并降低 prompt sensitivity。 | PriVTE prompt 需要做 prompt sensitivity / rubric sensitivity 测试，不能只报单一 prompt。 |
| SimpleQA | 2024, [OpenAI page](https://openai.com/index/introducing-simpleqa/), [arXiv](https://arxiv.org/abs/2411.04368) | 关注 factuality、single indisputable answer、not attempted。 | 对 PriVTE 的 `insufficient_evidence` 很有启发：模型应该知道何时不判断。 |
| GPQA | 2023/2024, [arXiv](https://arxiv.org/abs/2311.12022) | 高难、专家级、Google-proof QA benchmark。 | 可借鉴“专家验证”和“非专家难以可靠判断”的设定；PriVTE 标签应说明人工标注/复核机制。 |
| MMMU-Pro | 2024, [arXiv](https://arxiv.org/abs/2409.02813) | 更鲁棒的多学科多模态 benchmark。 | 可借鉴减少 shortcut、构造更难选项和报告 multimodal/text-only 差异。 |
| Abstain-QA / Do LLMs Know When to NOT Answer? | 2024, [arXiv](https://arxiv.org/abs/2407.16221) | 评估 LLM 面对不可答问题时的 abstention ability。 | MDU-RiskBench 应把 `insufficient_evidence`、低置信度和 human review 作为正式指标。 |
| Know Your Limits: A Survey of Abstention in LLMs | 2024, [arXiv](https://arxiv.org/abs/2407.18418) | 综述 LLM abstention，从 query、model、human values 角度组织。 | 可支撑 Discussion：风险筛查中“拒答/证据不足”不是失败，而是安全机制。 |

### 0.1.2A 高影响场景与安全评测

| 工作 | 年份 / 来源 | 核心内容 | 对 PriVTE 的启发 |
| --- | --- | --- | --- |
| HealthBench | OpenAI 2025, [arXiv](https://arxiv.org/abs/2505.08775), [OpenAI](https://openai.com/index/healthbench/) | 面向真实健康对话的 LLM benchmark，使用专家 rubric，强调 meaningful、trustworthy、unsaturated。 | PriVTE 可借鉴专家 rubric、现实场景、低置信度和缺失信息处理；但应避免医疗诊断化。 |
| MedSafetyBench | NeurIPS 2024 Datasets and Benchmarks, [paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/3ac952d0264ef7a505393868a70a46b6-Paper-Datasets_and_Benchmarks_Track.pdf), [arXiv](https://arxiv.org/abs/2403.03744) | 评估医学 LLM 的 safety。 | PriVTE 需要同样报告禁止用途、人工复核、错误风险和安全边界。 |
| MedBench | 2024, [arXiv](https://arxiv.org/abs/2407.10990) | 中文医疗大模型 benchmark。 | 对中文 benchmark、中文标签和中文 prompt 设计有参考价值。 |

### 0.1.3 近年隐私视频、匿名化和 privacy-utility 工作

| 工作 | 年份 / 来源 | 核心内容 | 对 PriVTE 的启发 |
| --- | --- | --- | --- |
| STPrivacy: Spatio-Temporal Privacy-Preserving Action Recognition | ICCV 2023, [CVF PDF](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_STPrivacy_Spatio-Temporal_Privacy-Preserving_Action_Recognition_ICCV_2023_paper.pdf) | 通过时空机制平衡动作识别和隐私。 | 支撑 privacy-utility tradeoff 的实验设计。 |
| Privacy-Preserving Optics for Enhancing Protection in Face De-Identification | CVPR 2024, [CVF PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Lopez_Privacy-Preserving_Optics_for_Enhancing_Protection_in_Face_De-Identification_CVPR_2024_paper.pdf), [arXiv](https://arxiv.org/abs/2404.00777) | 从成像/optics 层面增强 face de-identification。 | 说明“只靠后处理文本脱敏”不是唯一隐私路径；但 PriVTE 当前选择不发布图像。 |
| Disguise without Disruption | AAAI 2024, [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/27851) | Utility-preserving face de-identification。 | 可作为“为什么不发布匿名化人脸图像”的对照：即使匿名图仍需证明 privacy-utility。 |
| Selective, Interpretable, and Motion Consistent Privacy Attribute Obfuscation for Action Recognition | 2024, [arXiv](https://arxiv.org/abs/2403.12710) | 针对动作识别做选择性、可解释、运动一致的隐私属性遮蔽。 | PriVTE 可借鉴“保留任务相关动态，移除身份相关静态属性”的思想。 |
| Less Static, More Private: Transferable Privacy-Preserving Action Recognition | ICCV 2025, [CVF PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Xia_Less_Static_More_Private_Towards_Transferable_Privacy-Preserving_Action_Recognition_by_ICCV_2025_paper.pdf) | 将静态外观和动态动作解耦，移除隐私敏感静态信息。 | 非常贴近 PriVTE：我们也应强调“不输出外貌/服装/场景静态身份线索，只保留聚合行为证据”。 |
| Privacy Beyond Pixels: Latent Anonymization for Privacy-Preserving Video Understanding | 2025, [arXiv](https://arxiv.org/abs/2511.08666) | 在 video foundation model 的 latent space 做匿名化。 | 支撑“即使不发布像素，高维特征仍可能泄露隐私”；PriVTE 不应发布 face embedding 或高维 skeleton。 |
| Face Anonymization Made Simple | 2024, [arXiv](https://arxiv.org/abs/2411.00762) | diffusion-based face anonymization。 | 可作为匿名化图像路线的背景，但 PriVTE 的公开层仍应避免图像。 |

### 0.1.3A 隐私、PII 与 benchmark 泄露

| 工作 | 年份 / 来源 | 核心内容 | 对 PriVTE 的启发 |
| --- | --- | --- | --- |
| Multi-P2A | 2024, [arXiv](https://arxiv.org/abs/2412.19496) | 从多视角评估大视觉语言模型的隐私意识和隐私泄露风险。 | 直接支撑 MDU-RiskBench 的 privacy leakage audit。 |
| PII-Scope | 2024, [arXiv](https://arxiv.org/abs/2410.06704), [OpenReview PDF](https://openreview.net/pdf?id=O5r4VqQV6P) | 评估 LLM 训练数据 PII 泄露。 | PriVTE 不只是清洗文本，还要评估模型输出是否重新暴露隐私。 |
| MLLMU-Bench | NAACL 2025, [ACL PDF](https://aclanthology.org/2025.naacl-long.207.pdf), [arXiv](https://arxiv.org/abs/2410.22108) | 面向多模态大模型隐私保护和 unlearning 的 benchmark。 | 可作为“多模态模型会记忆/泄露隐私”的近年证据。 |
| Benchmarking Benchmark Leakage | 2024, [arXiv](https://arxiv.org/abs/2404.18824) | 研究 LLM benchmark 泄露问题并提出 benchmark transparency card。 | MDU-RiskBench 应写明数据发布时间、公开范围、split 和可能污染风险。 |
| PrivAuditor | NeurIPS 2024 Datasets and Benchmarks, [paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/12b18a15dcd73e1991e9959a94375fab-Paper-Datasets_and_Benchmarks_Track.pdf) | benchmark privacy vulnerabilities in adapted/fine-tuned LLMs。 | 支撑对 LLM baseline 的隐私泄露测试。 |

### 0.1.4 数据集元数据和治理新规范

| 工作 | 年份 / 来源 | 核心内容 | 对 PriVTE 的启发 |
| --- | --- | --- | --- |
| Croissant: A Metadata Format for ML-Ready Datasets | NeurIPS 2024 Datasets and Benchmarks, [arXiv](https://arxiv.org/abs/2403.19546), [NeurIPS PDF](https://papers.neurips.cc/paper_files/paper/2024/file/9547b09b722f2948ff3ddb5d86002bc0-Paper-Datasets_and_Benchmarks_Track.pdf) | 面向 ML-ready datasets 的机器可读元数据格式。 | MDU-RiskText 除 dataset card 外，可考虑提供 Croissant metadata。 |
| Croissant-RAI | 2024, [arXiv](https://arxiv.org/abs/2407.16883) | 面向 Responsible AI 的机器可读数据集文档格式。 | 适合 Public Lite / Controlled Research 的使用限制、敏感字段和风险声明。 |
| Croissant 1.1 specification | 2026, [MLCommons spec](https://docs.mlcommons.org/croissant/docs/croissant-spec-1.1.html) | 增强 provenance、usage policies、复杂多维数据建模。 | 对 PriVTE 的数据谱系、授权、split、版本和 release tier 很有价值。 |
| Navigating Dataset Documentations in AI | ICLR 2024, [ICLR](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8c67fc501a50977947c5bebbc39ca8f6-Abstract-Conference.html) | 大规模分析 HuggingFace dataset cards 的完成度和实践问题。 | 说明只写 README 不够，dataset card 需要完整、结构化、可审计。 |

### 0.1.5 领域背景的新近综述

| 工作 | 年份 / 来源 | 核心内容 | 对 PriVTE 的启发 |
| --- | --- | --- | --- |
| Prevalence of Internet Gaming Disorder among Chinese adolescents: A systematic review and meta-analysis | 2024, [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1876201824003502) | 中国青少年 IGD 患病率系统综述和 meta-analysis。 | 适合 Introduction 中说明中国语境下该问题的重要性，但不能替代本项目标签定义。 |
| Gaming disorder: systematic review and meta-analytic structural equation modeling | 2024, [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0747563224002164) | 综合 affective、cognitive、executive functioning 与 gaming disorder 的关系。 | 支撑“单段视频只能提供代理证据，不能完整覆盖长期功能损害”。 |
| Trends and Influencing Factors in Problematic Smartphone Use Prevalence | 2024, [SAGE](https://journals.sagepub.com/doi/full/10.1089/cyber.2023.0548) | Problematic smartphone use prevalence 的系统综述和 meta-analysis。 | 可作为未成年人数字设备风险背景。 |
| Perceived stress and mobile phone addiction in adolescents: meta-analysis | 2025, [BMC Psychology](https://link.springer.com/article/10.1186/s40359-025-03702-z) | 压力与青少年手机成瘾关系的 meta-analysis。 | 支撑讨论：风险筛查需要心理/社会背景，video-only 必然不完整。 |
| Youth privacy concerns in AI systems: systematic review | 2024, [arXiv](https://arxiv.org/abs/2412.16369) | 青少年对 AI 系统隐私担忧的系统综述。 | 可用于 Ethics / Social Impact：未成年人 AI 数据治理不仅是技术脱敏问题。 |
| Problematic gaming, psychiatric comorbidities, and adolescence | 2024, [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0306460324001400) | 青少年 problematic gaming 与精神共病的系统综述。 | 作为社会问题背景；强调不能用单段视频做诊断。 |
| Digital media use and adolescent mental health qualitative systematic review | 2026, [BMC Digital Health](https://link.springer.com/article/10.1186/s44247-026-00238-z) | 数字媒体使用与青少年心理健康的定性系统综述。 | 支撑 AISI 社会影响叙事，但要避免因果夸大。 |
| Digital Phenotyping for Adolescent Mental Health | 2025, [arXiv](https://arxiv.org/abs/2501.08851) | 用主动/被动手机数据预测青少年心理健康风险。 | 可作为“数字行为代理特征用于风险预测”的近年对照。 |

### 0.1.6 这一轮对论文定位的修正

新增文献后，Related Work 的重心应调整为：

```text
Old foundational works explain where the components come from.
Recent works explain why PriVTE is timely:
  - video LMMs still struggle with long temporal understanding;
  - new benchmarks emphasize contamination control and abstention;
  - privacy-preserving video analytics still trades utility against leakage;
  - machine-readable dataset governance is becoming a benchmark expectation.
```

因此 PriVTE 的新近性可以写成：

```text
At a time when video LMMs and long-video benchmarks are rapidly expanding,
PriVTE takes a different route: it removes raw visual access by design and
evaluates whether structured, privacy-filtered textual evidence can support
responsible text-only screening.
```

## 0.2 AAAI AISI 与 benchmark 论文叙事方式

AAAI AISI 不是单纯看算法指标。AAAI-25 AISI CFP 明确强调，社会影响 track 会按不同于 main track 的标准审稿，重点包括：

- problem significance
- engagement with literature, including non-CS literature
- novelty of approach
- justification of approach
- quality of evaluation
- facilitation of follow-up work
- scope and promise for social impact

来源：[AAAI-25 AISI CFP](https://aaai.org/conference/aaai/aaai-25/aisi-call/)

### 0.2.1 AISI review criteria 到 PriVTE 的映射

| AISI criteria | PriVTE 应怎样回应 |
| --- | --- |
| Significance of the problem | 未成年人数字设备过度使用风险是现实社会问题；原始视频高度敏感，常规公开视频 benchmark 不可行。 |
| Engagement with literature | 不能只引用 CV/LLM 文献，还要引用 adolescent digital use、privacy governance、dataset documentation、human review / abstention。 |
| Novelty of approach | PriVTE 不是新模型，而是 privacy-preserving video-to-text evidence encoding protocol + text-only benchmark。 |
| Justification of approach | 必须解释为什么不用 raw video、为什么不用 free caption、为什么不用直接 MLLM、为什么 public text 仍要审计。 |
| Quality of evaluation | 需要真实数据、LLM baseline、schema ablation、privacy-utility、high-risk recall、calibration、human review trigger。 |
| Facilitation of follow-up work | Public Lite schema、prompt、split、eval script、dataset card、privacy audit script，比“只说不能公开数据”强。 |
| Scope and promise | 辅助筛查、研究基准、人工复核；不诊断、不惩戒、不画像。 |

### 0.2.2 AAAI AISI 中值得学习的 dataset / benchmark 论文

| 工作 | 来源 | 论文故事线 | PriVTE 可借鉴 |
| --- | --- | --- | --- |
| Bridging the Gap: Enhancing LLM Performance for Low-Resource African Languages with New Benchmarks, Fine-Tuning, and Cultural Adjustments | AAAI 2025 AISI, [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/34996) | 从语言技术不平等出发，构建新 benchmark，量化 SOTA LLM 差距，再用 fine-tuning / cultural adjustment 缩小差距。 | PriVTE 可采用“现实群体被现有 benchmark 忽视 -> 构建新 benchmark -> 量化模型差距 -> 提供改进路径”的写法。 |
| Multi-OphthaLingua | AAAI 2025 AISI, [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/35053) | 医疗 LMIC 场景 + 多语言 QA benchmark + 语言偏差评估 + debiasing 方法。 | 可借鉴“benchmark 不只是数据集，还要揭示不公平/部署风险”。 |
| MedAlign | AAAI 2024 AISI, [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/30205) | 真实 EHR 任务难以用传统 QA 表达；用临床专家生成 instruction benchmark，并通过专家评价 LLM 输出。 | PriVTE 可借鉴专家/人工标注、真实场景任务、受控数据使用协议。 |
| LLeQA | AAAI 2024 AISI, [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/30232) | 法律援助社会问题 + expert-annotated long-form QA dataset + RAG pipeline + 公开代码/数据/模型。 | 可借鉴“领域专家标注 + interpretable long-form answer + release supports follow-up work”。 |
| CUPCase | AAAI 2025 AISI, [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/35050) | 现有医疗 benchmark 偏考试题，缺少真实复杂病例；构造真实 case-report benchmark，并比较 open-ended / multiple-choice。 | PriVTE 可写“现有 video/LLM benchmark 不覆盖敏感未成年人场景，真实数据有独特价值”。 |
| Trustworthy and Practical AI for Healthcare: Guided Deferral with LLMs | AAAI 2025 AISI, [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/35063) | 高风险医疗场景中 LLM 会幻觉且隐私严格；提出 guided deferral system，把不确定样本转给人。 | 直接借鉴 human-review flag、low-confidence handling、imbalanced calibration。 |
| Enhancing Privacy in the Early Detection of Sexual Predators Through FL and DP | AAAI 2025 AISI, [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/35005) | 儿童线上安全 + 私密对话不能集中训练；用 privacy-preserving pipeline 并讨论 privacy-utility。 | 与 PriVTE 很接近：未成年人 + 风险筛查 + 隐私保护 + 真实数据评估。 |
| Pioneering Explainable Video Fact-Checking with a New Dataset and Multi-role Multimodal Model Approach | AAAI 2025 AISI, [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/35048) | 现有视频事实核查缺少证据和解释；新数据集提供 veracity labels、rationales、supporting evidence。 | 直接启发 PriVTE：风险标签要配 supporting evidence，而不只是分类。 |
| BirdCollect | AAAI 2024 AISI, [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/30189) | 生态保护问题 + 真实野外数据 + dense annotation + benchmark tasks。 | 可借鉴“难采集真实数据本身是贡献”，但要说明 PriVTE 的敏感数据不能 raw release。 |
| DroughtSet | AAAI 2025 AISI, [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/35066) | 气候灾害问题 + 多源数据集 + benchmark + 可解释 spatio-temporal model。 | 可借鉴“dataset + benchmark + domain insight”三段贡献。 |
| CityPulse | AAAI 2024 AISI, [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/30216) | 城市变化难以细粒度测量；构建大规模街景时间序列数据并 city-wide implementation。 | 可借鉴“真实世界观测代理”写法：视频是风险筛查代理证据，不是完整真相。 |
| Finding epsilon and delta of Traditional Disclosure Control Systems | AAAI 2024 AISI, [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/30204) | 常用 disclosure control 机制未必真的保护隐私；用 DP 视角重新评估 privacy / accuracy / fairness。 | 支撑 PriVTE 的“文本化不等于匿名，必须做隐私评估”。 |

### 0.2.3 Benchmark 论文常见故事模板

从 AISI 和近年 benchmark 论文看，比较稳的故事结构是：

```text
1. Existing AI progress creates an opportunity, but current benchmarks miss a real social setting.
2. The real setting has constraints that make standard data/model assumptions invalid.
3. We collect/derive a benchmark under these constraints.
4. We define tasks, labels, splits, metrics, and release policy.
5. We evaluate representative models and show gaps, not just wins.
6. We analyze failure modes, fairness/privacy/robustness, and human review.
7. We release artifacts that enable follow-up work while respecting constraints.
```

对应到 PriVTE：

```text
1. Video LMMs and LLM benchmarks are rapidly advancing.
2. But sensitive minor videos cannot be released or sent to general MLLMs.
3. We introduce PriVTE to encode raw videos into structured, privacy-filtered text evidence.
4. We release MDU-RiskText Public Lite and define MDU-RiskBench.
5. We compare text-only LLM settings under fixed prompts and JSON outputs.
6. We evaluate schema ablations, privacy leakage, calibration, and human-review triggers.
7. We release schema, prompt, split, eval scripts, dataset card, and privacy audit protocol.
```

### 0.2.4 AISI 论文写法对 PriVTE 的提醒

不要把贡献写成：

```text
We build a better video classification system for addiction.
```

更稳的写法是：

```text
We introduce a privacy-preserving evidence encoding protocol and benchmark for text-only risk screening from sensitive minor videos.
```

不要只说：

```text
We cannot release raw data due to privacy.
```

要写成：

```text
Because raw videos cannot be released, the release mechanism itself is a technical and social contribution: schema-first evidence, release tiers, privacy audit, and benchmark scripts.
```

不要只报：

```text
LLM accuracy is X.
```

要报：

```text
Macro F1, high-risk recall, calibration, insufficient-evidence rate, human-review trigger rate, PII leakage rate, uniqueness risk, and privacy-utility curve.
```

### 0.2.5 PriVTE 可以借鉴的 Introduction 叙事骨架

```text
Paragraph 1: Social problem.
Minor digital device overuse is a growing concern, but responsible screening requires privacy, caution, and human review.

Paragraph 2: Data dilemma.
Videos contain observable behavioral proxy evidence, but raw minor videos are highly sensitive and cannot be publicly released or directly sent to general-purpose MLLMs.

Paragraph 3: Benchmark gap.
Recent video LMM benchmarks evaluate visual understanding, while LLM benchmarks evaluate text reasoning; neither addresses privacy-preserving text-only screening from sensitive videos.

Paragraph 4: Our approach.
We propose PriVTE, a schema-first video-to-text evidence encoding protocol that extracts key windows, proxy features, quality reports, and privacy-filtered textual evidence.

Paragraph 5: Dataset and benchmark.
We construct MDU-RiskText and MDU-RiskBench from a field-collected cohort in Ordos, Inner Mongolia, with release tiers and privacy audits.

Paragraph 6: Evaluation and boundaries.
We evaluate text-only LLMs, ablate evidence schemas, quantify privacy-utility tradeoffs, and require missing-information and human-review outputs. The system is for auxiliary screening, not diagnosis.
```

## 0.3 当前技术思路是否被文献支撑？

结论需要分两层：

```text
被较强支撑:
  在敏感长视频场景下，不直接公开 raw video、不直接让通用 MLLM 处理 raw video，
  而是构造受控、可审计、可发布的 evidence representation。

尚未被直接证明:
  将未成年人电子产品使用视频转成 schema-first text evidence 后，
  是否足以支持可靠的风险筛查。
```

因此 PriVTE 当前思路可以作为研究假设和 benchmark 设计方向，但不能在论文中预设它一定优于其他路线。我们需要用实验回答。

### 0.3.1 哪些文献支持这个方向？

| 支撑点 | 代表文献 | 对 PriVTE 的含义 |
| --- | --- | --- |
| Long-video MLLM 仍存在时序理解和长上下文瓶颈 | TempCompass、LongVideoBench、MLVU、LVBench | 直接把 30 分钟敏感视频丢给 MLLM 不是稳健基线；key-window 和 evidence compression 有必要。 |
| Video-LLM 方法本身也在做 frame/window selection 和 token compression | KeyVideoLLM、SlowFast-LLaVA、LongVU、Adaptive Keyframe Sampling | PriVTE 的“关键窗口 + 全局摘要 + 事件窗口”符合视频领域当前趋势，只是输出从 visual token 变成 text evidence。 |
| LVLM/LLM 存在隐私泄露和 PII 风险 | Multi-P2A、PII-Scope、MLLMU-Bench、PrivAuditor | 不把 raw minor video 交给通用模型是合理设计；模型输出本身也要 privacy audit。 |
| AISI / benchmark 论文重视真实约束下的数据发布机制 | AAAI AISI CFP、MedAlign、LLeQA、BirdCollect、DroughtSet | “不能公开 raw data，因此构造可发布 benchmark”本身可以成为贡献，不是弱点。 |
| 高风险场景 benchmark 强调 rubric、abstention、human review | HealthBench、MedSafetyBench、Guided Deferral、Abstain-QA | PriVTE 应把 `insufficient_evidence` 和 human-review flag 作为正式输出，而不是只追求分类。 |
| Benchmark 设计需要明确测量对象和证据链 | ECBD: Evidence-Centered Benchmark Design for NLP | MDU-RiskBench 必须写清楚：我们测的是 text-only evidence synthesis，不是诊断能力，也不是完整心理评估能力。 |

### 0.3.2 哪些地方还没有被文献直接支撑？

| 不确定点 | 风险 | 需要怎样验证 |
| --- | --- | --- |
| Video-only proxy evidence 与风险标签是否相关 | 标签来自问卷、心率、应用、人工综合；视频可能只能解释一部分。 | 做 full-modality vs video-only vs text-evidence 的上限实验。 |
| Schema-first text 是否保留足够信息 | 文本压缩可能丢掉细粒度时序和视觉线索。 | 对比 internal structured features、schema text、free caption、direct video/model upper bound。 |
| 视觉特征提取是否可靠 | gaze、手部动作、表情、rPPG 在遮挡/低光/运动下噪声大。 | 每类特征必须带 quality score；低质量样本触发 insufficient_evidence。 |
| 情绪/心理代理特征是否会过度推断 | 面部动作不等于心理状态，更不等于成瘾。 | 文本只写 observable proxy，不写 anxiety/addiction/self-control 等诊断性推断。 |
| 文本证据是否仍会泄露隐私 | 罕见事件、路径、时间、场景、应用名可能重识别。 | PII scan、manual audit、字段唯一性、rare-combination risk。 |
| clip-level 标签能否聚合为 person-level 标签 | 当前 `person_01` 只有 22/32 个 clip 匹配标签，且有否/轻度/中度混合。 | 先冻结任务粒度：clip-level benchmark、person-level benchmark，或两者都做。 |

### 0.3.3 可选技术路线比较

| 路线 | 做法 | 优点 | 风险 | 建议定位 |
| --- | --- | --- | --- | --- |
| A. Raw video -> MLLM -> risk | 直接让 video LMM 看视频并输出风险 | 实现简单，可作为内部 upper bound | 隐私风险最大；长视频时序理解不稳；不可公开复现 | 只能做 internal upper-bound，不做主线 |
| B. Video -> free caption -> LLM | 先 caption，再分类 | 容易快速跑通 | caption 可能泄露外貌/场景/学校/屏幕内容；模型风格不稳定 | 做 ablation / negative baseline |
| C. Video -> structured features -> internal scorer | 不生成文本，直接使用内部结构化特征估计任务上限 | 可能性能更高 | 公开性差；高维特征可能泄露身份；解释性弱 | 只能做 internal feature upper-bound，不做主线 |
| D. Video -> schema-first evidence -> text-only LLM | 本地抽取代理特征，隐私过滤后生成 JSON + 模板文本 | 最符合 AISI、privacy release、benchmark 目标 | 信息损失和特征可靠性需要验证 | 建议作为 PriVTE 主线 |
| E. Hybrid: structured evidence + selected internal model scores | 文本证据加内部模型置信度/质量分 | 可提升稳定性 | 容易变成黑箱分数，削弱 evidence 贡献 | 可做增强版，不做第一主张 |

当前最合理的写法不是：

```text
Schema-first video-to-text must be the best technical solution.
```

而是：

```text
Schema-first video-to-text is the most responsible release and benchmark design under minor-video privacy constraints. Its utility must be empirically measured against stronger internal but less releasable alternatives.
```

### 0.3.4 最小验证闭环

为了判断这个思路是否真的成立，建议按以下递进实验做，而不是一开始就投入完整 CV pipeline：

#### Step 1: 标签和任务粒度审计

- 明确 `label.xlsx` 是 clip-level 还是 person-level。
- 明确 person-level label 聚合规则。
- 按 participant split，避免同一人不同 clip 泄漏到 train/test。

#### Step 2: 原始非视觉上限

内部使用问卷、应用类型、心率、标签置信度，训练简单模型。

目的：

```text
估计完整标注来源下任务本身的可预测性。
```

#### Step 3: Video-only internal feature upper-bound

先不生成文本，提取可验证的低级/中级视频特征：

- 视频质量
- 人脸/手部/设备可见度
- 姿态变化
- 操作强度 proxy
- 眨眼/头姿/AU proxy
- 事件窗口统计

目的：

```text
判断 video-only proxy evidence 是否有信号。
```

#### Step 4: Text evidence conversion

把 Step 3 的结构化特征转换为：

- JSON only
- templated text only
- JSON + templated text
- free caption baseline

目的：

```text
判断 schema-first text 是否保留足够效用，并减少隐私泄露。
```

#### Step 5: LLM benchmark

比较：

- small open-source LLM
- strong closed-source LLM
- rubric-based prompt
- JSON-constrained output

指标：

- Macro F1
- high-risk recall
- calibration
- insufficient_evidence rate
- human-review trigger rate
- privacy leakage rate

#### Step 6: Go / no-go 判断

| 结果 | 解释 | 论文写法 |
| --- | --- | --- |
| text evidence 接近 video-only feature upper-bound，且隐私泄露低 | PriVTE 主张很强 | 强调 schema-first evidence 有效且可发布 |
| text evidence 有效但明显弱于 internal features | PriVTE 仍成立 | 写成 privacy-utility tradeoff |
| video-only 本身很弱 | 不是失败 | 写成 video-only information ceiling，强调 human review 和 missing information |
| free caption 性能高但隐私泄露高 | 支撑 PriVTE | 写成 schema-first 在隐私和复现上更可靠 |
| LLM 输出不稳定 | 支撑 benchmark 价值 | 强调 fixed rubric、JSON output、calibration 和 abstention |

### 0.3.5 当前判断

基于目前调研，PriVTE 的大方向是被支撑的，但应收缩主张：

```text
Supported:
  privacy-preserving evidence release and text-only benchmark is a defensible AISI contribution.

Not yet proven:
  video-derived text evidence is sufficient for robust individual-level risk screening.
```

因此论文应避免强结论：

```text
PriVTE accurately detects digital addiction from video.
```

应写成：

```text
PriVTE enables reproducible evaluation of text-only risk screening from privacy-filtered video evidence, while quantifying the limits of video-only proxy signals and requiring human review for uncertain cases.
```

## 0.4 未来应用形态：边缘计算 + 中心判断 + 隐私保护

PriVTE 的未来应用价值可以描述为一种 split-computation architecture：

```text
raw video and identity remain at the edge;
privacy-filtered evidence is sent to the center;
the central LLM performs evidence synthesis only;
the final linkage to the individual and human review happen locally.
```

更严谨的中文表述：

```text
本地端负责视频采集、原始数据隔离保存、视觉特征提取、隐私过滤和文本证据生成；
中心端只接收不含显式身份字段的结构化文本证据，并由大模型进行风险证据综合；
中心端不保存 raw video、图像、音频、问卷原文、心率精确序列或身份映射；
本地端保存请求 ID 与个体身份之间的映射，并负责结果回填和人工复核。
```

### 0.4.1 推荐系统流程

```text
Edge / local site
  1. raw video capture
  2. local isolated storage
  3. PriVTE local preprocessing
     - key-window extraction
     - ROI focusing
     - proxy feature extraction
     - quality estimation
     - privacy filtering
     - schema-first text generation
  4. local PII / privacy audit
  5. generate ephemeral request_id
  6. send evidence package to central model service

Central model service
  7. receive evidence package only
  8. run text-only LLM
  9. output risk_level, confidence, supporting_evidence,
     missing_information, needs_human_review
  10. return result to edge

Edge / local site
  11. map request_id back to local subject
  12. local human review
  13. local intervention / counseling / follow-up if appropriate
```

### 0.4.2 中心端“不知道是谁”需要哪些条件？

不能简单写：

```text
The server does not know whose data it is.
```

更严谨的说法是：

```text
The server receives no explicit identity-bearing fields and no local identity mapping.
Under the stated threat model, the system is designed to reduce linkability between evidence packages and individual minors.
```

为了支持这个声明，至少需要：

- 不上传姓名、学校、班级、身份证、手机号、IP 原文、精确采集时间、原始路径。
- 不上传 raw video、frames、audio、OCR/ASR 原文、face embeddings、精确心率、精确坐标。
- 使用随机、短期、不可语义解释的 `request_id`。
- 本地保存 `request_id -> subject_id` 映射，中心端不保存。
- 中心端日志只保存最小必要字段，且有保留期限。
- 请求最好经过机构级代理或批处理，减少网络元数据直接暴露个体/设备。
- 对发送的 evidence package 做 PII scan 和 rare-combination risk check。
- 中心端只返回结构化判断，不返回推断性身份信息。

### 0.4.3 威胁模型要写清楚

建议把系统威胁模型写成：

```text
Edge is trusted to access raw data under local authorization.
Central service is honest-but-curious: it follows the protocol but should not receive identity-bearing raw data.
Network attackers are mitigated by transport encryption.
Researchers and model developers should only access public-lite or controlled evidence, not raw videos.
```

需要承认的残余风险：

- 文本证据可能包含罕见行为组合，仍有重识别风险。
- 中心端可能通过网络元数据、时间、机构、设备信息推断来源。
- 如果多次请求使用稳定 ID，中心端可做 longitudinal linkage。
- LLM 输出可能复述或放大输入中的隐私线索。
- 本地端如果管理不当，raw video 和身份映射仍然是高风险资产。

### 0.4.4 论文中可以怎样使用这个应用设想？

这个架构适合放在 Discussion 或 Deployment Workflow，而不是作为已经完整实现的主贡献。

建议写成：

```text
PriVTE supports a split edge-center workflow: raw sensitive videos remain local,
while the central model service receives only privacy-filtered textual evidence
and returns structured risk-screening outputs for local human review.
```

不要写成：

```text
PriVTE fully anonymizes minors.
```

更稳的写法：

```text
PriVTE reduces raw identity exposure through local processing, schema-constrained evidence,
field-level privacy filtering, and separation of identity mapping from central inference.
Residual linkage risks are assessed through PII scanning, manual audit, and uniqueness analysis.
```

### 0.4.5 这对论文贡献的影响

这个应用形态进一步说明：PriVTE 不只是离线数据处理脚本，而是一个负责任 AI workflow 的基础组件。

对应贡献可以写成：

```text
Beyond dataset construction, PriVTE defines a deployable edge-center evidence workflow
in which raw videos and identity mappings remain local, while text-only central models
perform auditable evidence synthesis under privacy constraints.
```

但主贡献仍应是：

```text
benchmark + evidence encoding protocol + privacy-aware release framework
```

而不是：

```text
a deployed production system
```

## 0.5 围绕 PriVTE 方法主线的补充文献

这一节按 PriVTE 模块来组织文献。目标不是堆列表，而是判断每个模块在视频/隐私/LLM 文献中是否有支撑。

### 0.5.1 M1 Key-window / keyframe / clip selection

PriVTE 的 key-window extractor 可以借鉴长视频 MLLM 中的 keyframe sampling 和 token compression，但输出不是视觉 token，而是隐私过滤后的 evidence。

| 工作 | 来源 | 对 PriVTE 的意义 |
| --- | --- | --- |
| KeyVideoLLM | 2024, [arXiv](https://arxiv.org/abs/2407.03104) | 说明长视频理解需要选择关键帧；PriVTE 可把关键帧选择升级为关键时间窗选择。 |
| Adaptive Keyframe Sampling for Long Video Understanding | CVPR 2025, [CVF PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Tang_Adaptive_Keyframe_Sampling_for_Long_Video_Understanding_CVPR_2025_paper.pdf), [arXiv](https://arxiv.org/abs/2502.21271) | 强调 relevance + coverage；PriVTE 也需要同时覆盖全局和异常/高信息事件。 |
| Query-Conditioned Evidential Keyframe Sampling | 2026, [arXiv](https://arxiv.org/abs/2604.01002) | 强调针对具体问题选择证据帧；PriVTE 可将“风险筛查 rubric”作为 query 条件。 |
| From Frames to Clips: Adaptive Key Clip Selection | 2025, [arXiv](https://arxiv.org/abs/2510.02262) | 从 frame selection 转向 clip selection，更贴近 PriVTE 的 window-level evidence。 |
| LongVU | 2024, [arXiv](https://arxiv.org/abs/2410.17434) | 用 spatiotemporal adaptive compression 处理长视频；PriVTE 的文本证据是另一种 privacy-aware compression。 |
| Scaling Up Video Summarization Pretraining with LLMs | CVPR 2024, [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Argaw_Scaling_Up_Video_Summarization_Pretraining_with_Large_Language_Models_CVPR_2024_paper.html) | 支撑长视频摘要和专业标注摘要的价值，但 PriVTE 应避免自由摘要泄露隐私。 |
| LVSum | 2026, [arXiv](https://arxiv.org/abs/2604.10024) | 关注 timestamp-aware long video summarization；PriVTE 也需要 coarse time bins 和事件定位。 |

对 PriVTE 的设计启发：

```text
不要只做 uniform sampling。
应组合 coverage windows + event windows + quality windows + baseline windows。
```

### 0.5.2 M2 Temporal grounding / event boundary / evidence localization

PriVTE 的关键事件不应只是“全局统计”，还需要时间窗口和事件证据。

| 工作 | 来源 | 对 PriVTE 的意义 |
| --- | --- | --- |
| Grounded-VideoLLM | 2024, [arXiv](https://arxiv.org/abs/2410.03290) | Video-LLM 对 fine-grained temporal grounding 仍弱，因此 PriVTE 不应依赖自由视频描述，应显式输出事件窗口。 |
| Training-free Video Temporal Grounding | ECCV 2024, [ECCV PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10687.pdf), [arXiv](https://arxiv.org/abs/2408.16219) | 支撑用预训练模型做 temporal grounding，但需质量评估。 |
| HawkEye | 2024, [arXiv](https://arxiv.org/abs/2403.10228) | 构造 segment-level caption/negative spans，强调视频文本时间对齐。 |
| TE-TAD | CVPR 2024, [CVPR poster](https://cvpr.thecvf.com/virtual/2024/poster/31123) | temporal action detection 的时间坐标表达和自适应 query 对不同长度视频有启发。 |
| Generic Event Boundary Detection | CVPR 2021, [arXiv](https://arxiv.org/abs/2101.10511) | 虽然较早，但事件边界概念对 key-window 划分仍有用。 |
| MultiHop-EgoQA / GeLM | 2024, [arXiv](https://arxiv.org/abs/2408.14469) | 用 grounding module 找 scattered temporal evidence；PriVTE 的多事件证据也可能是 scattered evidence。 |

对 PriVTE 的设计启发：

```text
event_texts should be backed by coarse time bins and evidence ids.
LLM should not invent event timing.
```

### 0.5.3 M3 Privacy-preserving visual representation

这一组文献支持 PriVTE 的核心隐私选择：不发布 raw video，也不发布高维视觉特征。

| 工作 | 来源 | 对 PriVTE 的意义 |
| --- | --- | --- |
| STPrivacy | ICCV 2023, [CVF PDF](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_STPrivacy_Spatio-Temporal_Privacy-Preserving_Action_Recognition_ICCV_2023_paper.pdf) | privacy-preserving action recognition 需要显式平衡 utility 和 privacy。 |
| Selective, Interpretable, and Motion Consistent Privacy Attribute Obfuscation | CVPR 2024 poster, [CVPR](https://cvpr.thecvf.com/virtual/2024/poster/29272), [arXiv](https://arxiv.org/abs/2403.12710) | 支撑“保留运动相关信息，去除身份相关属性”。 |
| Less Static, More Private | ICCV 2025, [CVF PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Xia_Less_Static_More_Private_Towards_Transferable_Privacy-Preserving_Action_Recognition_by_ICCV_2025_paper.pdf) | 直接支持 PriVTE 的原则：减少静态外貌/场景线索，保留动态行为代理证据。 |
| Privacy Beyond Pixels | 2025/2026, [arXiv](https://arxiv.org/abs/2511.08666) | 说明即使是 latent video features 也会泄露性别、服装等隐私，因此 public dataset 不应发布高维特征。 |
| Anonymization for Skeleton Action Recognition | 2021, [arXiv](https://arxiv.org/abs/2111.15129) | skeleton 比 RGB 更隐私，但仍能泄露性别和身份；PriVTE 不应发布高维 skeleton sequence。 |
| Face De-identification: State-of-the-art Methods and Comparative Studies | 2024, [arXiv](https://arxiv.org/abs/2411.09863) | 可作为“为什么不发布匿名化脸图像”的背景：匿名化仍有残余风险和质量损失。 |
| FaceMotionPreserve | Scientific Reports 2024, [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11283491/) | 医疗视频中试图保留面部动态而去身份化；PriVTE 则更保守，只发布文本代理证据。 |

对 PriVTE 的设计启发：

```text
Public Lite should not include raw frames, face embeddings, high-dimensional skeletons, exact coordinates, or free-form appearance descriptions.
Controlled Research may include coarser relative features, but still not re-identifiable visual embeddings.
```

### 0.5.4 M4 Edge-center split and local privacy processing

这组工作支撑未来应用设想：本地处理、中心判断、隐私边界清晰。

| 工作 | 来源 | 对 PriVTE 的意义 |
| --- | --- | --- |
| SAMEdge | 2024, [arXiv](https://arxiv.org/abs/2409.14784) | edge-cloud video analytics 架构说明视觉模型可以分布到边缘和云端。 |
| SplitStream | JNCA 2024, [paper](https://cis.temple.edu/~wu/research/publications/Publication_files/splitstream-jnca.pdf) | workload-adaptive video analytics at the edge；支撑边缘/中心协同处理的系统价值。 |
| A survey on Deep Learning in Edge-Cloud Collaboration | 2025, [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0950705125000139) | 提供 model partitioning、privacy preservation、threat model 综述。 |
| Privacy-Preserving Edge Federated Learning for Mobile-Health | 2024, [arXiv](https://arxiv.org/abs/2405.05611) | 高敏感健康场景中，原始数据不集中上传的动机与 PriVTE 类似。 |
| Privacy-Preserving Live Video Analytics for Drones via Edge Computing | 2024, [MDPI](https://www.mdpi.com/2076-3417/14/22/10254) | 说明实时视频分析中 privacy、bandwidth、latency 都促使本地处理。 |

对 PriVTE 的设计启发：

```text
Edge does raw video processing and identity mapping.
Center does text-only evidence synthesis.
Threat model and metadata leakage must be explicitly stated.
```

### 0.5.5 M5 Evidence-centered benchmark and structured outputs

PriVTE 的 benchmark 不应只给 label，还要给证据、缺失信息和 review flag。

| 工作 | 来源 | 对 PriVTE 的意义 |
| --- | --- | --- |
| ECBD: Evidence-Centered Benchmark Design for NLP | ACL 2024, [ACL PDF](https://aclanthology.org/2024.acl-long.861.pdf), [arXiv](https://arxiv.org/abs/2406.08723) | 支撑“benchmark 应明确测量对象、输入证据、输出证据和能力声明”。 |
| StructTest | 2024, [arXiv](https://arxiv.org/abs/2412.18011) | 评估 LLM compositional structured outputs；PriVTE 也要评估 JSON 输出合法性和字段一致性。 |
| CONSTRUCT | 2026, [arXiv](https://arxiv.org/abs/2603.18014) | 对 structured outputs 做字段级可信度评分；可作为未来 review workflow 参考。 |
| SimpleQA | 2024, [arXiv](https://arxiv.org/abs/2411.04368) | 强调 not attempted / factuality；PriVTE 的 `insufficient_evidence` 可以借鉴。 |
| HealthBench | 2025, [arXiv](https://arxiv.org/abs/2505.08775) | 专家 rubric 和高风险场景评测，支撑 PriVTE 的 human-review 和 safety framing。 |

对 PriVTE 的设计启发：

```text
Model output should be schema-constrained:
  risk_level
  confidence
  supporting_evidence
  missing_information
  needs_human_review

Evaluation should include:
  classification metrics
  schema validity
  evidence consistency
  calibration
  abstention/review behavior
```

### 0.5.6 方法主线目前最稳的组合

根据这一轮补充文献，PriVTE 方法主线可以写成：

```text
Long sensitive video
  -> relevance + coverage based key-window selection
  -> local proxy feature extraction with quality gates
  -> privacy-preserving aggregation and discretization
  -> schema-first evidence package
  -> text-only LLM with structured output
  -> privacy, calibration, and human-review evaluation
```

最不稳的环节仍然是：

```text
video-derived proxy evidence 是否足以预测 multimodal reference labels
```

因此实验必须保留 internal upper bounds 和 negative baselines。

## 1. 视频到文本与事件证据编码

这组工作说明“视频可以转成文本”，但也说明 PriVTE 不能简单采用自由 caption。

| 工作 | 来源 | 做了什么 | 对 PriVTE 的启发 | 局限 |
| --- | --- | --- | --- | --- |
| Dense-Captioning Events in Videos / ActivityNet Captions | ICCV 2017, [arXiv](https://arxiv.org/abs/1705.00754) | 提出 dense video captioning，在长视频中定位事件并生成事件描述。 | 可借鉴“事件窗口 + 文本事件”的组织方式。 | 目标是自然语言 caption，不是隐私过滤、结构化证据或敏感数据发布。 |
| YouCook2 / instructional video procedure learning | AAAI 2018, [arXiv](https://arxiv.org/abs/1706.09780) | 用教学视频研究步骤、过程和文本描述。 | 可借鉴长视频分段、过程性事件表示。 | 场景是公开教学视频，不处理未成年人、身份线索和高敏感标签。 |
| HowTo100M | ICCV 2019, [arXiv](https://arxiv.org/abs/1906.03327) | 利用大规模 narrated video 学习视频-文本表示。 | 说明弱监督 video-text 对齐可扩展。 | 依赖公开网络视频和 narration，不适合作为敏感视频发布范式。 |
| Ego4D | CVPR 2022, [paper](https://openaccess.thecvf.com/content/CVPR2022/html/Grauman_Ego4D_Around_the_World_in_3000_Hours_of_Egocentric_Video_CVPR_2022_paper.html), [arXiv](https://arxiv.org/abs/2110.07058) | 大规模第一视角视频基准，关注长时程、多任务和真实世界视频。 | 可借鉴 long-form video benchmark 的任务组织、annotation pipeline 和 governance。 | 仍是视频基准，不是 text-only release；隐私风险更像 PriVTE 的反例和对照。 |

### 对 PriVTE 的结论

这些工作可以放在 Related Work 的 `Video-to-text and video-language datasets` 中，但需要明确区分：

```text
captioning: video -> natural language description
PriVTE: video -> privacy-filtered structured evidence -> templated text
```

PriVTE 的文本不是“看起来像人写的描述”，而是固定 schema 下的 evidence package。

## 2. 视频问答、结构化视频推理与事件粒度

这组工作不是直接竞争对象，但对事件窗口、时间关系和结构化评测有参考价值。

| 工作 | 来源 | 做了什么 | 对 PriVTE 的启发 | 局限 |
| --- | --- | --- | --- | --- |
| TVQA | EMNLP 2018, [arXiv](https://arxiv.org/abs/1809.01696) | 基于视频片段和字幕做 localized compositional QA。 | 可借鉴 local evidence、时间定位和文本辅助评测。 | 输入包含字幕/视频，不是隐私保护文本证据。 |
| AGQA | CVPR 2021, [arXiv](https://arxiv.org/abs/2103.16002) | 用 compositional spatio-temporal reasoning 评测视频理解。 | 可借鉴把复杂视频理解拆成可组合事件和关系。 | 偏通用视觉推理，不处理敏感数据发布或风险筛查。 |
| NExT-QA | CVPR 2021, [arXiv](https://arxiv.org/abs/2105.08276) | 强调 causal / temporal action reasoning 的视频问答。 | 对 PriVTE 的事件顺序、行为趋势和原因链有启发。 | 仍是 QA，不是 label-driven risk screening。 |

### 对 PriVTE 的结论

PriVTE 可以借鉴这些工作对时序证据的处理：

- 不只用全局摘要。
- 需要 event windows。
- 需要 temporal trend。
- 需要 missing information。
- 需要把判断依据定位到证据类型，而不是让 LLM 自由发挥。

## 3. 多模态 LLM 与视频 LLM benchmark

这组工作说明近年的评测趋势，但 PriVTE 应避免被写成“又一个多模态 LLM 视频基准”。

| 工作 | 来源 | 做了什么 | 对 PriVTE 的启发 | 局限 |
| --- | --- | --- | --- | --- |
| Video-ChatGPT | arXiv 2023, [arXiv](https://arxiv.org/abs/2306.05424) | 将视频表征接入 LLM，用于详细视频理解。 | 可作为“直接让模型看视频”的对照讨论。 | 直接处理视频，不符合本项目隐私边界。 |
| Video-Bench | arXiv 2023, [arXiv](https://arxiv.org/abs/2311.16103) | 面向 video LLM 的综合评测。 | 可参考 benchmark 任务组织和评测维度。 | 目标是 video LLM 能力，不是 text-only privacy-preserving benchmark。 |
| Video-MME | arXiv 2024, [arXiv](https://arxiv.org/abs/2405.21075) | 多模态视频理解评测，覆盖多时长、多类型视频。 | 可借鉴 long video evaluation 和任务难度分层。 | 默认模型访问视频，不适合未成年人敏感视频发布。 |
| MMBench | ECCV 2024 / arXiv, [arXiv](https://arxiv.org/abs/2307.06281) | 多模态模型能力评测，强调客观题和能力维度。 | 可参考能力维度和稳定评测协议。 | 图像/多模态评测，不是本项目的 text-only setting。 |
| MMMU | CVPR 2024 / arXiv, [arXiv](https://arxiv.org/abs/2311.16502) | 大规模多学科 multimodal benchmark。 | 可参考 benchmark 构造、难度分层和报告方式。 | 不涉及敏感视频转文本证据。 |

### 对 PriVTE 的结论

Related Work 中可以写：

```text
Recent multimodal LLM benchmarks evaluate how well models understand visual inputs.
In contrast, MDU-RiskBench deliberately removes raw visual inputs and evaluates text-only evidence synthesis under privacy constraints.
```

这能把 PriVTE 和 Video-MME、Video-Bench、MMBench 等工作区分开。

## 4. Text-only LLM benchmark 与评测方法

这组工作是 MDU-RiskBench 的直接方法参考。

| 工作 | 来源 | 做了什么 | 对 PriVTE 的启发 | 可采用做法 |
| --- | --- | --- | --- | --- |
| MMLU | ICLR 2021, [arXiv](https://arxiv.org/abs/2009.03300) | 多学科文本任务评测大模型知识和推理。 | 可借鉴固定题集、统一输入、统一指标。 | 固定 split、统一 prompt、报告 per-class / per-domain。 |
| BIG-bench | ICLR 2023, [arXiv](https://arxiv.org/abs/2206.04615), [GitHub](https://github.com/google/BIG-bench) | 大规模协作式任务集合，用多任务方式评估模型能力。 | 可借鉴 benchmark task card 和任务多样性。 | 为 MDU-RiskBench 写 task card、prompt card。 |
| HELM | TMLR 2023, [arXiv](https://arxiv.org/abs/2211.09110), [project](https://crfm.stanford.edu/helm/latest/) | Holistic evaluation，强调 accuracy、calibration、robustness、fairness、efficiency 等多维度。 | 非常适合 PriVTE：不能只报 accuracy。 | 报 Macro F1、high-risk recall、calibration、abstention / review rate、privacy metrics。 |
| MT-Bench / Chatbot Arena | NeurIPS 2023 Datasets and Benchmarks, [arXiv](https://arxiv.org/abs/2306.05685) | 用 pairwise / LLM-as-judge 评估开放式对话模型。 | 可参考 judge protocol，但 PriVTE 应优先使用结构化标签和 JSON output。 | 若评估解释质量，可用人工或 rubric，而不是只依赖 LLM judge。 |

### 对 PriVTE 的结论

MDU-RiskBench 应该明确包含：

- `zero-shot`
- `few-shot`
- `rubric-based prompting`
- `JSON-constrained output`
- `small/strong LLM comparison`
- `calibration`
- `insufficient_evidence` / human-review trigger

核心主张不应是“某个 LLM 分数最高”，而应是：

```text
Evidence representation and privacy-aware schema design matter as much as model choice.
```

## 5. 数据集文档化、数据卡与治理

这组工作支撑 dataset card、release policy 和 ethics statement。

| 工作 | 来源 | 做了什么 | 对 PriVTE 的启发 |
| --- | --- | --- | --- |
| Datasheets for Datasets | CACM 2021 / arXiv, [arXiv](https://arxiv.org/abs/1803.09010), [Microsoft page](https://www.microsoft.com/en-us/research/publication/datasheets-for-datasets/) | 提出数据集发布时应回答采集、组成、用途、限制、维护等问题。 | MDU-RiskText 必须写 dataset card，说明不含 raw video、image、audio、OCR/ASR。 |
| Data Statements for NLP | TACL 2018, [ACL Anthology](https://aclanthology.org/Q18-1041/) | 为 NLP 数据集记录语言、说话人、场景和标注信息。 | 可借鉴对数据来源、人口统计、采集条件的谨慎说明。 |
| Model Cards for Model Reporting | FAT* 2019 / arXiv, [arXiv](https://arxiv.org/abs/1810.03993) | 规范报告模型用途、性能、限制和伦理。 | MDU-RiskBench 的 LLM prompt 可以写 model card 或 prompt card。 |
| NIST AI Risk Management Framework | NIST, [official](https://www.nist.gov/itl/ai-risk-management-framework) | AI 风险管理框架。 | 可用于伦理和风险管理章节。 |
| NIST De-identification | NIST, [official](https://www.nist.gov/itl/iad/deidentification) | 去标识化与隐私工程参考。 | 可支撑 PII scan、字段压缩、重识别风险分析。 |
| FTC COPPA FAQ | FTC, [official](https://www.ftc.gov/business-guidance/resources/complying-coppa-frequently-asked-questions) | 儿童隐私保护相关要求。 | 可作为未成年人数据治理的国际背景。 |

### 对 PriVTE 的结论

MDU-RiskText 不能只发布数据文件，还应发布：

- dataset card
- data use agreement
- release tiers: Public Lite / Controlled Research / Raw Internal
- PII scan report
- manual privacy audit protocol
- uniqueness / rare-combination risk report

## 6. 面部行为、情绪代理特征与质量控制

这组工作支撑 proxy feature extractor，但需要避免心理诊断化表述。

| 工作 | 来源 | 做了什么 | 对 PriVTE 的启发 | 注意事项 |
| --- | --- | --- | --- | --- |
| OpenFace 2.0 | IEEE FG 2018, [IEEE](https://ieeexplore.ieee.org/document/8373812) | 面部行为分析工具，输出 landmarks、head pose、AUs、gaze 等。 | 可作为本地特征提取器的参考：AUs、头姿、眨眼、gaze。 | 不输出 face embedding，不发布高维几何序列。 |
| AffectNet | IEEE TAC / arXiv, [arXiv](https://arxiv.org/abs/1708.03985) | 大规模面部表情/情绪数据集。 | 可参考情绪类别、表情不确定性和标注噪声。 | PriVTE 应输出“负向表情相关线索”，不输出心理状态诊断。 |

### 对 PriVTE 的结论

推荐文本表达：

```text
negative-affect-related facial action increased
frustration-related proxy event detected
low confidence due to limited face visibility
```

避免：

```text
the child is anxious
the child is addicted
the child has poor self-control
```

## 7. rPPG / 视频生理代理特征

这组工作可以作为可选模块背景。PriVTE 当前原始数据中有接触式或外部心率记录，但 public text 不应发布精确心率。

| 工作 | 来源 | 做了什么 | 对 PriVTE 的启发 | 注意事项 |
| --- | --- | --- | --- | --- |
| DeepPhys | ECCV 2018, [arXiv](https://arxiv.org/abs/1805.07888) | 用 CNN attention 从视频估计生理信号。 | 可作为 rPPG 代理特征的技术背景。 | rPPG 受光照、运动、遮挡、肤色、压缩影响大。 |
| PhysFormer | CVPR 2022, [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Yu_PhysFormer_Facial_Video-Based_Physiological_Measurement_With_Temporal_Difference_Transformer_CVPR_2022_paper.html) | 用 temporal difference transformer 做 facial video-based physiological measurement。 | 可参考时序建模和质量控制。 | 不应把 rPPG 写成心率真值。 |

### 对 PriVTE 的结论

论文中应把心率相关数据分成三类：

```text
contact/external heart-rate record: internal label source or upper-bound experiment
rPPG from video: optional proxy feature with quality gate
public text: trend/bin only, no exact HR or HRV
```

## 8. 行为风险筛查与领域背景

这组材料用于动机和边界，不应把 PriVTE 写成临床诊断。

| 材料 | 来源 | 对 PriVTE 的作用 |
| --- | --- | --- |
| WHO Gaming disorder FAQ | WHO, [official](https://www.who.int/standards/classifications/frequently-asked-questions/gaming-disorder) | 支撑“成瘾/障碍”是严肃诊断术语，本文不应直接诊断。 |
| Problematic internet / smartphone use screening literature | 后续需要系统补充 | 用于介绍未成年人数字设备使用风险的社会背景。 |

### 下一步需要补充的领域论文

后续应补：

- adolescent problematic internet use systematic review
- problematic smartphone use meta-analysis
- gaming disorder screening scales
- Chinese adolescent digital media use / internet addiction screening studies
- questionnaire-based labels and their limitations

这些文献主要服务 Introduction 和 label construction，不应取代 PriVTE 的核心方法贡献。

## 9. 隐私保护视频处理与发布策略

当前优先结论：

```text
Do not release raw video.
Do not release de-identified frames as the primary public benchmark.
Release privacy-filtered text evidence with audit records.
```

需要继续调研的方向：

- face anonymization / de-identification in video
- privacy-preserving action recognition
- skeleton-based action representation and re-identification risk
- visual privacy attacks from blurred / masked / low-resolution video
- uniqueness and linkage risk for structured behavioral records

这些工作将支撑 Privacy Evaluation 和 Release Policy。

## 10. 推荐论文叙事中的 Related Work 结构

建议正文 Related Work 分 5 段：

### 10.1 Digital Device Use Risk Screening

写社会问题、问卷/长期观察的重要性，以及 video-only 的信息上限。

### 10.2 Video-language Understanding and Captioning

覆盖 ActivityNet Captions、YouCook2、HowTo100M、Video QA。重点说明 PriVTE 不是 free-form captioning。

### 10.3 Multimodal LLM Benchmarks

覆盖 Video-MME、Video-Bench、MMBench、MMMU。重点说明这些 benchmark 默认模型访问视觉输入，而 PriVTE/MDU-RiskBench 是 text-only。

### 10.4 LLM Evaluation and Benchmark Design

覆盖 MMLU、BIG-bench、HELM、MT-Bench。重点借鉴统一 prompt、结构化输出、校准、鲁棒性和多指标评估。

### 10.5 Privacy-preserving Data Release and Documentation

覆盖 Datasheets、Data Statements、Model Cards、NIST。重点说明 MDU-RiskText 的 Public Lite / Controlled Research / Raw Internal 分级。

## 11. 对实验设计的直接启发

从上述文献可以导出 PriVTE 的实验主线：

### RQ1: Text evidence 是否支持风险筛查？

参考 MMLU / HELM 的标准化评测方式。

模型/设置：

- open-source small LLM
- open-source stronger LLM
- closed-source strong LLM
- rubric-based LLM
- JSON-constrained LLM

指标：

- Accuracy
- Macro F1
- Weighted F1
- High-risk recall
- Calibration error
- Confusion matrix

### RQ2: 证据表达方式是否重要？

参考 video captioning 和 LLM benchmark 的对照方式。

比较：

- Global text only
- Global + fixed windows
- Global + event windows
- JSON only
- templated natural language only
- JSON + templated natural language

### RQ3: Schema-first 是否优于 free caption？

对照 ActivityNet-style free caption。

评估：

- screening performance
- schema validity
- PII leakage
- reproducibility
- manual audit pass rate

### RQ4: Privacy-utility tradeoff 如何？

参考 HELM 多指标思想和 NIST privacy framing。

比较：

- internal full features
- fine-grained text
- Public Lite text
- minimal text

指标：

- Macro F1
- high-risk recall
- PII leakage rate
- uniqueness risk
- manual privacy audit pass rate

### RQ5: video-only 的信息上限在哪里？

比较：

- questionnaire + heart-rate + app + video labels
- video-only structured features
- video-only text evidence
- video-only text evidence + LLM

结论应写成：

```text
video-only screening has an information ceiling; PriVTE is designed for auxiliary screening and review, not diagnosis.
```

## 12. 当前调研缺口

下一轮需要补强：

1. 顶会/顶刊中的 privacy-preserving video analytics。
2. 未成年人视频、教育场景或医疗/心理场景中的数据发布伦理。
3. 结构化文本证据作为 LLM 输入的 benchmark 论文。
4. 行为风险筛查领域的权威量表和系统综述。
5. 置信度、人工复核和 abstention / reject option 在 high-stakes AI 中的评测方法。
6. 重识别风险、字段唯一性和 tabular privacy audit 的方法。

## 13. 初步参考文献清单

- Li et al. MVBench: A Comprehensive Multi-modal Video Understanding Benchmark. CVPR 2024. [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Li_MVBench_A_Comprehensive_Multi-modal_Video_Understanding_Benchmark_CVPR_2024_paper.html)
- Wu et al. LongVideoBench: A Benchmark for Long-context Interleaved Video-Language Understanding. NeurIPS 2024 Datasets and Benchmarks. [arXiv](https://arxiv.org/abs/2407.15754)
- Zhou et al. MLVU: A Comprehensive Benchmark for Multi-Task Long Video Understanding. 2024. [arXiv](https://arxiv.org/abs/2406.04264)
- Wang et al. LVBench: An Extreme Long Video Understanding Benchmark. ICCV 2025. [CVF PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_LVBench_An_Extreme_Long_Video_Understanding_Benchmark_ICCV_2025_paper.pdf)
- Mangalam et al. EgoSchema: A Diagnostic Benchmark for Very Long-form Video Language Understanding. NeurIPS 2023 Datasets and Benchmarks. [arXiv](https://arxiv.org/abs/2308.09126)
- Yu et al. KeyVideoLLM: Towards Large-scale Video Keyframe Selection. 2024. [arXiv](https://arxiv.org/abs/2407.03104)
- Xu et al. SlowFast-LLaVA: A Strong Training-Free Baseline for Video Large Language Models. 2024. [arXiv](https://arxiv.org/abs/2407.15841)
- Shen et al. LongVU: Spatiotemporal Adaptive Compression for Long Video-Language Understanding. 2024. [arXiv](https://arxiv.org/abs/2410.17434)
- Adaptive Keyframe Sampling for Long Video Understanding. 2025. [arXiv](https://arxiv.org/abs/2502.21271)
- Lin et al. Video-LLaVA: Learning United Visual Representation by Alignment Before Projection. EMNLP 2024. [ACL](https://aclanthology.org/2024.emnlp-main.342/)
- Wang et al. MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark. NeurIPS 2024 Datasets and Benchmarks. [arXiv](https://arxiv.org/abs/2406.01574)
- White et al. LiveBench: A Challenging, Contamination-Free LLM Benchmark. 2024. [arXiv](https://arxiv.org/abs/2406.19314)
- Wei et al. SimpleQA: Measuring short-form factuality in large language models. 2024. [arXiv](https://arxiv.org/abs/2411.04368)
- Rein et al. GPQA: A Graduate-Level Google-Proof Q&A Benchmark. 2023/2024. [arXiv](https://arxiv.org/abs/2311.12022)
- Yue et al. MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark. 2024. [arXiv](https://arxiv.org/abs/2409.02813)
- HealthBench. OpenAI 2025. [arXiv](https://arxiv.org/abs/2505.08775)
- MedSafetyBench. NeurIPS 2024 Datasets and Benchmarks. [arXiv](https://arxiv.org/abs/2403.03744)
- Cai et al. MedBench: A Large-Scale Chinese Benchmark for Evaluating Medical Large Language Models. 2024. [arXiv](https://arxiv.org/abs/2407.10990)
- Multi-P2A: Evaluating Privacy Awareness and Leakage in LVLMs. 2024. [arXiv](https://arxiv.org/abs/2412.19496)
- PII-Scope: A Benchmark for Privacy Leakage in LLMs. 2024. [arXiv](https://arxiv.org/abs/2410.06704)
- MLLMU-Bench: A Benchmark for Multimodal Large Language Model Unlearning. NAACL 2025. [ACL](https://aclanthology.org/2025.naacl-long.207/)
- Benchmarking Benchmark Leakage in Large Language Models. 2024. [arXiv](https://arxiv.org/abs/2404.18824)
- PrivAuditor. NeurIPS 2024 Datasets and Benchmarks. [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/12b18a15dcd73e1991e9959a94375fab-Paper-Datasets_and_Benchmarks_Track.pdf)
- Bowman and Dahl. What Will it Take to Fix Benchmarking in Natural Language Understanding? Evidence-Centered Benchmark Design. 2024. [arXiv](https://arxiv.org/abs/2406.08723)
- Tang et al. Adaptive Keyframe Sampling for Long Video Understanding. CVPR 2025. [arXiv](https://arxiv.org/abs/2502.21271)
- Tang et al. Query-Conditioned Evidential Keyframe Sampling. 2026. [arXiv](https://arxiv.org/abs/2604.01002)
- From Frames to Clips: Adaptive Key Clip Selection. 2025. [arXiv](https://arxiv.org/abs/2510.02262)
- Argaw et al. Scaling Up Video Summarization Pretraining with Large Language Models. CVPR 2024. [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Argaw_Scaling_Up_Video_Summarization_Pretraining_with_Large_Language_Models_CVPR_2024_paper.html)
- LVSum: A Large-Scale and High-Quality Dataset for Long Video Summarization. 2026. [arXiv](https://arxiv.org/abs/2604.10024)
- Grounded-VideoLLM. 2024. [arXiv](https://arxiv.org/abs/2410.03290)
- Training-free Video Temporal Grounding. ECCV 2024. [arXiv](https://arxiv.org/abs/2408.16219)
- HawkEye: Training Video-Text LLMs for Grounding Text in Videos. 2024. [arXiv](https://arxiv.org/abs/2403.10228)
- Generic Event Boundary Detection. CVPR 2021. [arXiv](https://arxiv.org/abs/2101.10511)
- MultiHop-EgoQA / GeLM. 2024. [arXiv](https://arxiv.org/abs/2408.14469)
- SAMEdge. 2024. [arXiv](https://arxiv.org/abs/2409.14784)
- SplitStream. Journal of Network and Computer Applications 2024. [PDF](https://cis.temple.edu/~wu/research/publications/Publication_files/splitstream-jnca.pdf)
- StructTest: Benchmarking LLMs' Reasoning through Compositional Structured Outputs. 2024. [arXiv](https://arxiv.org/abs/2412.18011)
- CONSTRUCT: A Benchmark for Evaluating Structured Outputs of LLMs. 2026. [arXiv](https://arxiv.org/abs/2603.18014)
- Krishna et al. Dense-Captioning Events in Videos. ICCV 2017. [arXiv](https://arxiv.org/abs/1705.00754)
- Zhou et al. Automatic Learning of Procedures from Web Instructional Videos. AAAI 2018. [arXiv](https://arxiv.org/abs/1706.09780)
- Miech et al. HowTo100M: Learning a Text-Video Embedding by Watching Hundred Million Narrated Video Clips. ICCV 2019. [arXiv](https://arxiv.org/abs/1906.03327)
- Grauman et al. Ego4D: Around the World in 3,000 Hours of Egocentric Video. CVPR 2022. [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Grauman_Ego4D_Around_the_World_in_3000_Hours_of_Egocentric_Video_CVPR_2022_paper.html)
- Lei et al. TVQA: Localized, Compositional Video Question Answering. EMNLP 2018. [arXiv](https://arxiv.org/abs/1809.01696)
- Grunde-McLaughlin et al. AGQA: A Benchmark for Compositional Spatio-Temporal Reasoning. CVPR 2021. [arXiv](https://arxiv.org/abs/2103.16002)
- Xiao et al. NExT-QA: Next Phase of Question-Answering to Explaining Temporal Actions. CVPR 2021. [arXiv](https://arxiv.org/abs/2105.08276)
- Maaz et al. Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models. 2023. [arXiv](https://arxiv.org/abs/2306.05424)
- Ning et al. Video-Bench: A Comprehensive Benchmark and Toolkit for Evaluating Video-based Large Language Models. 2023. [arXiv](https://arxiv.org/abs/2311.16103)
- Fu et al. Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis. 2024. [arXiv](https://arxiv.org/abs/2405.21075)
- Liu et al. MMBench: Is Your Multi-modal Model an All-around Player? ECCV 2024 / arXiv. [arXiv](https://arxiv.org/abs/2307.06281)
- Yue et al. MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark. CVPR 2024 / arXiv. [arXiv](https://arxiv.org/abs/2311.16502)
- Hendrycks et al. Measuring Massive Multitask Language Understanding. ICLR 2021. [arXiv](https://arxiv.org/abs/2009.03300)
- Srivastava et al. Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models. ICLR 2023. [arXiv](https://arxiv.org/abs/2206.04615)
- Liang et al. Holistic Evaluation of Language Models. TMLR 2023. [arXiv](https://arxiv.org/abs/2211.09110)
- Zheng et al. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. NeurIPS 2023 Datasets and Benchmarks. [arXiv](https://arxiv.org/abs/2306.05685)
- Gebru et al. Datasheets for Datasets. CACM 2021. [arXiv](https://arxiv.org/abs/1803.09010)
- Bender and Friedman. Data Statements for Natural Language Processing. TACL 2018. [ACL Anthology](https://aclanthology.org/Q18-1041/)
- Mitchell et al. Model Cards for Model Reporting. FAT* 2019. [arXiv](https://arxiv.org/abs/1810.03993)
- Baltrusaitis et al. OpenFace 2.0: Facial Behavior Analysis Toolkit. IEEE FG 2018. [IEEE](https://ieeexplore.ieee.org/document/8373812)
- Mollahosseini et al. AffectNet. IEEE TAC / arXiv. [arXiv](https://arxiv.org/abs/1708.03985)
- Chen and McDuff. DeepPhys: Video-Based Physiological Measurement Using Convolutional Attention Networks. ECCV 2018. [arXiv](https://arxiv.org/abs/1805.07888)
- Yu et al. PhysFormer: Facial Video-Based Physiological Measurement With Temporal Difference Transformer. CVPR 2022. [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Yu_PhysFormer_Facial_Video-Based_Physiological_Measurement_With_Temporal_Difference_Transformer_CVPR_2022_paper.html)
- NIST AI Risk Management Framework. [official](https://www.nist.gov/itl/ai-risk-management-framework)
- NIST De-identification. [official](https://www.nist.gov/itl/iad/deidentification)
- FTC COPPA FAQ. [official](https://www.ftc.gov/business-guidance/resources/complying-coppa-frequently-asked-questions)
- WHO Gaming disorder FAQ. [official](https://www.who.int/standards/classifications/frequently-asked-questions/gaming-disorder)
