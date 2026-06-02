# Zero-Retention PriVTE 方案：采集中实时证据编码、不保存原始视频

## 0. 快速结论

这个想法值得做，而且比“录完视频后再脱敏”更有新意。它的核心提升是把隐私保护从后处理阶段前移到采集阶段：

```text
record-then-sanitize
  -> sense-and-discard
```

但需要区分两个层次：

1. **工程层面的不落盘**：摄像头流逐帧处理，不调用视频保存接口。这是可行的，MVP 难度中等，但本身不足以支撑 CVPR 主会。
2. **研究层面的 zero-retention visual evidence encoding**：提出一个在线、不可回看、低留存、可审计的视觉证据编码方法，并系统评估 privacy、utility、real-time 三者权衡。这有成为 CVPR/ACM MM/FAccT 论文的潜力。

我对投稿判断是：

```text
AAAI AISI 主线:
  PriVTE + MDU-RiskText + MDU-RiskBench 是更稳的当前论文主线。

CVPR 主线:
  Zero-Retention PriVTE 可以作为第二篇或增强版方法论文，
  但必须从“未成年人风险筛查系统”抽象成更通用的
  privacy-preserving streaming visual evidence encoding。
```

截至 2026-06-02，CVPR 2026 主会投稿时间已经过去。若考虑主会，现实目标应是 CVPR 2027 或相关 workshop / ACM MM / FAccT。

## 1. 核心想法

当前 PriVTE 的基础设定是：

```text
先录制 raw video
  -> 本地处理
  -> 隐私过滤
  -> 生成 text evidence
  -> 不公开 raw video
```

进一步的更强设定是：

```text
录制过程中实时提取特征和证据；
raw frames 只在内存缓冲区中短暂存在；
不生成本地视频文件；
最终只保留结构化特征、质量指标和文本证据。
```

可以暂命名为：

```text
Zero-Retention PriVTE
```

或：

```text
Streaming PriVTE
```

核心口号：

```text
From record-then-sanitize to sense-and-discard.
```

中文解释：

```text
不是先保存视频再脱敏，而是在采集过程中实时抽取可用证据，
原始视频不落盘，证据生成后原始帧立即丢弃。
```

## 2. 这个方向是否有创新性？

有潜在较大创新，但取决于做到什么程度。

### 2.1 作为普通工程实现

如果只是：

```text
OpenCV 打开摄像头 -> 逐帧处理 -> 不调用 VideoWriter
```

这只是工程实现，不足以作为 CVPR 级创新。

### 2.2 作为隐私保护视觉证据编码方法

如果进一步做到：

- 在线 key-window / event-window 选择；
- 实时 ROI 和代理特征提取；
- 只保留聚合特征，不保留可重建帧；
- 有明确 non-retention threat model；
- 有可审计日志证明 raw video 未落盘；
- 有 privacy-utility-real-time 三维评估；
- 与 record-then-sanitize、face blur、free caption、direct Video-LLM 等路线比较；

那么它可能成为一个有新意的视觉隐私方法。

可能的论文定位：

```text
privacy-preserving streaming visual evidence encoding
```

而不仅是：

```text
video-to-text preprocessing
```

## 3. 与现有文献的关系

### 3.1 支撑方向的相关工作

| 方向 | 相关工作 | 对本方案的启发 |
| --- | --- | --- |
| 长视频关键帧/关键片段选择 | KeyVideoLLM, Adaptive Keyframe Sampling, LongVU | 长视频不能全量保留和处理，需要 relevance + coverage 的在线选择。 |
| 视频摘要与 temporal grounding | LVSum, Grounded-VideoLLM, Training-free Video Temporal Grounding | 最终证据需要时间窗口和事件定位，而不是整段视频自由总结。 |
| 隐私保护视觉表征 | STPrivacy, Less Static More Private, Privacy Beyond Pixels | 隐私保护不能只靠模糊脸；高维特征也可能泄露身份。 |
| 隐私保护光学/采集端保护 | Privacy-Preserving Optics for Face De-identification, CVPR 2024 | 隐私可以前移到采集端，而不是采集后再处理。 |
| 边缘-中心视频分析 | SAMEdge, SplitStream, edge-cloud collaboration survey | 支撑本地实时处理、中心只做轻量判断的系统结构。 |
| 结构化证据与 LLM 评测 | ECBD, StructTest, HealthBench | 输出应是结构化证据和可评估判断，而不是自由文本。 |

### 3.2 关键差异

很多隐私视频工作仍然保留某种图像、匿名化视频或高维视觉特征。

Zero-Retention PriVTE 更激进：

```text
public / central side never sees raw frames, de-identified frames, or high-dimensional visual embeddings.
```

它只看到：

```text
coarse evidence states
quality indicators
event summaries
privacy-filtered text
```

## 4. 系统架构

### 4.1 数据流

```text
Camera stream
  -> volatile frame buffer
  -> online ROI detection
  -> online proxy feature extraction
  -> streaming aggregation
  -> event/window scoring
  -> privacy filtering
  -> evidence state update
  -> discard raw frame
  -> final JSON + templated text evidence
```

### 4.2 最终保留的数据

只保留：

- 匿名本地 sample id；
- 粗粒度时间段；
- 全局行为统计；
- 事件类型；
- 质量指标；
- 隐私处理日志；
- text evidence；
- 本地人工复核记录。

不保留：

- 视频文件；
- 原始帧；
- 关键帧；
- 音频；
- OCR/ASR 原文；
- face embedding；
- 高维 skeleton；
- 精确心率；
- 精确坐标；
- 外貌和场景描述。

## 5. 技术模块

### M1. Volatile frame buffer

只在内存中维护短窗口，例如：

```text
last 1-5 seconds for feature extraction
last 10-30 seconds for event confirmation
```

处理完成后立即释放。

注意：如果需要更严格保证，应避免 swap、自动缓存、调试 dump、日志保存图像。

### M2. Online key-window scoring

离线 PriVTE 可以先看完整视频再选 key windows。

Zero-Retention PriVTE 不允许回看完整视频，因此需要在线评分：

```text
score_t = coverage_score + event_score + quality_score + baseline_score
```

系统只保留窗口的统计状态，而不是窗口视频。

### M3. Streaming proxy feature extraction

逐帧或低频采样提取：

- 人脸可见度；
- 手部可见度；
- 设备可见度；
- 头部姿态；
- 眨眼 proxy；
- 手部操作强度；
- 姿态变化；
- 表情相关 AU proxy；
- 屏幕注视 proxy。

所有特征在线更新为聚合统计：

```text
count
ratio
trend
peak
duration bin
confidence
quality
```

不保存逐帧序列。

### M4. Streaming event detection

检测事件类型，例如：

- sustained_screen_attention；
- interaction_spike；
- repeated_actions_increased；
- posture_forward_sustained；
- low_blink_rate_window；
- negative_affect_related_peak；
- low_quality_or_occlusion_window。

事件输出：

```json
{
  "time_bin": "08-10min",
  "event_type": "interaction_spike",
  "evidence": [
    "touch_rate_above_baseline",
    "repeated_actions_increased"
  ],
  "quality": "medium",
  "confidence": 0.72
}
```

### M5. Privacy-aware state machine

每个字段进入 evidence package 前，都经过：

```text
field allowlist
field denylist
granularity compression
PII scan
rare-combination check
```

### M6. Evidence generation

最终输出：

```text
structured JSON
templated natural language
limitations
privacy_processing_summary
```

## 6. 难度分析

### 6.1 工程 MVP 难度：中等

可以较快做一个 MVP：

```text
摄像头 / 视频流输入
逐帧处理
不保存视频
提取简单质量和行为统计
生成 JSONL
```

难点不在“不保存视频”，而在：

- 实时稳定检测人脸、手部、设备；
- 低算力设备上运行；
- 不保存帧的情况下做事件确认；
- 质量估计和异常窗口选择；
- 保证日志和缓存不泄露帧。

### 6.2 CVPR 级难度：高

如果目标是 CVPR，不能只做系统 demo。

需要至少一个清晰算法贡献，例如：

1. **Online privacy-preserving key-window selection**

   在不保留视频的条件下，在线选择覆盖性和事件性证据。

2. **Non-retentive visual evidence state representation**

   设计一种不可逆、低重识别风险、但保留筛查效用的 evidence state。

3. **Privacy-utility-real-time benchmark**

   同时评估：

   - 任务效用；
   - 隐私泄露；
   - 实时性能；
   - 存储占用；
   - 与 record-then-sanitize 的差异。

4. **Verifiable non-retention protocol**

   至少要证明应用层不写视频文件、不保存帧、不保留可重建视觉序列。

更强版本可以考虑：

- trusted execution environment；
- secure enclave；
- remote attestation；
- frame buffer audit；
- no-swap / no-dump runtime；
- storage write monitor。

但这会把难度显著提高。

## 7. CVPR 可行性判断

### 7.1 如果只做“采集中不保存视频”

CVPR 可能性较低。

原因：

```text
这是合理工程，但视觉算法创新不足。
```

### 7.2 如果做成“零留存视觉证据编码方法”

CVPR 有一定可能，但需要：

- 明确视觉算法问题；
- 和长视频 keyframe selection / video summarization / privacy-preserving action recognition 对话；
- 有真实敏感场景数据；
- 有隐私-效用-实时性三维评估；
- 有强 baseline；
- 有系统级 non-retention 证据。

可能的 CVPR 题目方向：

```text
Zero-Retention Visual Evidence Encoding for Privacy-Preserving Risk Screening
```

或：

```text
Streaming Privacy-Preserving Video-to-Text Evidence Encoding Without Video Retention
```

### 7.3 更稳的投稿路径

如果方法创新不够强，可能更适合：

- AAAI AISI；
- ACM FAccT；
- CHI / CSCW；
- ACM Multimedia；
- NeurIPS Datasets and Benchmarks；
- ML4H / health AI workshop；
- CVPR workshop on privacy / responsible vision。

## 8. 实验设计

### 8.1 Utility

比较：

- offline PriVTE；
- zero-retention streaming PriVTE；
- free caption baseline；
- direct Video-LLM internal upper bound；
- internal structured feature upper bound。

指标：

- Macro F1；
- high-risk recall；
- calibration error；
- insufficient_evidence rate；
- human-review trigger rate。

### 8.2 Privacy

比较：

- raw video；
- blurred / anonymized video；
- high-dimensional visual features；
- offline PriVTE text；
- zero-retention PriVTE text。

指标：

- 是否存在可恢复视觉帧；
- PII leakage rate；
- identity attribute leakage；
- face / skeleton re-identification attack success；
- rare-combination uniqueness；
- manual privacy audit pass rate。

### 8.3 Real-time system

指标：

- FPS；
- latency；
- CPU / GPU / memory；
- energy；
- storage writes；
- dropped frames；
- edge device compatibility。

### 8.4 Non-retention audit

检查：

- 是否生成视频文件；
- 是否生成临时图片；
- 是否写入系统缓存；
- 是否写入日志；
- 是否保留可重建 frame sequence；
- 是否产生 debug dumps；
- 是否发生 swap。

## 9. 最小可行版本

第一阶段不需要立刻做完整系统。

MVP 可以是：

```text
输入: 摄像头流或模拟视频流
处理: 实时抽帧、提取质量指标和简单行为 proxy
存储: 只保存 JSON evidence，不保存视频和帧
输出: person-level / clip-level evidence JSONL
```

MVP 模块：

1. 视频流读取器；
2. volatile ring buffer；
3. frame sampling；
4. face / hand / device visibility estimation；
5. simple motion / interaction proxy；
6. online event counter；
7. evidence JSON generator；
8. storage audit log。

## 10. 结论

这个方向有价值，而且比“录完视频再脱敏”更强。

但要区分两种主张：

```text
工程主张:
  我们可以在采集中不保存视频，只保存证据。

CVPR 级主张:
  我们提出一种零留存、在线、隐私保护的视觉证据编码方法，
  在隐私、效用和实时性之间取得可量化的权衡。
```

建议当前论文仍以 AAAI AISI 的 benchmark / evidence encoding 为主线。

同时可以把 Zero-Retention PriVTE 作为：

- 一个更强的技术扩展；
- 一个未来 CVPR / ACM MM / FAccT 方向；
- 一个可能形成第二篇论文的方法创新点。

## 11. 如果要冲 CVPR，论文主线应如何改

CVPR 不太会因为“我们没有保存视频”本身接收一篇论文。主线应该改成一个计算机视觉问题：

```text
Can a video system preserve task-relevant visual evidence while never retaining raw visual data?
```

更具体地说，可以定义为：

```text
Zero-retention streaming visual evidence encoding:
given a camera stream, produce a compact textual / symbolic evidence state
for downstream decision-making, while raw frames are held only in volatile memory
and discarded immediately after local feature extraction.
```

这样写的好处是：

- 它仍然和 PriVTE 一致；
- 它从单一应用上升为视觉隐私方法；
- 它能和隐私保护动作识别、视频匿名化、长视频压缩、边缘视频分析等 CVPR 相关方向对话；
- MDU 场景可以作为真实敏感应用 case study，而不是唯一实验支撑。

建议 CVPR 版本贡献写成三点：

1. **Problem**：提出 zero-retention visual evidence encoding，在采集端实时生成证据状态，不保存 raw video、keyframe 或可重建高维序列。
2. **Method**：提出一个在线 evidence state 机制，包括 volatile buffer、online key-window scoring、streaming proxy extraction、privacy-aware state compression 和 templated evidence generation。
3. **Benchmark / Evaluation**：构建 privacy-utility-real-time 评估协议，比较 raw video、blurred video、offline PriVTE、zero-retention PriVTE、direct Video-LLM 等路线。

## 12. 关键技术难点

### 12.1 不保存视频不是难点，难点是不可回看

离线处理可以先看完整视频，再选关键片段。zero-retention 不能回看，所以必须在线决定哪些状态值得保留。

可行思路：

```text
short volatile buffer
  -> frame-level proxy features
  -> window-level streaming statistics
  -> event score
  -> update bounded evidence state
  -> discard frames and dense features
```

### 12.2 要避免“低维特征也泄露身份”

不能简单保存 face embedding、skeleton sequence、dense pose、逐帧 AU、逐帧 gaze 等高维时间序列。它们虽然不是像素，但仍可能重识别。

公开或中心侧只应保留：

- 粗粒度区间；
- 统计比例；
- 趋势；
- 事件类别；
- 质量等级；
- 缺失信息；
- 隐私处理日志。

### 12.3 要有 non-retention threat model

最小版 threat model 可以这样定义：

```text
中心服务器不可信或半可信；
本地采集端由研究系统控制；
系统保证应用层不写入 raw video、frame image、audio、OCR/ASR 原文或可重建视觉序列；
不防御恶意操作系统、恶意摄像头驱动、物理攻击或屏幕录制。
```

更强版本可以加入：

- tmpfs / encrypted memory buffer；
- no-swap；
- 禁止 debug dump；
- storage write monitor；
- TEE / secure enclave；
- remote attestation。

但这些会把项目从 CV 方法扩展到安全系统，工作量明显增加。

## 13. 最小实现路线

建议先做一个“可证明思路成立”的 MVP，不要一开始追求完整边缘设备部署。

### 阶段 A：用已有视频做 streaming replay

输入仍然是已有 raw video，但程序按实时流逐帧读入，模拟摄像头采集：

```text
read frame
extract proxy features
update evidence state
discard frame
never call VideoWriter
never save image
only save JSON evidence
```

这个阶段可以快速比较：

- offline PriVTE；
- streaming PriVTE；
- streaming PriVTE with smaller memory budget；
- streaming PriVTE with lower frame rate。

### 阶段 B：接真实摄像头流

使用 webcam / mobile camera / edge device 做实时采集，不生成本地视频文件。

输出：

- `sample_id`；
- `evidence.json`；
- `evidence.txt`；
- `storage_audit.log`；
- `runtime_metrics.json`。

### 阶段 C：做隐私审计

检查：

- 是否出现 `.mp4`、`.avi`、`.jpg`、`.png` 等落盘文件；
- 日志里是否出现 OCR/ASR、文件名、路径、设备 ID；
- 是否保存了逐帧坐标或高维 face/skeleton 特征；
- evidence text 是否包含学校、家庭、外貌、屏幕内容等敏感信息；
- 稀有行为组合是否可能重识别。

## 14. 实验表建议

| 实验 | 目的 | 对比方法 | 指标 |
| --- | --- | --- | --- |
| Utility | 证明 zero-retention 仍保留筛查信号 | offline PriVTE, streaming PriVTE, free caption, direct Video-LLM upper bound | Macro F1, high-risk recall, ECE, insufficient evidence rate |
| Privacy | 证明中心侧泄露更少 | raw video, blurred video, high-dimensional features, PriVTE text | PII leakage, identity attribute prediction, rare-combination uniqueness |
| Real-time | 证明可部署 | 不同 FPS / buffer / device | latency, FPS, CPU/GPU, memory, storage writes |
| Ablation | 证明方法模块必要 | no event state, no quality gate, no baseline normalization | F1 drop, calibration, false high-risk rate |
| Non-retention audit | 证明系统确实不落盘 | storage monitor, file scan, log scan | video/image artifact count, raw frame residual count |

## 15. 对当前项目的建议

短期不要把 AAAI AISI 主线完全切到 CVPR。更稳的策略是双线推进：

```text
主线论文:
  PriVTE / MDU-RiskText / MDU-RiskBench
  目标 AAAI AISI，强调社会影响、隐私保护数据发布、text-only benchmark。

增强方法:
  Zero-Retention PriVTE
  目标 CVPR workshop / ACM MM / FAccT，成熟后再考虑 CVPR 主会。
```

如果后续要真正冲 CVPR 主会，需要尽快补两类实验支撑：

1. **通用公开视频集实验**：用公开视频数据验证 zero-retention evidence encoding 的通用 privacy-utility tradeoff。否则只有 MDU 私有数据，CVPR 可复现性会弱。
2. **MDU 真实敏感场景实验**：作为强应用 case study，说明该方法为什么在未成年人敏感视频中有社会价值。

## 16. 参考文献入口

这些工作可作为后续 Related Work 的入口，文档先记录方向，不在这里展开完整综述。

- [CVPR 2026 Call for Papers](https://cvpr.thecvf.com/Conferences/2026/CallForPapers)：CVPR 主题覆盖 video understanding、efficient methods、fairness/privacy/ethics 等方向，但主会仍需要清晰视觉方法贡献。
- [Privacy-Preserving Optics for Enhancing Protection in Face De-Identification, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Lopez_Privacy-Preserving_Optics_for_Enhancing_Protection_in_Face_De-Identification_CVPR_2024_paper.pdf)：说明隐私保护可以前移到采集/成像端，而不只是后处理。
- [STPrivacy: Spatio-Temporal Privacy-Preserving Action Recognition, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_STPrivacy_Spatio-Temporal_Privacy-Preserving_Action_Recognition_ICCV_2023_paper.pdf)：支撑 privacy-utility tradeoff 的视觉识别实验设计。
- [Selective, Interpretable, and Motion Consistent Privacy Attribute Obfuscation for Action Recognition, arXiv 2024](https://arxiv.org/abs/2403.12710)：与“保留任务相关动态、移除身份相关属性”的思路接近。
- [KeyVideoLLM, arXiv 2024](https://arxiv.org/abs/2407.03104)：支撑长视频关键帧/关键窗口选择。
- [Adaptive Keyframe Sampling for Long Video Understanding, arXiv 2025](https://arxiv.org/abs/2502.21271)：支撑 relevance + coverage 的长视频采样思路。
- [TempCompass, ACL Findings 2024](https://aclanthology.org/2024.findings-acl.517/)：说明 Video LLM 的时序感知仍需要专门评估，不能简单依赖自由视频理解。
- [Privid: Practical, Privacy-Preserving Video Analytics, NSDI 2022](https://www.usenix.org/conference/nsdi22/presentation/cangialosi)：作为 privacy-preserving video analytics 的系统类对照。
