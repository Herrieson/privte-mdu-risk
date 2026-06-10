# PriVTE-Trace 设计草案

> 这是当前的 provisional research draft，不是最终定案。
> 下面的结论是基于近期文献和现有实验现象做出的阶段性判断，后续还要继续比较替代方案。

## 1. 命名建议

这版我不建议继续叫 `V4`。更合适的是把它当成一个新的编码协议来命名：

```text
PriVTE-Trace
```

含义是：

```text
privacy-safe behavioral trace encoding
```

它强调的不是“再加几个统计量”，而是把敏感视频压缩成一条**有先后顺序的行为轨迹**，让 text-only LLM 直接读这条轨迹做筛查。

## 2. 为什么需要这一版

当前 `behavior_v3_temporal` 已经比前面版本更接近“序列证据”，但它仍然偏 episode-centric，LLM 读到的还是一组比较碎的状态和计数。

从最近一轮结果看，`v3_temporal + gpt-5-5` 在 `all_current` 上已经到 `accuracy=0.56`，但 `macro_f1_observed_labels=0.2456`，而且预测几乎集中在 `mild_risk`。这说明问题不只是模型，而是证据契约还不够分层。

核心瓶颈是：

```text
现在的证据太像“统计摘要”
而不是“可判断的行为轨迹”
```

所以下一版应该把输出从“episode 列表”升级成“trace 轨迹”。

## 3. 核心目标

PriVTE-Trace 只做三件事：

1. 把视频里的可观察代理证据压成少量行为阶段。
2. 把阶段之间的转折点显式写出来。
3. 把不确定性、混淆因素和缺失信息和主轨迹一起输出。

它仍然坚持：

```text
video-only proxy evidence -> privacy-filtered text -> text-only screening
```

不走：

```text
raw video -> free-form caption -> vague answer
```

## 4. 建议的输出契约

建议把输出从“特征块 + episode 列表”换成下面这种更像 trace 的结构：

```text
trace_header
trace_signature
trace_phases[]
transition_events[]
risk_cues[]
counterevidence[]
missing_information[]
needs_human_review
```

### 4.1 `trace_header`

记录最基础的可审计信息：

- 样本编号；
- 证据质量总评；
- 覆盖位置；
- 抽样帧数；
- 可用但未作为输入的模态；
- 隐私约束摘要。

### 4.2 `trace_signature`

用一行短字符串概括整个轨迹，方便 LLM 快速抓主线。

例如：

```text
no_device_or_visible_only -> passive_engagement -> interaction_burst -> confounded_motion -> disengagement
```

或更短一点：

```text
visible_only | passive | active | confounded | gap
```

它不是分类标签，而是**行为轨迹签名**。

### 4.3 `trace_phases[]`

每个 phase 建议包含：

- `phase_id`
- `relative_position`，例如 early / middle / late
- `phase_type`
- `duration_bin`
- `confidence`
- `support_level`
- `supporting_cues`
- `counterevidence`
- `privacy_filter`
- `risk_relevance`

建议阶段词表控制在一个很小的集合里：

- `no_device_or_visible_only`
- `brief_passive_contact`
- `sustained_passive_engagement`
- `active_interaction_burst`
- `repetitive_operation_run`
- `motion_confounded_activity`
- `disengagement_or_gap`
- `insufficient_quality`

### 4.4 `transition_events[]`

这是这一版和 v3 最大的区别之一。

除了告诉模型“看到了什么阶段”，还要告诉它：

- 从什么阶段切到什么阶段；
- 这个切换是强化证据，还是削弱证据；
- 这个切换是由设备附近活动、姿态变化，还是持续参与触发的。

这样 LLM 才更容易判断：

```text
是短暂接触
还是持续参与
是持续参与
还是频繁交互
是交互
还是被姿态/镜头运动混淆
```

## 5. 这版为什么更可能有效

### 5.1 更好地区分 `no_observed_risk`

现在很多样本里，LLM 只要看见“设备可见 + 一点被动参与”，就容易往 `mild_risk` 走。

PriVTE-Trace 要显式写出：

- 设备是否只是可见；
- 被动接触是不是很短；
- 有没有真正的持续参与；
- 有没有交互 burst；
- 有没有重复操作 run；
- 有没有明显混淆。

这样 `no_observed_risk` 才能有更清楚的正向证据，而不是只靠“缺少什么”来判断。

### 5.2 更好地区分 `mild_risk` 和 `moderate_risk`

当前 v3 的问题是“只要有一串 passive episode，就容易被读成 mild”。

Trace 版本要把 moderate 的条件写得更像轨迹：

- 多个阶段反复进入 passive engagement；
- 中间穿插 active interaction burst；
- 交互不是单点，而是有复现；
- 混淆因素不能主导解释。

也就是说，moderate 不是“更多计数”，而是“更完整的行为轨迹”。

### 5.3 更像给 LLM 的“小型行为叙事”

你前面提到的方向其实很对：最终系统希望像一个很小的专用 VLLM 输出文字描述。

PriVTE-Trace 应该输出的是：

```text
一条有顺序、可审计、可复核的行为叙事
```

而不是：

```text
一堆抽象统计值
```

## 6. 技术实现思路

这版可以保留现有本地视觉骨干，但把聚合逻辑换掉。

### 6.1 仍然保留的部分

- 本地帧采样；
- 设备/屏幕可见性；
- 手部/姿态/人脸相关代理；
- 运动突增检测；
- 质量估计；
- 隐私过滤；
- text-only 渲染。

### 6.2 新增的部分

1. **phase segmentation**

   把 frame-level / clip-level proxy 先压成 phase。

2. **transition scoring**

   对 phase 之间的变化打分，保留最关键的转折点。

3. **trace compression**

   最终只保留 3 到 6 个高信息量 phase，避免 episode 太碎。

4. **trace rendering**

   生成一段短的自然语言 trace，再配一个 JSON 结构。

## 7. 建议的渲染形态

建议最终给 LLM 的文本长这样：

```text
轨迹摘要: early 阶段主要为 no_device_or_visible_only; middle 出现短时 passive engagement; late 出现有限 interaction burst, 但 repetitive_operation 仍弱且 confounded_motion 较明显.
```

后面再跟 3 到 5 条 phase 证据和 2 到 4 条 counterevidence。

也就是说：

```text
先给一条总轨迹
再给阶段证据
最后给不确定性
```

## 8. 和当前版本的边界

PriVTE-Trace 不是简单把 `v3_temporal` 改名。

它的变化点是：

- 从 episode list 转向 phase trace；
- 从计数导向转向转折点导向；
- 从“证据块堆叠”转向“小型行为叙事”；
- 从“能看见什么”转向“发生了什么顺序变化”。

## 9. 和后续 Zero-Retention 的关系

这版很适合成为后续 `sense-and-discard` 路线的中间层。

因为如果未来做实时采集端编码，那么最终保留下来的也不该是原始帧，而应该是：

- phase trace；
- transition events；
- privacy summary；
- human review flag。

所以 PriVTE-Trace 可以看成是：

```text
从当前离线证据编码
走向未来 streaming zero-retention encoding 的过渡版本
```

## 10. 一句话结论

如果要给下一版一个更像论文方法名的代号，我建议直接用：

```text
PriVTE-Trace
```

它比 `V4` 更能说明这版方法的本质变化：

```text
不是更大的特征表
而是更好的行为轨迹
```

## 11. 仍需继续验证的问题

这版目前是“最值得继续推进”的方向，但还没到最终锁死的程度。后面还要继续验证：

- `PriVTE-Trace` 是否真的能比 `v3_temporal` 更好地区分 `no_observed_risk` 和 `mild_risk`。
- phase 压缩后是否会损失对 `moderate_risk` 所需的重复性和持续性线索。
- 轨迹摘要会不会又退化成另一种“更高级的统计摘要”。
- 现在的文本长度、信息密度和 LLM 可读性是否刚好，不需要再调。
- 是否需要把 `zero-retention` 单独拆成下一篇方法论文，而不是并入当前主线。

所以当前更准确的表述是：

```text
PriVTE-Trace is the leading candidate, not the final verdict.
```

## 12. 底层技术栈候选

这一节先记录当前更合理的底层技术选择。它不是最终实现清单，但可以作为后续收敛的依据。

### 12.1 当前推荐保留的底座

| 模块 | 当前建议 | 原因 |
| --- | --- | --- |
| 视频读取 / 抽帧 | 短期继续用 `OpenCV VideoCapture`；中期可补 `FFmpeg / PyAV` 作为更稳的解码层 | 现有代码已经跑通 OpenCV；PyAV/FFmpeg 更适合后续稳定抽帧和 streaming / zero-retention |
| 设备 / 屏幕候选区域 | `YOLO11n` 设备检测 + bright-rectangle 屏幕样区域启发式 | YOLO 负责常见设备类，启发式补充未被 COCO 类别覆盖的屏幕样区域 |
| 手部 / 人脸 / 姿态代理 | `MediaPipe Tasks` 本地模型 | CPU 友好，适合边缘端；输出只用于本地聚合，不释放关键点序列 |
| 局部运动 / 交互突增 | OpenCV frame difference；后续可升级 ROI optical flow | 当前 frame diff 简单可审计；光流可提高设备区域活动和混淆区分 |
| phase / trace 构建 | deterministic finite-state machine + hysteresis smoothing + transition scoring | 数据量小、隐私敏感，先用可解释规则比训练黑盒序列模型更稳 |
| schema 校验 | 后续建议引入 `pydantic` 或 `jsonschema` | 论文主线是 schema-first evidence，必须保证输出字段稳定可验证 |
| 文本生成 | 模板化 renderer + allowlist vocabulary | 避免 free-form caption 带来场景、外貌和身份线索 |
| LLM baseline | 只读 text evidence；provider-agnostic JSON 输出 | 保持 MDU-RiskBench 的 text-only 设定，可比较 GPT / Claude / 开源 LLM |

### 12.2 可作为后续升级的技术

- `PyAV / FFmpeg`：提升长视频解码稳定性，也更容易接 streaming non-retention。
- ROI-level optical flow：比简单 frame diff 更能区分设备区域交互和全局/姿态运动。
- 小规模内部设备框标注 + YOLO fine-tuning：如果通用 YOLO 对手机/平板/屏幕漏检明显，可以作为后续增强。
- `ruptures` 等 change-point detection：可作为 phase segmentation 的消融，但不建议一开始就作为主线。
- MediaPipe blendshapes / face action proxies：只能作为非常保守的可观察代理，不建议输出情绪标签。

### 12.3 暂时不建议作为主线的技术

- 直接 video captioning / Video-LLM：隐私风险高，且容易生成不可审计的自由场景描述。
- 发布视觉 embedding、face embedding、pose sequence：即使不是原图，也可能泄露身份或身体特征。
- 直接做“情绪识别”“疲劳识别”“成瘾识别”：伦理和有效性风险都较高，容易偏离 risk screening。
- GPU-only 大模型作为核心前处理：不符合边缘处理、低成本部署和可复现 benchmark 的主线。

### 12.4 当前更稳的技术组合

当前最稳的组合可以概括为：

```text
OpenCV / FFmpeg video IO
  -> YOLO11n + screen heuristic ROI
  -> MediaPipe hand / face / pose proxies
  -> ROI motion + quality estimation
  -> deterministic phase trace construction
  -> schema validation + template text rendering
  -> text-only LLM screening
```

这个组合的优点是：

- 工程上可落地；
- 可以在本地 / 边缘端运行；
- 输出可解释、可审计；
- 不需要发布图像、embedding 或高维姿态序列；
- 和 AAAI AISI 的 benchmark / responsible AI 主线一致。
