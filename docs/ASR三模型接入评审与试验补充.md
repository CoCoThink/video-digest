# ASR 三模型接入评审与试验补充

| 项目 | 内容 |
|---|---|
| 版本 | v1.0 |
| 日期 | 2026-07-21 |
| 状态 | Qwen 默认、FireRed 快速模式已实现；FunASR 实验模式已接入，尚未达到稳定级准入门槛 |
| 范围 | 评审原有 Qwen/FireRed 实现，验证第三候选模型，并记录本轮修复与选型结论 |

## 1. 结论

本轮没有证据支持直接替换默认模型。当前部署分层如下：

| backend | 定位 | 当前决定 |
|---|---|---|
| `qwen` | 默认质量模式 | 保持默认；领域词和成品字幕质量已有长视频证据，资源和失败模式已知 |
| `firered` | 可选快速模式 | 保留；推荐同时启用 CPU FireRedPunc，适合优先吞吐的独立服务实例 |
| `funasr` | 实验模式 | 代码接入可用，但 4 GB 卡显存余量小，且缺少人工真值、长视频和许可审查，不进入默认/稳定承诺 |

三种模式共用同一 Worker、任务状态、`items.json`、排版和下载接口，但一个服务实例
只常驻一个 backend。切换需要修改 `VD_ASR_BACKEND` 并重启，不支持前端逐任务选择。

## 2. 原有两套实现评审与修复

### 2.1 Qwen

既有 GPU ASR / CPU ForcedAligner 容量 1 流水线设计正确：它复用 qwen-asr 的分块、
解析、offset 和合并语义，仅重叠独立的 GPU/CPU 工作。120 秒 A/B 中产物完全一致，
耗时从 119.366 秒降至 84.337 秒，峰值分配显存未增加。

本轮发现并修复两处接口完整性问题：

1. 公共 backend 接口虽然需要领域上下文，Worker 到 Qwen `_infer_asr` 之间此前没有
   完整透传。现在 `VD_ASR_CONTEXT` 去除首尾空白后传给每个音频块。
2. Qwen 产物此前主要依赖排版层发现正文/token 错位。现在 ASR 返回后立即执行公共
   契约校验，异常以 `ASR_OUTPUT_MISMATCH` 失败，不生成残缺字幕。

保留限制：Qwen 在 4 GB 卡上的资源余量有限，默认块长继续保持 30 秒；ForcedAligner
固定 CPU，不能与 ASR 一起搬到 4 GB GPU。

### 2.2 FireRed

原有 AED 接入的 BF16 forced-align 兼容、decoder runaway 检测、递归二分和 Punc
字符投影设计成立，95/157 分钟长视频也验证了完整覆盖。不过仍有四个工程风险：

1. **源码不可复现**：仅配置源码路径不足以保证行为。本轮强制目录为 revision
   `4e7d9aaf4482a47cec1724807026b9b151926eb5` 的干净 git checkout，并检查 checkpoint
   必需文件；不满足时 Worker 不 ready。
2. **硬边界切词**：固定 30 秒可能落在语音中间。现在目标切点前默认搜索 5 秒内的
   最低能量位置，仍按 PCM frame 连续覆盖，块长不会超过上限。该策略不是 VAD。
3. **能力语义不清**：AED 当前接口不支持强制语言或可靠 LID。强制语言现在明确
   拒绝，自动模式记录 `Unknown`；context 非空时记录 `ASR_CONTEXT_UNSUPPORTED`。
4. **校验分散**：FireRed 的局部异常检测保留，同时最终结果进入与 Qwen/FunASR
   相同的正文/items/绝对时间戳校验。

这些修复提高可复现性和失败可见性，不改变 2026-07-20 历史报告中的性能数字。

## 3. 第三模型选择与接入

本轮选择 `FunAudioLLM/Fun-ASR-Nano-2512` 作为候选，原因是它能在目标 4 GB CUDA
环境中加载，原生返回标点与字符时间戳，并提供 hotwords 接口，能够适配现有
`items.json -> layout -> SRT/TXT` 契约。实现位于
`server/pipeline/backends/funasr.py`，依赖固定于 `requirements-funasr.txt`。

接入行为：

- `AutoModel` 固定 `device="cuda:0"`，关闭版本检查、进度条和冗余日志；
- 复用 FireRed 的 PCM 精确切块与低能量边界搜索；
- `VD_ASR_CONTEXT` 按中英文逗号、分号和换行拆成 hotwords，去空并保持首次出现顺序；
- `VD_ASR_LANGUAGE` 非空时透传给模型；未强制时返回 `Unknown`，不虚构 LID；
- 丢弃纯标点 timestamp item，由显示正文保留标点，语义字符进入公共契约校验；
- CUDA OOM 清缓存后整任务重试一次，第二次仍失败则任务失败；
- `/api/config` 下发 FunASR RTF 和标点能力，`/api/health` 暴露实际 model ID。

## 4. 试验结果与可比性

硬件为 NVIDIA GeForce RTX 3050 Laptop GPU 4 GB。FunASR 的本轮服务路径试验为
180 秒音频：

| 指标 | Fun-ASR-Nano-2512 |
|---|---:|
| 音频时长 | 180 秒 |
| 转写 RTF | 0.122 |
| 峰值分配显存 | 3.755 GB |
| 标点 | 模型原生 |
| 时间戳 | 模型原生字符时间戳，经公共契约校验 |

这次结果只证明“当前环境能完整运行并产出合法字幕数据”。它不能与下面的 95 分钟
同片结果直接比较，因为样本、运行长度和测试日期不同：

| 同一 95 分钟样本历史结果 | Qwen3-ASR-1.7B int8 | FireRedASR2-AED |
|---|---:|---:|
| 转写 RTF | 0.6580 | 0.1147 |
| 峰值分配显存 | 2.649 GB | 2.567 GB |
| 主要结论 | 领域词、英文缩写和成品断句更稳 | 约 5.57 倍端到端速度；加 CPU Punc 后仍约 5.45 倍 |

FunASR 的 3.755 GB 峰值已接近 4 GB 物理上限，不能从一次 180 秒成功推断小时级文件
不会 OOM。当前也没有同片人工逐字稿，不能宣称其 CER 优于 Qwen 或 FireRed。

## 5. 统一契约与失败策略

所有 backend 必须返回：

```json
{
  "text": "带显示标点的全文",
  "language": "模型返回的语言标签或 Unknown",
  "items": [{"t": "词", "s": 1.23, "e": 1.56}],
  "backend": "qwen|firered|funasr",
  "model_id": "实际模型标识",
  "warnings": []
}
```

公共校验在 `items.json` 落盘前执行：

- `text/items` 类型和字段完整；
- 按 Unicode 字母、数字、英文撇号归一化后字符完全一致；
- 时间戳可转为有限浮点数，单项 `end >= start`；
- 绝对时间单调，且不超出音频时长 0.25 秒容差。

违反契约统一抛 `AsrOutputError`，任务状态记 `ASR_OUTPUT_MISMATCH`。FireRed 的局部
runaway 检测和 Punc fail-open 属于 backend 内部恢复，不替代最终公共校验。

## 6. 配置与依赖

公共/Qwen 环境固定在 `requirements.txt`；FireRed 和 FunASR 分别通过
`requirements-firered.txt`、`requirements-funasr.txt` 叠加。关键配置如下：

| 配置 | Qwen | FireRed | FunASR |
|---|---|---|---|
| backend 值 | `qwen` | `firered` | `funasr` |
| 模型路径 | `VD_ASR_MODEL_PATH` + `VD_ALIGNER_MODEL_PATH` | `VD_FIRERED_MODEL_PATH` | `VD_FUNASR_MODEL_PATH` |
| 源码路径 | 无 | `VD_FIRERED_SOURCE_DIR`，必填且校验 revision/clean | 无 |
| context | 自由文本，每块透传 | 不支持，记录 warning | 分隔并去重为 hotwords |
| language | 自动或提示 | 仅自动，元数据 `Unknown` | 自动或提示；自动元数据 `Unknown` |
| 边界搜索 | qwen-asr 内部逻辑 | `VD_FIRERED_BOUNDARY_SEARCH_SEC=5` | `VD_FUNASR_BOUNDARY_SEARCH_SEC=5` |
| ETA RTF | `VD_RTF=0.65` | `VD_FIRERED_RTF=0.12` | `VD_FUNASR_RTF=0.13` |

完整启动命令和全部环境变量见 `server/README.md`。

## 7. 验证与准入门槛

本轮自动测试共 37 项通过，覆盖 factory、Qwen context、公共契约、PCM frame 连续性、
低能量边界、FireRed 异常二分/Punc、FunASR 结果适配、上传和排版回归；另完成默认
Qwen 的 30 秒完整 API 烟雾测试。

FunASR 从 experimental 升为稳定可选 backend 前必须全部完成：

1. 用 20–50 个代表性文件覆盖中文、英文、中英混合、会议、多说话人、噪声和小时级音频；
2. 制作人工逐字稿并计算 CER/WER、领域词召回率，和 Qwen/FireRed 做同片同参数比较；
3. 在 4 GB 卡上记录各类文件峰值显存、OOM 重试和 Worker 存活，至少完成 3 小时上限样本；
4. 核对分块边界附近的漏字/重复字，并验证 hotwords 的收益与误触发；
5. 审阅 FunASR 项目及模型的独立许可，确认目标部署和分发方式合规；
6. 只有质量、稳定性和许可均通过后，才讨论 UI 展示或默认模型变更。

因此当前推荐配置仍是：质量优先使用 Qwen；吞吐优先的独立实例使用
FireRedASR2-AED + CPU FireRedPunc；FunASR 仅用于受控试验。
