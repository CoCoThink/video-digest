# FireRedASR v2 本机跑通与多样本评估

> 日期：2026-07-20
> 目标：先在本项目的 4 GB 显卡和测试视频上跑通 FireRedASR v2，验证速度、显存、时间戳和字幕产物，再决定是否替换服务端默认 ASR。

## 1. 结论

`FireRedASR2-AED` 已在 10 分钟、95 分钟和 157 分钟三条项目视频上完整跑通，能直接生成词级时间戳，并通过现有排版层输出 SRT。

- 606.296 秒视频完整处理成功，21 个 30 秒分块，无 OOM。
- 模型推理 81.887 秒，推理 RTF 0.1351；含预处理、加载和后处理共 108.053 秒。
- GPU 常驻分配 2.345 GB，推理峰值分配 2.567 GB，适合 RTX 3050 4 GB。
- 输出 3098 字、2816 个带时间戳 token、91 条 cue；时间戳无逆序、重叠或零长度区间。
- 95 分钟方案设计视频上，FireRed 端到端 11:20，约为 Qwen 的 5.57 倍速；两者分窗加权字符一致度 93.46%。Qwen 的领域词和标点更稳，FireRed 也有明确胜例。
- 157 分钟视频的 316 个主块中有 2 块 decoder runaway，递归二分到 15/7.5 秒后恢复；95 分钟视频 191 块无重试。
- 两条长视频上 FireRed 端到端约为 Qwen 的 4.35–5.57 倍速，但速度优势不代表识别准确率更高。
- FireRedPunc 已在 95/157 分钟两条长视频上按 30 秒时间块完成验证，合计 507 个块均无字符投影降级；Punc RTF 为 0.00251–0.00271，语义字符保留率 100%。
- 综合结果支持“Qwen 默认质量模式 + FireRed 可选快速模式”，**不支持直接替换默认模型，也不支持任何一方全面超过另一方**。
- 服务端接入按启动级 backend 开关设计，详见 `docs/FireRedASR-v2-接入设计.md`。

产物：

- `Transformer其实是个简单的架构.fireredasr2-aed.srt`
- `Transformer其实是个简单的架构.fireredasr2-aed.txt`
- `Transformer其实是个简单的架构.fireredasr2-aed.items.json`
- `Transformer其实是个简单的架构.fireredasr2-aed.metrics.json`

## 2. 模型选择

FireRedASR2 有两个 ASR 变体：

| 变体 | 参数规模 | 本机判断 |
|---|---:|---|
| FireRedASR2-LLM | 8B+ | 4 GB 显存无法直接加载，不作为首次验证对象 |
| FireRedASR2-AED | 1.152B | 面向效率，原生支持词级时间戳和置信度；半精度可装入 4 GB GPU |

官方资料：[FireRedASR2S 仓库](https://github.com/FireRedTeam/FireRedASR2S)、[FireRedASR2-AED 模型](https://huggingface.co/FireRedTeam/FireRedASR2-AED)。官方源码固定到本次验证使用的提交：

```text
4e7d9aaf4482a47cec1724807026b9b151926eb5
```

## 3. 环境与安装

| 项目 | 值 |
|---|---|
| GPU | NVIDIA RTX 3050 Laptop，4096 MiB |
| Python | 3.11.15（复用 `qwen-asr` 环境） |
| PyTorch / CUDA | 2.9.1+cu128 / CUDA 12.8 |
| torchaudio | 2.9.1+cu128 |
| 模型 checkpoint | 4.73 GB，1,152,521,275 参数，FP32 存储 |
| 推理 dtype | BF16（仅强制对齐输入转 FP32） |

本次 `model.pth.tar` 的 SHA-256：

```text
4677cbd30988d63ed3e777f6a42a1e5260a3865317f6e15e488bef40954f7054
```

复用现有环境时，只需补充官方 AED 的轻量依赖：

```bash
QP=/home/guan/mamba/envs/qwen-asr/bin/python
$QP -m pip install \
  cn2an==0.5.23 kaldiio==2.18.1 kaldi_native_fbank==1.15 textgrid==1.6.1

git clone https://github.com/FireRedTeam/FireRedASR2S.git /path/to/FireRedASR2S
git -C /path/to/FireRedASR2S checkout 4e7d9aaf4482a47cec1724807026b9b151926eb5

$QP -m pip install 'modelscope>=1.33,<2'
/home/guan/mamba/envs/qwen-asr/bin/modelscope download xukaituo/FireRedASR2-AED
```

本机模型目录：

```text
/home/guan/.cache/modelscope/models/xukaituo--FireRedASR2-AED/snapshots/master
```

## 4. 关键兼容性问题：BF16 时间戳静默退化

首次 30 秒验证时正文识别成功，但官方时间戳路径抛出：

```text
RuntimeError: log_probs must be float64, float32 or float16 (half) type
```

原因链路：

1. RTX 3050 被 `torch.cuda.is_bf16_supported()` 判定支持 BF16，官方 `use_half=True` 因而把模型和输入转成 BF16。
2. AED 的 CTC logits 也成为 BF16。
3. `torchaudio.functional.forced_align` 不接受 BF16。
4. 官方实现捕获异常后把 `timestamp` 置空，上层又按音频总时长均匀摊分 token 时间；调用仍返回成功，因此容易被误认为是真实对齐。

项目脚本在调用 `forced_align` 前仅把 `log_probs` 转为 FP32，ASR 编码/解码仍用 BF16。修复后 30 秒片段得到随语音变化的真实时间区间，正文与置信度不变。

## 5. 完整视频结果

运行命令：

```bash
QP=/home/guan/mamba/envs/qwen-asr/bin/python

$QP fireredasr2_to_srt.py \
  --input 'Transformer其实是个简单的架构.mkv' \
  --out 'Transformer其实是个简单的架构.fireredasr2-aed.srt' \
  --source-dir /path/to/FireRedASR2S \
  --model-dir /home/guan/.cache/modelscope/models/xukaituo--FireRedASR2-AED/snapshots/master \
  --chunk-sec 30 \
  --beam-size 1
```

### 5.1 性能

| 指标 | FireRedASR2-AED beam 1 |
|---|---:|
| 音频时长 | 606.296 s |
| 分块 | 21（30 s，末块 6.296 s） |
| 预处理 | 2.359 s |
| 模型加载 | 17.361 s |
| 推理 | 81.887 s |
| 推理 RTF | 0.1351 |
| 端到端总耗时 | 108.053 s |
| 模型常驻分配显存 | 2.345 GB |
| 推理峰值分配显存 | 2.567 GB |
| 首块 / 后续 30 s 块 | 5.847 s / 2.8–5.2 s |

一次独立 30 秒冷启动探测的进程最大 RSS 为约 9.76 GB。checkpoint 以 FP32 保存，加载时会同时经历 checkpoint 和模型对象占用；服务化前需把主机内存峰值纳入部署约束。

### 5.2 时间戳与字幕结构

| 检查 | 结果 |
|---|---:|
| 带时间戳 token | 2816 |
| 非法区间（end <= start） | 0 |
| start 逆序 | 0 |
| token 重叠 | 0 |
| cue 数 | 91 |
| cue 最短 / 最长 | 4.6 s / 7.0 s |
| < 1 s / > 7 s cue | 0 / 0 |
| cue 重叠 | 0 |

30 秒硬边界处存在约 0.1–0.5 秒自然空隙，但没有时间戳倒退或重叠。当前脚本按 PCM frame 精确切块并累加 offset，不会累计浮点切块误差。

FireRedASR2-AED 本身不输出标点。只复用项目 `server.pipeline.layout` 时虽然能满足时长、CPS 和折行约束，但 32 条 cue 触及 7 秒上限，语义断句观感不如“有标点全文 + 排版层”的 Qwen 管线。后续长视频验证确认 FireRedPunc 能显著改善该问题，FireRed 部署建议同时启用 CPU Punc。

## 6. 初步质量对比

没有人工逐字稿，不能给出可靠 CER；这里只沿用 Qwen 实测报告中的高频术语做定向检查。

| 术语统计 | FireRed AED beam 1 | Qwen3 1.7B int8（既有报告） |
|---|---:|---:|
| `词向量` / `磁向量` | 10 / **8 错** | 18 / 0 |
| `点积` / `点击` | 0 / **4 错** | 3 / 1 |
| `依次` / `一四` | 2 / 0 | 2 / 0 |
| 全文字符数 | 3098 | 3369 |

官方推荐示例使用 beam 3，因此又对 60–150 秒术语密集片段做了 A/B：

| 指标（90 秒片段） | beam 1 | beam 3 |
|---|---:|---:|
| `词向量` / `磁向量` | 0 / 6 | 1 / 5 |
| `点积` / `点击` | 0 / 4 | 0 / 4 |
| RTF | 约 0.12 | 0.2055 |
| 峰值分配显存 | 2.567 GB | 2.567 GB |

beam 3 明显变慢，只修正 1 处“磁向量”，没有改善“点击”，不值得作为本机默认值。

## 7. 与现有 Qwen 管线的工程对比

| 维度 | FireRedASR2-AED | Qwen3-ASR-1.7B int8 |
|---|---|---|
| 4 GB GPU | 可用，峰值 2.567 GB | 可用，峰值约 3.02 GB |
| 606 s 总耗时 | 108.1 s | 约 382 s |
| 时间戳 | AED CTC 内置，GPU 上完成 | 独立 0.6B ForcedAligner，CPU 约 75 s |
| 标点 | AED 原始无标点；已接入 FireRedPunc | ASR 全文有标点 |
| 本视频术语 | 较差 | 较好 |
| 服务端接入成本 | 需新增 backend、切块和依赖 | 已完成并稳定运行 |

两套计时并非严格同版本同实现的基准，但量级差异明确：FireRed AED 的速度和显存很有吸引力；FireRedPunc 已解决工程上的标点恢复问题，当前主要差距是本视频的领域词准确度，而不是可运行性。

## 8. 本轮代码范围与下一步

新增 `fireredasr2_to_srt.py`，功能包括：

- 音频/视频统一转 16 kHz 单声道 PCM；
- 长音频按 PCM frame 精确切块；
- 模型只加载一次，顺序处理分块并累加时间偏移；
- BF16 forced-align 兼容修复；
- 输出 SRT、TXT、items JSON 和 metrics JSON；
- 复用现有字幕排版层。

当前实现状态：

1. FireRedPunc 已完成服务接入；95/157 分钟全文标点 RTF 分别为 0.00271/0.00251，语义字符保留率均为 100%。
2. `server.pipeline.asr` 已抽象为 Qwen/FireRed backend，并通过启动配置切换；默认值仍保持 Qwen。
3. 后续需要准备人工逐字稿或至少 3–5 条不同领域样本，计算 CER/WER，避免由现有视频直接决定默认模型选型。

## 9. 2 小时 37 分长视频对比

测试输入与参照字幕：

```text
test/test.mp4          9458.159 s（02:37:38）
test/test-current.srt  修复前的 Qwen 产物（含 layout 字符映射缺陷）
```

FireRed 使用与短视频相同的 AED BF16、beam 1、30 秒分块配置。原始运行产物为 `test/test-fireredasr2-aed.*`；发生 decoder runaway 的两个分块经递归二分重跑后，最终对比产物为：

```text
test/test-fireredasr2-aed.repaired.srt
test/test-fireredasr2-aed.repaired.txt
test/test-fireredasr2-aed.repaired.items.json
test/test-fireredasr2-aed.repaired.metrics.json
```

### 9.1 FireRed 性能与稳定性

| 指标 | 结果 |
|---|---:|
| 音频时长 | 9457.941 s |
| 30 秒主分块 | 316 |
| 预处理 | 13.021 s |
| 模型加载 | 10.835 s |
| 原始推理 | 1077.784 s（17:57.8） |
| 原始推理 RTF | 0.1140 |
| 原始端到端 | 1106.073 s（18:26.1） |
| GPU 常驻 / 推理峰值 | 2.345 / 2.567 GB |
| 修复后全文 / token | 40,752 字 / 38,744 token |
| 修复后 cue | 1358 |

长音频暴露了短样本没有覆盖到的 decoder runaway：

- 第 264 块（约 `02:11:30`）输出 750 token，尾部持续重复“嗯”。
- 第 315 块（约 `02:37:00`）输出 750 token，持续重复“好的”。
- 两块都因 `targets length is too long for CTC` 对齐失败，官方实现随后静默生成均匀时间戳。
- 第 264 块拆为两个 15 秒子块后恢复为 129 token；第 315 块处于视频收尾低语音区，其中一个 15 秒子块仍异常并继续拆为 7.5 秒，最终恢复为 105 token。两块均无重复和对齐异常。

脚本现已增加异常检测与递归二分重试：token 密度异常、无时间戳或 95% 以上等长的均匀时间戳都会废弃当前结果并拆半，最小到 7.5 秒；最小块仍异常则明确失败，不再接受伪时间戳。用连续 PCM 精确复现并验证了完整自动重试路径，相对原始长跑只增加约 4.57 秒推理；按同一常驻模型估算，带自动修复的本次端到端耗时约 18 分 31 秒。

### 9.2 覆盖与时间轴（含 2026-07-20 更正）

| 指标 | Qwen 修复前 | Qwen 修复后 | FireRed 修复版 |
|---|---:|---:|---:|
| cue | 633 | 2750 | 1358 |
| 对齐字符 | 9139 | 37,784 | 40,245 |
| 首 / 末时间 | 0.08 / 9355.38 s | 0.08 / 9456.20 s | 0.22 / 9456.75 s |
| cue 显示时间合计 | 1721.3 s | 7911.4 s | 9084.2 s |
| 视频时间覆盖率 | 18.2% | 83.6% | 96.0% |
| > 5 秒空档 | 222 | 10 | 2 |
| > 30 秒空档 | 90 | 1 | 0 |
| 最大空档 | 154.5 s | 30.1 s | 11.0 s |
| 非法 cue | 0 | 0 | 0 |

初次对比把 Qwen 的后处理缺陷误判为模型长音频退化。保留的 Qwen
`items.json` 实际包含 42,319 字正文、37,784 个对齐字符，最后一个 token 在
`9455.763s`；模型和分块偏移都覆盖到了视频结尾。真正根因是旧版
`layout.align_chars()` 用不完整的固定标点表过滤正文，首次遇到 `A.I.` 的点号后
错误消费 token 并永久失步，导致后续绝大多数正文因没有时间戳而被字幕层丢弃。

改为完整复刻 Qwen 对齐器的 Unicode 清洗规则后，37,784 个字符全部映射成功，
无需重跑模型即可重建 `test/test-qwen3-asr-1.7b-int8.fixed.srt`。Qwen 与 FireRed
在全部 316 个 30 秒窗口上的加权字符一致度为 0.828，中位数为 0.840，未观察到
随视频时长持续下降或时间轴漂移。

### 9.3 识别质量

两者在全片时间轴上整体内容接近，但各有所长。没有人工逐字稿，以下判断只针对语义明显或专有名词有已知拼写的样本。

FireRed 明显更好的例子：

| 当前字幕 | FireRed | 判断 |
|---|---|---|
| `人件制件制` | `仁者见仁，智者见智`（无标点输出） | FireRed 正确 |
| `有的小伙伴说是老鼠` | `有的小伙伴说是我的老粉儿` | FireRed 更符合语义 |
| `整个系架构变复杂` | `整个系统架构变复杂` | FireRed 正确 |

当前字幕更好的例子：

| 当前字幕 | FireRed | 判断 |
|---|---|---|
| `context engineering` | `contact engineering` | 当前正确 |
| `Harness` 的近似音译 | 多次识别为 `哈马斯`、`哈喇子`、`哈达机器` | FireRed 专名不稳定 |
| 带标点、按语义分句 | 无标点，主要按 7 秒上限硬切 | 当前字幕观感更好 |

FireRed 修复版的平均分块置信度为 0.856、中位数 0.862；312 个正常 30 秒块中只有 2 块低于 0.7。专业词频也因覆盖完整而更多，例如 `Agent` 22 次、`智能体` 13 次、`模型` 42 次、`MCP` 5 次，但词频不能直接视为准确率。

### 9.4 字幕可读性

Qwen 修复版有标点且短句自然。FireRed AED 原始输出无标点，仅靠排版层时
804/1358 条 cue 触及 7 秒上限。对同一份 repaired items 按 316 个 30 秒时间块
运行 FireRedPunc 后，新增 3,298 个标点，cue 增至 2204，触及 7 秒上限的 cue 降至
152，超过 9.5 CPS 的 cue 为 14 条；字符投影降级、非法 cue 和重叠均为 0。
标点推理耗时 23.718 秒，RTF 0.00251。可读性短板已明显收窄，但标点模型不会
修正领域词识别错误。

长视频上的综合结论：

1. **覆盖和时间轴：两者都覆盖到结尾。** FireRed cue 显示覆盖率更高，但 Qwen 修复后的 37,784 个对齐字符 100% 进入字幕，不存在长视频漂移或后半段漏识别。
2. **正文准确度：大体同一量级。** 全片分窗字符一致度约 0.83，FireRed 修正常见中文短语更好，但英文专名仍有明显错误；没有人工稿，不能据此判定总体 CER 胜负。
3. **稳定性：FireRed 需要兜底。** 316 个块中 2 个发生 decoder runaway；15 秒二分重试可恢复，脚本已实现自动检测和重试。
4. **速度与观看体验各有取舍。** FireRed 约 18.5 分钟完成，明显快于 Qwen 的 80.5 分钟；启用 CPU FireRedPunc 仅增加约 24 秒推理，并显著减少按 7 秒上限硬切的字幕。

更正后的建议是：继续保留 Qwen3-ASR-1.7B int8 为默认高质量方案，把
FireRedASR2-AED + CPU FireRedPunc 作为可选快速 backend；用人工校对片段计算 CER
后，再决定是否调整默认值。

## 10. 95 分钟方案设计视频对比

测试输入：

```text
test/方案设计域AI实践分享.mp4   5714.155 s（01:35:14）
```

两种模型均使用 16 kHz 单声道 PCM、30 秒分块和修复后的同一套字幕布局参数。
Qwen 使用服务端实际的 int8 GPU / ForcedAligner CPU 重叠流水线；FireRed 使用
AED BF16、beam 1 和异常块递归二分。

### 10.1 性能、覆盖与稳定性

| 指标 | Qwen3-ASR-1.7B int8 | FireRedASR2-AED | FireRed + Punc |
|---|---:|---:|---:|
| 端到端耗时 | 3790.096 s（63:10） | 680.158 s（11:20） | 约 695.621 s（11:36）¹ |
| 转写 RTF | 0.6580 | 0.1147 | 0.1147 + 0.00271 |
| 峰值显存 | 2.649 GB | 2.567 GB | 2.567 GB（Punc 在 CPU） |
| 对齐字符 | 25,370 | 25,496 | 25,496 |
| token / cue | 25,029 / 1572 | 25,235 / 825 | 25,235 / 1372 |
| 7 秒裁剪 cue | 19 | 347 | 87 |
| 最后 token | 5706.337 s | 5707.265 s | 5707.265 s |
| 异常 / 重试 / Punc 降级 | 0 | 0 | 0 / 0 / 0 |

¹ 在 AED 实测端到端时间上叠加 Punc 推理 15.463 秒；Punc 一次性加载另需
5.998 秒，常驻服务后按任务摊销。

FireRed 原始端到端约为 Qwen 的 5.57 倍速；叠加 Punc 推理后仍约为 5.45 倍速。
两者均完整覆盖到视频结尾，191 个 30 秒
窗口中不存在只有一方有正文的整窗漏识别；全文顺序相似度 0.9367，分窗加权
相似度 0.9346、中位数 0.9389。前四分之一与后四分之一相似度分别为 0.9277 和
0.9453，没有随视频时长下降或时间轴漂移。

FireRed 原始 cue 显示覆盖率更高主要来自无标点文本被延长到 7 秒上限，不代表
识别内容更多：两者对齐字符只差 0.5%。启用 Punc 后新增 1,983 个标点，字符投影
降级为 0，7 秒裁剪 cue 从 347 降至 87。Qwen 归一化正文与 token
25,370/25,370 完全一致，本次没有后处理丢字。

### 10.2 内容抽样与结论

没有人工逐字稿，不能给出可靠 CER。语义明确的样本显示 Qwen 对“组件、设计域、
回调、系统功能树、需求实例化、MCP”等核心词更稳定；FireRed 经常把“组件”识别
为“组建/主键”、把“回调”识别为“毁掉”。FireRed 也正确识别了 Qwen 出错的
“应用的架构”“产物是一个 skill”“衔接到详设”。

这条视频进一步支持当前选型：Qwen 保持默认；FireRedASR2-AED + CPU FireRedPunc
作为显著更快、带异常重试保护的可选 backend。逐项数据和例句见
`docs/方案设计域AI实践分享-ASR对比报告.md`。
