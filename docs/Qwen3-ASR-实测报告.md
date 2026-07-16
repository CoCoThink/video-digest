# Qwen3-ASR 本机部署与实测报告

> 目标：在本机 4 GB 显卡上跑通 Qwen3-ASR（GPU），从视频提取字幕，并对比 0.6B / 1.7B / int8 等方案的速度、资源、准确性。
> 视频：`Transformer其实是个简单的架构.mkv`（时长 606 s ≈ 10 min）。

---

## 0. 环境

| 项目 | 值 |
|---|---|
| 平台 | WSL2 (Linux 6.18) |
| GPU | NVIDIA RTX 3050 Laptop，**4096 MiB (4 GB)**，compute cap **8.6**（Ampere）|
| 驱动 / CUDA | Driver 592.27 / CUDA 13.1 |
| 环境管理 | mamba，env 名 `qwen-asr` |
| Python | 3.11.15 |
| 关键包 | torch 2.9.1+cu128、qwen-asr 0.0.6、transformers 4.57.6、vllm 0.14.0、bitsandbytes 0.49.2、jieba 0.42.1（后两者本次装）|
| 已缓存模型 | Qwen3-ASR-0.6B、Qwen3-ASR-1.7B、Qwen3-ForcedAligner-0.6B |
| ffmpeg | 8.1.2 |

**核心约束：4 GB 显存**。这是贯穿全程的瓶颈——它决定了模型选型、分块大小、量化与卸载策略。

---

## 1. 跑通字幕提取（基础方案）

### 1.1 抽取音频
```bash
ffmpeg -y -i "Transformer其实是个简单的架构.mkv" -vn -ac 1 -ar 16000 -c:a pcm_s16le audio_16k.wav
```
得到 16 kHz 单声道 wav，606 s。

### 1.2 关键坑：音频编码器分块过大导致 OOM
- 第一次用 0.6B ASR + 0.6B 对齐器（都在 GPU），**权重装得下**（约 1.9 GB），但转写时 **OOM 在音频编码器的 conv2d**。
- 原因：带时间戳时库默认按 `MAX_FORCE_ALIGN_INPUT_SECONDS = 180` 秒分块，180 s 音频进编码器的激活值撑爆 4 GB。
- **解决**：monkeypatch 把分块降到 **30 s**，激活骤降，显存稳定在 ~2.1 GB，OOM 消失。

### 1.3 结果（0.6B）
- 峰值显存 **1.91 GB**，自动识别 Chinese，3398 字，146 条字幕，全程 GPU。
- 产物：`Transformer其实是个简单的架构.qwen3-asr.0.6b.srt`

### 1.4 关键 API 认知
- `Qwen3ASRModel.from_pretrained(...)`（transformers 后端）/ `.LLM(...)`（vLLM 后端）
- `transcribe(audio, context='', language=None, return_time_stamps=False)`
- 时间戳来自 `ForcedAlignResult.items[i].{text, start_time, end_time}`（单位**秒**），字/词级，长音频自动分块合并。
- **字级 token 不含标点**（所以 SRT 无标点），但 `text` 全文**含标点**。

---

## 2. 0.6B vs 1.7B 对比

### 2.1 1.7B 在 4 GB 上装不下
- 1.7B 的 bf16 权重 ≈ **3.4 GB**，加 CUDA 上下文，`device_map="cuda:0"` 直接加载即 **OOM**。
- 对齐器再叠 0.6B（~1.2 GB）更不可能同时在 GPU。
- **解决**：accelerate **CPU 分层卸载**（`device_map="auto"` + `max_memory` 上限），全精度 bf16，溢出层放内存；对齐器放 CPU。为公平，两模型用相同管线（对齐器 CPU、分块 30 s）。

### 2.2 指标对比（相同管线）

| 指标 | 0.6B | 1.7B (bf16) |
|---|---|---|
| 运行方式 | 全 GPU | GPU + CPU 卸载 |
| 峰值显存 | **1.91 GB** | **3.39 GB** |
| 加载 | 14.2 s | 14.4 s |
| ASR 解码 | **75 s** | **603 s** |
| 对齐(CPU) | ~73 s | ~73 s |
| 总耗时 | **149 s (2.5 min)** | **677 s (11.3 min)** |
| 转写字数 / 字幕条 | 3398 / 146 | 3373 / 145 |

> ⚠️ 1.7B 的 603 s 被 **CPU 卸载**严重拖累，不代表纯 GPU 速度。

### 2.3 准确性：1.7B 明显更好（尤其专业术语）
整体相似度 96.5%，差异集中在**同音术语**：

| 正确词 (错词) | 0.6B | 1.7B (bf16) | Whisper large-v3(参考) |
|---|---|---|---|
| 词向量 (磁向量) | 15 / **2 错** | 18 / 0 | 3 |
| 点积 (点击) | 0 / **4 错** | 3 / 1 | 0 / **5 错** |
| 依次 (一四) | 1 / **1 错** | 2 / 0 | 2 |

例：`磁向量`→`词向量`、`做点击`→`做点积`、`一四和K三`→`依次和K3`。
**注意**：Whisper large-v3 在「点积」上全错（5 次点击），比两个 Qwen 都差。

---

## 3. 1.7B 是否跑满了显存？—— 没有，且可优化

- 第一次手动设 `--gpu-mem-mib 2800`，很保守：峰值仅 **3.39 / 4.0 GB**，还把 ~0.6 GB 权重白放 CPU；权重在 CPU → 每生成一个 token 都要 CPU↔GPU 搬数据，是慢的主因之一。
- **机制**：accelerate 的 `max_memory` **只限"权重"放多少上 GPU**；CUDA 上下文（WSL 下 ~0.4–0.5 GB）、每 token 激活、加载瞬时峰值都叠加在其上、不计入。

### 3.1 调高上限后（cap 3100 MiB + 分块 15 s）

| 配置 | 显存上限 | 峰值 | ASR 解码 | 总耗时 |
|---|---|---|---|---|
| 原始 | 2800 MiB | 3.39 GB | 603 s | 677 s (11.3 min) |
| **调优** | 3100 MiB | **3.69 GB** | **411 s** | **500 s (8.3 min)** |

**显存利用 3.39→3.69 GB，ASR 解码快 32%，总耗时省 ~3 min，准确度不变。**

- 试 `3350 MiB` → **OOM**。天花板约：权重上限 ~3.1 GB / 峰值 ~3.7 GB。
- **1.7B 的 3.4 GB 权重物理上无法整块进 4 GB**，总有 ~0.3 GB 要留 CPU。要彻底进 GPU 只能靠量化或换卡。

---

## 4. int8 量化 vs 0.6B —— int8-1.7B 综合更好

### 4.1 关键修复
int8 首次失败（bnb 的 `SCB` 属性 bug）。**修复：`device_map={"": 0}` 而非 `"cuda:0"`**，绕过 bnb 的 dispatch 检查。

### 4.2 int8-1.7B 结果（全程 GPU）

| 指标 | 值 |
|---|---|
| 峰值显存 | **3.02 GB**（全 GPU，无卸载）|
| ASR 解码 | **306 s**（比 bf16 卸载版 603 s 快一倍）|
| 对齐(CPU) | 75 s |
| 总耗时 | **382 s (6.4 min)** |
| 字数 / 字幕条 | 3369 / 145 |

### 4.3 int8 几乎无损
- **int8-1.7B vs 满血 bf16-1.7B 相似度 = 99.64%**，三个术语表现**完全一致**（词向量 18/0、点积 3/1、依次 2/0）。
- int8 保留了 1.7B 相对 0.6B 的全部准确度优势。

### 4.4 三方总表

| 方案 | 峰值显存 | 总耗时 | 准确度 | 运行方式 |
|---|---|---|---|---|
| **0.6B** | 1.91 GB | 149 s (2.5min) | 良好 | 全 GPU |
| 1.7B bf16(卸载) | 3.39 GB | 677 s (11.3min) | 最好 | GPU+CPU |
| **1.7B int8** | 3.02 GB | 382 s (6.4min) | **≈最好** | **全 GPU** |

**结论**：要准 → int8-1.7B（本机跑 1.7B 的最佳方式）；要快 → 0.6B。

---

## 5. 有没有"比 int8-1.7B 更好、又能在 GPU 上跑"的方案？

### 5.1 准确度：基本到顶
- int8 ≈ bf16（99.6%），反量化换不来精度；且 **1.7B 已是 Qwen3-ASR 开源系列最大**（0.6B/1.7B；更强的 Flash 仅云端 API）。
- **Context 提示实测无效**：喂领域术语后，术语错误没修好，反而把 `w_k/q1/v1` 退化成 `W/Q一/K一`（丢了 q/k/v 区分），改动 ~10%。→ 只对**特定生僻专有名词**有用，对本视频添乱。

### 5.2 三个提速/改进方向的实测结论

**① vLLM 后端 —— ❌ 4 GB 上跑不了 1.7B**
- int8/bnb：**失败**，`Qwen3ASRForConditionalGeneration does not support BitsAndBytes quantization`（模型缺 `packed_modules_mapping`，vLLM 0.14 不支持该架构 bnb 量化）。
- bf16：装不下（启动时仅 ~3.21 GB 空闲，权重 3.4 GB）。
- fp8：需 Ada/Hopper，3050 是 Ampere，不支持。
- （0.6B 可以用 vLLM 且更快，但那是降准确度换速度，不算"更好"。）

**② flash-attn —— ⚠️ 工具链可装但未跑通，且本场景收益低**
- 初始无 `nvcc`；已用 mamba（conda-forge + nvidia）装上 **CUDA toolkit 12.8** + conda **gcc 14.3**。
- 编译踩坑链：`CUDA_HOME` 未设 → `cuda_runtime_api.h` 在 `targets/x86_64-linux/include`（需加 `CPATH`）→ 系统 gcc 15 被 CUDA 12.8 拒（需 ≤14，改用 `-ccbin` 指向 conda gcc 14.3）。
- 最后一次编译在会话中断/OOM 时终止，**尚未装成**。
- **即便装成，本场景预期收益低**：int8-1.7B 解码是显存带宽瓶颈（反量化），flash-attn 主要加速注意力（占比小）；30 s 短分块获益有限；对齐器也无法挪回 GPU（int8-1.7B 已占 3 GB，无空间）。

**③ 字幕标点断句后处理 —— ✅ 已跑通，唯一稳赚的收益**
- 现状：原 SRT 按对齐器**字级 token**断句、**无标点**；`.txt` 全文**有标点**。
- `build_srt_punct.py`：把带标点全文对齐到有时间戳的 token 上，在句号/问号/逗号处自然断句（含一字前瞻，避免句末标点被孤立）。
- 用已有的 `items.1.7b-int8.json`（125 KB，含全文+逐 token 时间戳）纯 CPU 秒级生成 `...1.7b-int8.punct.srt`（50 条 cue），**无需 GPU**。
- 实测对比（前 12 条）：

  | 维度 | 原(字级 token) | 新(标点断句) |
  |---|---|---|
  | 断句 | 固定字数硬切，**词被切断**（"转换成一/个词向量"） | 句号/逗号处断，**语义完整** |
  | 标点 | 无 | 有（，。？） |
  | 英文 | 粘连（"Iloveyoubaby"） | 还原空格（"I love you baby"） |
  | 观感 | 机械 | 自然，接近人工字幕 |

- **结论**：这是三项杠杆里唯一确定见效、且零风险/零 GPU 的一项。最终推荐字幕产物即 `...1.7b-int8.punct.srt`。

**④ 加大分块 —— ⚠️ 有边际提速，已递减；默认改 45 s**
- 动机：int8 峰值只 3.02 GB，还剩近 1 GB 余量，试试加大分块能否提速。
- 实测（int8-1.7B，606 s 音频，对齐器在 CPU）：

  | 分块 | 峰值显存 | ASR decode | 对齐 | **总时长** | 文本字数 |
  |---|---|---|---|---|---|
  | 30 s（基线） | 3.017 GB | 296.9 s | 77.2 s | **375.2 s** | 3369 |
  | **45 s** | 3.017 GB | 285.4 s | 76.6 s | **363.3 s** | 3371 |
  | 90 s | 3.261 GB | 279.0 s | 82.4 s | **362.3 s** | 3348 |

- 30→45 s：省 **3.2%**，**峰值显存不变**（激活极小）、文本几乎一致（+2 字）；45→90 s：几乎不再省（对齐反而变慢），且峰值爬到 3.26 GB。
- 根因：瓶颈是**逐 token 自回归解码**，随音频**总长**走、与分块数无关；加大分块只省"边界重初始化"那点开销 → 天然递减。180 s 当年 OOM 是 **bf16 权重本就逼近上限**，与激活无关。
- **决定**：默认分块 30 s → **45 s**（`qwen3_asr_compare.py` / `qwen3_asr_to_srt.py` 已改），白捡 ~3%、零风险；再大不值得。
- **⚠️ 更正（服务端）**：此处"零风险"仅基于**单条 606 s 样本**——"峰值显存不变"不成立于任意内容。峰值是**内容相关**的：语速密集的块生成 token 多、激活更大，同一影片不同块峰值实测在 3.02↔3.26 GB 跳动，个别更高。一条 95 min 音频在 45 s 下某密集块越过 4 GB，触发驱动级 OOM 并致 Worker 硬退出（`WORKER_CRASHED`）。**故服务端默认已回退 30 s**（`server/config.py`，`VD_CHUNK_SEC` 可调）。本节结论仅适用于两个 CLI 脚本的短样本基准，不作为服务端长音频默认依据。详见《实现与踩坑》§4.8。

**⑤ 字幕排版规范化 —— ✅ 通用后处理，已完成**
- 在标点断句(③)之上，`build_srt_layout.py` 再套一层**行业通用**的字幕可读性规则（参数化，默认按中文；换任何中文视频可直接复用，非本视频定制）：
  1. 最短显示时长 `--min-dur`(默认 1.0 s) + **最长显示时长封顶** `--max-dur`(默认 7 s，超长即截断，防对齐空洞产生的滞留长条)
  2. 阅读速度上限 `--cps`(默认 9 字/秒)，不足则**只往后借空隙**延长、绝不移轴
  3. 每行字数 + **两行折行** `--line-width`(默认 18)，折点优先级 标点 > 拉丁/中文边界·空格 > 任意
  4. 合并过短碎片 `--merge-below`
  5. **禁止在词/公式内部断行**：
     - 拉丁/数字/下划线 token(护住 `w_k`、`Q1`、`RNN`、`QKV`)——含"上一字↔当前字""当前字↔下一字"双向保护
     - **中文词(jieba 分词)**——断句/折行都不落在词中间(消掉 `每/个`、`相/乘` 这类)，通用、不含任何领域词
  6. 消重叠 / 规整空隙
- 实测(前后对比)：

  | 指标 | 排版前 | 排版后 |
  |---|---|---|
  | <1 s 过短条 | 23 | **0** |
  | >9 字/秒读不完 | 19（最高 **219** 字/秒） | **0 条 >9.5**（残留仅 9.0 舍入） |
  | 最短 / 最长时长 | 0.08 s / 11 s | **1.0 s / 7.0 s** |
  | `w_k` 等公式被劈开 | 有 | **无** |
  | 中文词被劈开(每/个、相/乘) | 有 | **无** |
  | cue 数 | 48 | 47（更整） |

- **jieba 带出的洞察**：护住 `相乘` 后暴露出 `同理…相乘` 那段**对齐几乎全丢**(22 字里仅 3 字有时间戳，中间 6.4 s 空洞)——这正是"对齐时间戳精度"的实证；`--max-dur` 封顶把它从 11 s 截到 7 s 兜底。
- **局限**：②只能把显示时间拉到空隙上限；个别"字多、对齐 span 本就短"的段是**对齐时间戳精度**问题(公式密集处尤甚)，排版层治标不治本，脚本会主动打印被截断/仍偏快的条目。
- **依赖**：`jieba`(纯 Python，已装)；缺失时自动降级为无中文词约束、其余规则照常。
- **最终推荐字幕产物**：`...1.7b-int8.layout.srt`（47 条，词/公式完整、时长规范）。

### 5.3 小结
- **想更准**：到本机天花板了，int8-1.7B 最优；再往上要上云端 API 或换更大显存的卡。
- **想更快**：vLLM 跑不了 1.7B（架构不支持量化）；flash-attn 收益低且工具链麻烦；加大分块只有 ~3% 边际收益（已取，默认 45 s）。
- **想字幕更好看**：标点断句(③) + 排版规范化(⑤)，均**零风险、零 GPU、通用**，**已完成**，最终产物 `...1.7b-int8.layout.srt`。

---

## 6. 产物与脚本清单

### 脚本
| 文件 | 作用 |
|---|---|
| `qwen3_asr_to_srt.py` | 基础转写→SRT（`--chunk-sec` 控分块防 OOM）|
| `qwen3_asr_compare.py` | 对比用：计时/显存/落盘 SRT+txt+metrics；`--load-8bit`（int8）、`--gpu-mem-mib`（卸载）、`--context`、`--dump-items` |
| `build_srt_punct.py` | 标点感知断句，从 `items.*.json`+全文生成更自然 SRT |
| `build_srt_layout.py` | 在标点断句上加通用排版规则（最短/最长时长、CPS、折行、合并、拉丁+中文词(jieba)不劈开、消重叠），参数化、输出前后对比统计 |
| `test_int8_load.py` | int8 加载策略探测（确认 `device_map={"":0}` 可行）|
| `test_vllm.py` | vLLM 后端 fit-test（确认 bnb 不支持该模型）|

### 字幕/文本
- `*.qwen3-asr.0.6b.srt/.txt` — 0.6B
- `*.qwen3-asr.1.7b.srt/.txt` — 1.7B bf16 卸载（chunk30）
- `*.qwen3-asr.1.7b-hi.srt` — 1.7B bf16 卸载调优（cap3100/chunk15）
- `*.qwen3-asr.1.7b-int8.srt/.txt` — int8-1.7B 原始（字级 token 断句、无标点）
- `*.qwen3-asr.1.7b-int8-ctx.txt` — int8 + context（实验，无增益）
- `*.qwen3-asr.1.7b-int8.punct.srt` — int8-1.7B + 标点断句（③）
- `*.qwen3-asr.1.7b-int8.layout.srt` — **★ 最终推荐**：标点断句 + 排版规范化 + 中文分词（③⑤）
- `items.1.7b-int8.json` — 全文 + 逐 token 时间戳（供离线重排版，无需重跑 GPU）
- `metrics.*.json` — 各配置指标

### 复现命令
```bash
QP=/home/guan/mamba/envs/qwen-asr/bin/python

# 0.6B（最快）
CUDA_VISIBLE_DEVICES=0 $QP qwen3_asr_compare.py --audio audio_16k.wav --tag 0.6b \
  --asr Qwen/Qwen3-ASR-0.6B --aligner-device cpu --chunk-sec 30

# int8-1.7B（最准、全 GPU，推荐；--chunk-sec 现默认 45，可省略）
CUDA_VISIBLE_DEVICES=0 $QP qwen3_asr_compare.py --audio audio_16k.wav --tag 1.7b-int8 \
  --asr Qwen/Qwen3-ASR-1.7B --aligner-device cpu --load-8bit --chunk-sec 45

# 1.7B bf16 CPU 卸载（全精度，慢）
CUDA_VISIBLE_DEVICES=0 $QP qwen3_asr_compare.py --audio audio_16k.wav --tag 1.7b \
  --asr Qwen/Qwen3-ASR-1.7B --aligner-device cpu --gpu-mem-mib 3100 --chunk-sec 15

# 生成更自然的标点断句字幕（需先有 items.*.json）
$QP build_srt_punct.py --items items.1.7b-int8.json --out out.punct.srt

# 最终推荐：标点断句 + 通用排版规范化（零 GPU，任何中文视频通用）
$QP build_srt_layout.py --items items.1.7b-int8.json \
  --out "Transformer其实是个简单的架构.qwen3-asr.1.7b-int8.layout.srt"
#   可调：--min-dur 1.0 --max-dur 7 --cps 9 --line-width 18 --merge-below 4
#   （中文分词折断用 jieba，自动生效；未装则降级为无中文词约束）
```

---

## 7. 最终方案与剩余空间

**最终方案（已定型、可交付）**：
`int8-1.7B（全 GPU，3.02 GB）+ 分块 45 s + 标点断句 + 排版规范化（含 jieba 中文分词）`
→ 最终字幕 `...1.7b-int8.layout.srt`。要快则换 0.6B（≈2.5 min，精度略降）。

**已完成的优化**：
- ✅ 标点断句（③）、排版规范化（⑤：最短/最长时长、CPS、折行、合并、拉丁+中文词不劈开、消重叠）
- ✅ 分块 30→45 s（白捡 ~3%，默认已改）

**已确认走不通 / 收益低（不再尝试）**：
- ❌ vLLM 跑 1.7B（架构不支持 bnb 量化）；⚠️ flash-attn（工具链已装未编成，收益低）；⚠️ 更大分块（仅边际，已到 45 s）

**剩余空间（均非"通用+本地+零风险"，按需再议）**：
1. VAD 静音切分——通用但收益不确定（干净旁白幻听少），且需"逐段 ASR + 偏移还原 + padding"防时间戳漂移。
2. LLM 通用纠错——潜在精度收益最大，但需云端 API 或 CPU 小模型（4 GB 放不下），有过度纠正风险，越出"本机 GPU"边界。
3. 精度天花板：本机已到顶（int8-1.7B）；再上一档只能云端 Qwen3-ASR-Flash API 或换更大显存的卡。
4. （历史备选）flash-attn 完整编译环境变量见前文 5.2 ②；vLLM + 0.6B 高吞吐路线（降精度换速度）。

---

## 附录：通用参考（来自原 vLLM 部署方案，本机未采用，备将来大卡/服务化用）

> 以下路线针对 **≥8 GB 显卡的 vLLM 后端**，本机 4 GB 走的是 int8 + transformers 后端（见正文）。此处仅留存通用价值，供换卡、服务化或国内网络下载时参考。

### A. OpenAI 兼容 API 服务（vLLM serve）
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-ASR-1.7B \
  --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.85
```
```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [{"role":"user","content":[
    {"type":"audio_url","audio_url":{"url":"https://example.com/test.wav"}}]}]}'
```
- 同时支持 OpenAI transcription API（`/v1/audio/transcriptions`），原 Whisper 客户端可基本无缝迁移。
- 需 ≥8 GB 显存（含 KV cache + 对齐器）；OOM 时调低 `--gpu-memory-utilization`（如 0.7）或换 0.6B。

### B. Web UI 快速体验
```bash
# 带时间戳
qwen-asr-demo --asr-checkpoint Qwen/Qwen3-ASR-1.7B \
  --aligner-checkpoint Qwen/Qwen3-ForcedAligner-0.6B --cuda-visible-devices 0
```
浏览器开 `http://<ip>:8000`；远程用麦克风需配 HTTPS（`--ssl-certfile` / `--ssl-keyfile`）。

### C. 国内网络：ModelScope 镜像预下载权重
```bash
pip install modelscope
modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir ./models/Qwen3-ASR-1.7B
modelscope download --model Qwen/Qwen3-ForcedAligner-0.6B --local_dir ./models/Qwen3-ForcedAligner-0.6B
# 加载时把 model= / forced_aligner= 传本地目录即可
```

### D. 按显存选型参考（换卡时用）
| 配置 | 建议 |
|---|---|
| 4–6 GB（本机 3050） | 0.6B 全 GPU；或 **1.7B int8**（本报告方案，全 GPU 3.02 GB）|
| 8–12 GB（3060 / 4070） | 1.7B + Aligner，vLLM，`gpu_memory_utilization≈0.8` |
| 24 GB+（4090 / A100） | 1.7B + Aligner，拉高并发批量吞吐 |
| 无独立 GPU / 端侧 | sherpa-onnx + Qwen3-ASR-0.6B int8（CPU 方案）|

### E. vLLM 后端要点（换卡跑 vLLM 时）
- 代码逻辑必须包在 `if __name__ == "__main__":`（vLLM 用 spawn，否则报错）。
- GPU 选择用 `CUDA_VISIBLE_DEVICES`，vLLM 不支持 `cuda:0` 风格设备参数。
- 安装：`uv pip install "vllm[audio]"` + `uv pip install -U "qwen-asr[vllm]"`；报"model not supported"再换 nightly（cu129 wheel）。
