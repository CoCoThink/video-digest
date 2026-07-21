# FireRedASR v2 服务端接入设计

| 项目 | 内容 |
|---|---|
| 版本 | v1.2 |
| 日期 | 2026-07-21 |
| 状态 | 已实现并加固：固定源码校验、低能量边界、公共输出契约和上下文能力声明 |
| 范围 | FireRedASR2-AED 可选快速 backend；不替换 Qwen 默认路径 |
| 依据 | `FireRedASR-v2-实测报告.md`、`方案设计域AI实践分享-ASR对比报告.md` |

实现落点（截至 2026-07-21）：

- 已实现启动级 backend 工厂、Qwen 兼容包装和 FireRedASR2-AED 常驻后端；
- 已实现精确 PCM 切块、BF16 forced-align 兼容、异常检测、递归二分与 OOM 重试；
- 已实现 Worker ready Event、启动错误摘要，以及 config/health/status 模型元数据；
- 已通过 29 项自动测试和 10 秒音频完整 API 烟雾测试；
- 已实现 FireRedPunc CPU 常驻、逐块推理、严格字符投影和单块 fail-open；保留显式开关，FireRed 部署推荐启用。
- 95/157 分钟全文标点实测 RTF 为 0.00271/0.00251，语义字符保留率均为 100%，无投影降级。
- 已要求 FireRedASR2S 源码为指定 revision 的干净 git checkout，避免本地补丁或上游漂移改变推理行为。
- 已把硬 30 秒边界升级为“最大 30 秒、目标点前默认 5 秒低能量搜索”；时间轴仍按 PCM frame 连续且不丢不重。
- 统一 backend 接口已扩展 `context` 参数；FireRed 不支持时记录 `ASR_CONTEXT_UNSUPPORTED`，强制语言则明确拒绝。
- 当前全局 factory 另含 FunASR 实验 backend，整体选型见 `docs/ASR三模型接入评审与试验补充.md`；本文仍只约束 FireRed 路径。

## 1. 目标与边界

### 1.1 目标

1. 保持 Qwen3-ASR-1.7B int8 为默认质量模式，并保证现有产物逐字节回归。
2. 增加 `firered` 启动级 backend，在 4 GB GPU 上提供实测约 4–6 倍于 Qwen 的快速转写。
3. FireRed 输出继续满足统一的 `items.json -> layout -> SRT/TXT` 契约。
4. 把 BF16 时间戳兼容、decoder runaway 检测、递归二分和标点恢复纳入正式路径。
5. backend、模型版本和降级事件可观测、可回滚。

### 1.2 非目标

- 首期不做前端逐任务模型选择，也不在同一 Worker 内同时加载多个 ASR。
- 不接入 8B+ 的 FireRedASR2-LLM；4 GB GPU 不满足其常驻条件。
- 不引入 FireRedVAD/LID。当前只做目标切点前的低能量搜索，它不检测语音活动、不删除静音；完整 VAD 仍属后续独立优化。
- 没有人工标注集前，不宣称 FireRed 或 Qwen 的总体 CER 更优。

## 2. 关键决策

### 2.1 启动级切换

全局配置为 `VD_ASR_BACKEND=qwen|firered|funasr`，默认 `qwen`。Scheduler 启动 Worker 时把完整
配置传入；Worker 只构造选中的 backend，并在其生命周期内常驻。

```text
API / Scheduler（不 import torch）
              |
              | spawn(cfg)
              v
GPU Worker -> backend factory -> QwenBackend    （默认）
                            \-> FireRedBackend  （快速）
                            \-> FunAsrBackend   （实验，详见三模型补充）
```

不采用逐任务切换的原因：RTX 3050 4 GB 无法同时常驻两套模型；模型交换会引入
11–18 秒加载、约 10 GB FireRed 冷加载主机 RSS 峰值，并增加 CUDA 上下文损坏风险。
未来若必须提供前端模式选择，应使用独立 Worker/独立 GPU，而不是在当前单 Worker
中来回卸载模型。

### 2.2 统一 backend 契约

新增目录：

```text
server/pipeline/asr.py                 # factory / facade，保持现有 import 稳定
server/pipeline/backends/base.py       # Protocol + 上下文解析 + 公共输出校验
server/pipeline/backends/qwen.py       # 迁移现有实现，行为不变
server/pipeline/backends/firered.py    # AED、切块、重试、时间偏移
server/pipeline/backends/funasr.py     # FunASR 实验 backend（非本文重点）
server/pipeline/backends/wav_chunks.py # FireRed/FunASR 公共 PCM 切块
server/pipeline/punctuation.py         # FireRedPunc 适配与字符投影
```

建议接口：

```python
class AsrBackend(Protocol):
    name: str
    model_id: str

    def transcribe(
        self, audio_path: str, *, language: str | None,
        context: str = "",
        progress_cb: Callable[[int], None] | None,
    ) -> dict: ...
```

标准返回值：

```json
{
  "text": "带标点或原始全文",
  "language": "Chinese",
  "items": [{"t": "词", "s": 1.23, "e": 1.56}],
  "backend": "firered",
  "model_id": "FireRedASR2-AED",
  "warnings": []
}
```

硬性不变量：

- `items` 按绝对时间单调排列，时间有限且满足 `0 <= s <= e <= audio_duration+tolerance`；
- `text` 经 Qwen 对齐器同款 Unicode 规则归一化后，必须与 `items[].t` 拼接结果一致；
- 不满足契约时任务明确失败或标点层降级，禁止静默丢字、伪造成功时间戳；
- `layout`、`subtitle` 和下载 API 不感知具体 backend。
- `context` 非空时 FireRed 不静默接受：产物 warning 记录 `ASR_CONTEXT_UNSUPPORTED`；
  `language` 非空则启动任务后明确失败，因为 AED 当前接口不支持强制语言。

## 3. FireRed 转写流程

```text
audio_16k.wav
  -> PCM frame 精确迭代最大 30 s 主块
     -> 目标切点前 5 s 内搜索低能量边界
  -> FireRedASR2-AED BF16 / beam 1
  -> 检测正文、token 密度、时间戳合法性与均匀回退
       正常 -> 恢复绝对 offset
       异常 -> 30 -> 15 -> 7.5 s 递归二分
       7.5 s 仍异常 -> ASR_UNSTABLE，任务失败
  -> 每主块按序合并 text/items
  -> 可选 FireRedPunc
  -> 统一契约校验
  -> items.json -> layout -> SRT/TXT
```

### 3.1 音频切块

- 输入沿用公共 `preprocess.to_wav_16k()`，不重复重采样。
- 按 WAV frame 计算 offset，禁止用累计浮点切割，确保长视频无漂移。
- 每次只写当前主块和其重试子块，处理完立即删除；不预先复制整条 WAV 为 N 个文件。
- 默认最大主块 30 秒、低能量搜索窗口 5 秒、最小重试块 7.5 秒。边界只向前移动，
  因此块长不超上限；所有 offset 由 frame 计算，连续覆盖原 WAV。历史实测仍是固定
  30 秒边界，新增策略已由切块单元测试验证，但不能据此改写历史性能数据。

### 3.2 异常检测与二分

任一条件触发废弃本次结果并二分：

1. 有正文但没有时间戳；
2. token 数大于 `max(100, duration_sec * 12)`；
3. token >= 100 且至少 95% token 时长完全相同，判定为官方均匀时间戳回退；
4. 时间戳包含 NaN/Inf、越界、倒序、负时长或不可接受的重叠；
5. 可选增加尾部重复 n-gram 检测，作为 token 密度阈值的补充而非替代。

每次重试事件记录主块 offset、duration、原因、废弃字符/token 数和子块层级。
最小块仍异常时抛 `ASR_UNSTABLE`，不能接受伪时间戳继续生成字幕。157 分钟样本
316 个主块中 2 块需要重试；95 分钟样本 191 块无重试。

### 3.3 BF16 forced-align 兼容

FireRed AED 在 Ampere 上使用 BF16，而 `torchaudio.functional.forced_align` 不接受
BF16。正式适配层必须在 forced-align 调用点仅把 `log_probs` 转为 FP32，编码器和
解码器仍保持 BF16。补丁需幂等，并用真实时间戳分布测试防止上游异常被均匀时间戳
静默掩盖。长期应跟踪上游修复，避免永久维护全局 monkey patch。

## 4. 标点恢复

官方 FireRedPunc 是面向中文和英文的 BERT 标点模型，并提供独立 Python API。
该能力保持可关闭，代码默认关闭以避免缺少 checkpoint 时改变现有部署；配置
checkpoint 的 FireRed 部署推荐启用，异常时仍可通过环境变量立即回滚。

### 4.1 部署方式

- 采用 FireRedPunc 常驻 CPU，保护 4 GB GPU 的 ASR 余量。独立资源探测中模型加载
  6.69 秒、进程峰值 RSS 约 1.95 GB；正式按时间戳分成 191 个块后，95 分钟/
  25,710 字标点耗时 15.463 秒，RTF 0.00271，仅令 AED 实测 RTF 0.1147 增加约
  2.4%。
- 不采用 GPU 常驻：CPU 已满足性能目标，避免与 AED 争用 4 GB 显存。
- 按主 30 秒块恢复标点，避免把 2–4 万字全文一次送入 BERT。可用容量 1 的 CPU
  标点线程与下一块 GPU ASR 重叠，但必须先做串行等价性和线程安全测试。

### 4.2 字符投影守门

标点模型输出不得直接替换 ASR 正文。实现 `project_punctuation(raw, punctuated)`：

1. 对两者做 Unicode 字母/数字归一化并忽略大小写；
2. 两条字符序列一致时，只把新增标点/空格投影到原始 ASR 字符上；
3. 保留原始字符和大小写，使其仍能与 timed items 严格对应；
4. 若 Punc 改词、漏词或插入作为词内字符的撇号，记录 `PUNC_TEXT_MISMATCH` 并
   fail-open 使用无标点原文，不使整个转写任务失败。

这样 FireRedPunc 只能改变展示标点，不能再次制造 `text/items` 失配。

## 5. Worker、进度与错误处理

### 5.1 Worker 生命周期

`worker_main()` 改为：

```python
backend = create_backend(cfg)
ready_event.set()
for task_id in task_q:
    _process_one(backend, cfg, ...)
```

增加独立 `multiprocessing.Event` 表示模型真正加载完成；`/api/health` 的
`model_loaded` 读取 Event，而不是仅判断进程存活。backend 路径、
依赖或 checkpoint 配错导致 Worker 启动失败时，健康检查必须明确为未就绪。

### 5.2 进度

FireRed 以“完成的主块 / `ceil(duration/chunk_sec)`”为进度基准。低能量搜索可能使
实际主块数略多，循环末尾显式补到 99；递归子块不让进度回退，也不虚增进度，当前
主块的所有叶子成功后才完成该主块。进入 `postprocessing` 后由现有状态机收尾。

### 5.3 错误和降级

| 场景 | 行为 |
|---|---|
| AED OOM | 复用 `GPU_OOM` 分类；清缓存整任务重试一次，仍失败则终止 |
| 最小块仍异常 | `ASR_UNSTABLE`，任务失败并保留诊断日志 |
| Punc 加载失败 | backend 启动不就绪；关闭 `VD_FIRERED_PUNC_ENABLED` 后重启可回退到无标点 AED |
| Punc 单块失败/改词 | 使用原始无标点正文，记录 warning，任务继续 |
| text/items 不一致 | `ASR_OUTPUT_MISMATCH`，禁止生成残缺字幕 |
| checkpoint/source 不存在 | Worker 不 ready，健康接口暴露配置错误摘要 |

## 6. 配置与 API

新增配置建议：

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `VD_ASR_BACKEND` | `qwen` | 全局支持 `qwen` / `firered` / `funasr`；本文取 `firered` |
| `VD_ASR_CONTEXT` | 空 | FireRed 不消费；非空时写 `ASR_CONTEXT_UNSUPPORTED` warning |
| `VD_FIRERED_MODEL_PATH` | 空 | AED checkpoint，选择 FireRed 时必填 |
| `VD_FIRERED_SOURCE_DIR` | 空 | 指定 revision 的干净 git checkout；选择 FireRed 时必填 |
| `VD_FIRERED_BEAM_SIZE` | `1` | 实测 beam 3 收益不足且更慢 |
| `VD_FIRERED_MIN_CHUNK_SEC` | `7.5` | 异常二分下限 |
| `VD_FIRERED_BOUNDARY_SEARCH_SEC` | `5` | 在目标切点前搜索低能量边界，必须小于 `VD_CHUNK_SEC` |
| `VD_FIRERED_PUNC_ENABLED` | `false` | 显式启用；FireRed 部署推荐 `true`，保留快速回滚开关 |
| `VD_FIRERED_PUNC_MODEL_PATH` | 空 | FireRedPunc checkpoint |
| `VD_FIRERED_PUNC_DEVICE` | `cpu` | 已完成 CPU 实测；4 GB GPU 推荐保持 CPU |
| `VD_RTF` | `0.65` | 现有 Qwen 前端 ETA，保持向后兼容 |
| `VD_FIRERED_RTF` | `0.12` | FireRed 前端 ETA，已覆盖实测 Punc CPU 开销 |

`GET /api/config` 增加 `asr_backend` 和实际选中 backend 的 `rtf`；`GET /api/health`
增加 `backend`、`model_id`、`model_loaded`。任务创建时把 `asr_backend/model_id`
写入 `status.json`，历史任务可追溯其产物来源。下载接口和文件名保持不变。

首期不接受 API 请求指定 backend，防止调用方认为同一队列可任意混跑模型。

## 7. 依赖与版本固定

- FireRedASR2S 首次已验证源码提交：
  `4e7d9aaf4482a47cec1724807026b9b151926eb5`。
- AED `model.pth.tar` SHA-256：
  `4677cbd30988d63ed3e777f6a42a1e5260a3865317f6e15e488bef40954f7054`。
- 官方仓库为 Apache-2.0，并提供独立 AED/Punc Python API。部署镜像应 clone 固定
  commit 或安装由该 commit 构建的 wheel，不允许运行时跟随 `main`。
- 当前补充依赖已锁定在 `requirements-firered.txt`：`cn2an==0.5.23`、
  `kaldiio==2.18.1`、`kaldi_native_fbank==1.15`、`textgrid==1.6.1`；该文件通过
  `-r requirements.txt` 复用公共/Qwen 基础环境。
- FireRedPunc `model.pth.tar` SHA-256：
  `9a69e465132c79f639ba5a057cf06c90b2d7e36e4bbea56225ece260dcd0b0d4`。

官方资料：

- <https://github.com/FireRedTeam/FireRedASR2S>
- <https://huggingface.co/FireRedTeam/FireRedASR2-AED>
- <https://huggingface.co/FireRedTeam/FireRedPunc>

## 8. 测试与上线门槛

### 8.1 自动测试

1. backend factory：默认 Qwen、三种合法名称可分派、非法名称拒绝、FireRed 缺路径时不 ready。
2. Qwen 回归：同一 `items.json`、SRT、TXT 与改造前 byte-identical。
3. PCM 切块：总 frame 不丢不重，offset 精确，末块正确。
4. FireRed 异常：无时间戳、token 过密、均匀回退、非法时间戳均触发二分。
5. 递归终止：30 -> 15 -> 7.5；最小块异常明确失败。
6. 标点投影：只加标点时成功；改词、漏词、撇号变化时 fail-open。
7. 统一契约：正文/token 100% 对应，时间戳单调，cue 无非法区间/重叠。
8. Worker：ready Event、backend 元数据、错误分类和 respawn。
9. **待补单测**：源码守门应覆盖非固定 revision、dirty checkout、缺少 checkpoint 文件均拒绝启动；当前实现已具备检查，仍需隔离 git fixture。
10. 低能量切块：边界偏向安静区、总 frame 不丢不重、块长不越过上限。
11. 能力声明：context 产生 warning；强制 language 明确拒绝；自动语言记录 `Unknown`。

### 8.2 真实样本门槛

- 606 秒、95 分钟、157 分钟三条现有视频全部覆盖到结尾；
- FireRed 推理 RTF <= 0.15，AED 峰值显存 <= 2.8 GB；
- decoder runaway 必须被检测并恢复或明确失败，不接受均匀伪时间戳；
- 加 Punc 后归一化正文保留率 100%，非法 cue=0；
- 制作至少 30 分钟人工逐字稿，覆盖技术旁白、会议、多说话人和中英混合；报告
  CER/WER、领域词召回率，不能只以两模型互相一致度代替真值；
- FireRed 快速模式的 CER 相对 Qwen 恶化超过 20% 时不上线，或必须在 UI 明确标注
  速度/质量取舍。默认模型变更需另行评审，不属于本接入任务。

## 9. 实施顺序与回滚

1. **已完成：重构不改行为**：抽出 `QwenBackend` 和 factory，默认仍为 Qwen；通过回归测试。
2. **已完成：接入 AED**：迁移切块、BF16 兼容、异常检测和递归二分。
3. **已完成：接入 Punc**：完成 CPU 资源实测、字符投影、fail-open 和长视频复跑。
4. **已完成：可观测性**：ready Event、backend 元数据、重试/warning 日志和健康接口。
5. **已完成：可复现性与边界优化**：强制固定且干净的源码 checkout；加入公共输出
   校验、上下文 warning、语言能力校验和低能量主块边界。
6. **待灰度**：测试环境用 `VD_ASR_BACKEND=firered` 跑人工稿；生产初期保持 Qwen，
   FireRed 只作为独立快速部署配置。

回滚只需恢复 `VD_ASR_BACKEND=qwen` 并重启服务。任务产物仍为统一 SRT/TXT，历史
任务不需迁移；status 中的 backend 元数据用于区分来源。
