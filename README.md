# video-digest：在线视频生成字幕

在浏览器里选一个本地视频，几分钟后拿到 SRT/TXT 字幕。视频不离开本机：浏览器用 ffmpeg.wasm 无损提取音轨（`-c:a copy`，只解封装不重编码），只上传音频；后端单机单 GPU 跑 ASR，再做标点断句和字幕排版。

```
浏览器                                   服务端
┌────────────────────────┐             ┌─────────────────────────────────────────┐
│ 选择视频               │             │ API 进程 (FastAPI，不碰 CUDA)            │
│  → ffmpeg.wasm 提取音轨 │  分块上传   │  校验 / 入队 / 状态 / 下载               │
│  → 16MiB 分块续传       │ ─────────▶ │        │                                │
│  → 轮询进度、预览、下载 │ ◀───────── │        ▼                                │
└────────────────────────┘  SRT / TXT  │ 常驻 GPU Worker 子进程                  │
                                        │  ASR → 时间戳 → 标点断句 → 排版 → 产物  │
                                        └─────────────────────────────────────────┘
```

文件系统即状态（`data/` 下每任务一个目录），无数据库。

## 特性

- 视频不上传，只传无损提取的音轨；也可以直接上传音频文件
- 分块上传、断点续传，默认 16MiB 一块，可以放在 Cloudflare 代理后面使用
- 三种 ASR 后端，启动时选定一个常驻：

  | 后端 | 模型 | 定位 | 本机实测 RTF* |
  |---|---|---|---|
  | `qwen`（默认） | Qwen3-ASR-1.7B int8 + Qwen3-ForcedAligner-0.6B（CPU） | 质量优先 | ~0.65 |
  | `firered` | FireRedASR2-AED + FireRedPunc（CPU，可选） | 快速模式 | ~0.12 |
  | `funasr` | Fun-ASR-Nano-2512 | 实验性 | ~0.13 |

  \* RTX 3050 4GB 实测，详见 `docs/` 下各实测报告。

- 字幕后处理：按标点断句，限制最短/最长显示时长、阅读速度（CPS）、行宽并折成两行，用 jieba 分词避免把中文词和 `w_k`、`Q1` 这类 token 从中间断开
- 任务队列（含上传中、排队、执行中，默认上限 10 个）、超时控制、历史任务列表、删除任务
- 支持中文、英文、中英混合（FireRed 不支持指定语言）

## 环境要求

- Linux / WSL2，Python 3.11
- NVIDIA GPU + CUDA 12.8 驱动（默认配置按 4GB 显存调优）
- 系统已安装 `ffmpeg`
- 现代浏览器（ffmpeg.wasm 已放在 `web/vendor/ffmpeg/`，不依赖 CDN）

## 快速开始

```bash
# 1. 安装依赖（默认 Qwen 后端）
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. 启动（默认 http://localhost:8200/）
./run.sh
```

首次启动会从 HuggingFace 下载模型。`run.sh` 默认使用 `hf-mirror.com` 镜像，已设置的 `HF_ENDPOINT` 优先。看到 Worker 日志里模型加载完成（Qwen 约 14 秒）后，打开页面上传即可。

`run.sh` 常用用法：

```bash
VD_PORT=8300 ./run.sh                 # 改端口
VD_PYTHON=/path/to/python ./run.sh    # 指定解释器
./run.sh --bg                         # 后台运行，日志写入 ./server.log
```

脚本会自动找到装好所选后端依赖的 Python，并在启动前检查端口能否绑定（WSL2/Hyper-V 保留端口段会让 bind 失败，`ss` 却看不到占用者）。也可以直接运行 `python -m server.main`。

### 切换后端

FireRed（快速模式）：

```bash
pip install -r requirements-firered.txt
git clone https://github.com/FireRedTeam/FireRedASR2S.git /opt/FireRedASR2S
git -C /opt/FireRedASR2S checkout 4e7d9aaf4482a47cec1724807026b9b151926eb5
modelscope download --model xukaituo/FireRedPunc --local_dir /models/FireRedPunc

VD_ASR_BACKEND=firered \
VD_FIRERED_MODEL_PATH=/models/FireRedASR2-AED \
VD_FIRERED_SOURCE_DIR=/opt/FireRedASR2S \
VD_FIRERED_PUNC_ENABLED=true VD_FIRERED_PUNC_MODEL_PATH=/models/FireRedPunc \
./run.sh
```

Fun-ASR-Nano（实验性，使用前请看一下它的[模型许可](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE)）：

```bash
pip install -r requirements-funasr.txt
modelscope download --model FunAudioLLM/Fun-ASR-Nano-2512 --local_dir /models/Fun-ASR-Nano-2512

VD_ASR_BACKEND=funasr VD_FUNASR_MODEL_PATH=/models/Fun-ASR-Nano-2512 \
VD_ASR_CONTEXT='应用架构,设计域,SDK,API' ./run.sh
```

## 配置

所有配置都用 `VD_` 前缀的环境变量设置，也可以写在项目根目录的 `.env` 里，定义见 `server/config.py`。常用项：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `VD_HOST` / `VD_PORT` | `0.0.0.0` / `8200` | 监听地址 |
| `VD_DATA_DIR` | `./data` | 任务数据目录 |
| `VD_ASR_BACKEND` | `qwen` | `qwen` / `firered` / `funasr` |
| `VD_ASR_LANGUAGE` | 自动 | 例如 `Chinese`、`English`（FireRed 不支持） |
| `VD_ASR_CONTEXT` | 空 | Qwen：自由文本上下文；FunASR：热词列表 |
| `VD_CHUNK_SEC` | `30` | 分块时长，显存大的卡可以调高来提速 |
| `VD_QUEUE_CAPACITY` | `10` | 队列容量 |
| `VD_MAX_DURATION_SEC` / `VD_MAX_SIZE_BYTES` | 3 小时 / 200MB | 上传限制 |

完整配置表和 API 列表见 [server/README.md](server/README.md)。

## API 概览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/uploads` | 初始化分块上传 |
| PUT | `/api/uploads/{id}` | 上传一个分块（请求头 `Upload-Offset`） |
| POST | `/api/uploads/{id}/complete` | 完成上传并创建任务 |
| GET | `/api/tasks` / `/api/tasks/{id}` | 任务列表 / 任务状态 |
| GET | `/api/tasks/{id}/result?format=srt\|txt` | 下载字幕 |
| DELETE | `/api/tasks/{id}` | 取消或删除任务 |
| GET | `/api/config` / `/api/health` | 前端配置 / 健康检查 |

## 项目结构

```
server/                 后端服务
  main.py               FastAPI 应用与路由，同时托管 web/
  scheduler.py          任务队列与调度
  worker.py             常驻 GPU Worker 子进程
  store.py              基于文件系统的任务存储
  pipeline/             转写管线：预处理、ASR、标点、排版、字幕输出
    backends/           qwen / firered / funasr 后端实现
web/                    前端页面（原生 JS）与自托管 ffmpeg.wasm
tests/                  单元测试
docs/                   需求、详细设计、实现踩坑、模型实测报告
*.py（根目录）          离线实验脚本：命令行转写、对比、重建 SRT
```

根目录脚本可以脱离服务单独使用，例如：

```bash
python qwen3_asr_service_to_srt.py --input video.mkv --out video.srt
python build_srt_layout.py --items video.items.json --out video.srt   # 由 items.json 重新排版，不必重跑 ASR
python fireredpunc_items.py --items x.items.json --out x.srt --model-dir ... --source-dir ...
```

参数说明用 `--help` 查看。服务端的 `server/pipeline/` 是从这些脚本迁移过来的，ASR 输出相同时生成的字幕逐字节一致。

## 测试

```bash
python -m unittest discover -s tests -v
```

需要在装好默认后端依赖的环境中运行。

## 安全说明

服务自身没有认证和访问控制，设计上用于单用户或内网环境，所有访问者都能看到全部历史任务。如果要暴露到公网，请在反向代理层加上鉴权。服务端已做的防护：task_id 为 128 位随机值，读取时按白名单正则校验；下载固定以附件形式返回；前端预览只用 `textContent` 渲染。

## 文档

- [需求](docs/在线视频生成字幕系统-需求.md)
- [详细设计](docs/在线视频生成字幕系统-详细设计.md)
- [实现与踩坑](docs/在线视频生成字幕系统-实现与踩坑.md)：WSL2 端口、4GB 显存 OOM、bitsandbytes 等问题的记录
- [ASR 三模型接入评审与试验补充](docs/ASR三模型接入评审与试验补充.md)
- 实测报告：[Qwen3-ASR](docs/Qwen3-ASR-实测报告.md)、[FireRedASR v2](docs/FireRedASR-v2-实测报告.md)
