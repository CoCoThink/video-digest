# 在线视频生成字幕系统 — 服务端

需求见 `docs/在线视频生成字幕系统-需求.md`，实现依据 `docs/在线视频生成字幕系统-详细设计.md`，模型评审与试验结论见 `docs/ASR三模型接入评审与试验补充.md`。

## 架构一句话

浏览器用 ffmpeg.wasm 无损提取音轨（视频不出本机）→ 仅音频上传 → **API 进程**（FastAPI，不碰 CUDA）校验入队 → **常驻 GPU Worker 子进程**（默认 Qwen3-ASR-1.7B int8，也可启动为 FireRedASR2-AED 快速模式或 Fun-ASR-Nano 实验模式）转写 → 标点断句 + 排版规范化 → SRT/TXT。文件系统即状态，无数据库。

## 运行

Python 3.11 依赖已固定在 `requirements.txt`；系统还需安装 ffmpeg。FunASR 和 FireRed 的附加依赖分别见 `requirements-funasr.txt`、`requirements-firered.txt`。

**推荐用项目根的启动脚本 `run.sh`**（自动定位环境 Python、预检端口、打印地址）：

```bash
./run.sh                     # 前台启动（默认 :8200）
VD_PORT=8300 ./run.sh        # 改端口（任意 VD_* 变量均可覆盖，见下表）
VD_PYTHON=/path/to/python ./run.sh   # 指定解释器
VD_ASR_BACKEND=firered VD_FIRERED_MODEL_PATH=/models/FireRedASR2-AED \
  VD_FIRERED_SOURCE_DIR=/opt/FireRedASR2S \
  VD_FIRERED_PUNC_ENABLED=true VD_FIRERED_PUNC_MODEL_PATH=/models/FireRedPunc \
  ./run.sh
VD_ASR_BACKEND=funasr VD_FUNASR_MODEL_PATH=/models/Fun-ASR-Nano-2512 \
  VD_ASR_CONTEXT='应用架构,设计域,特性树,SDK,API,Socket' ./run.sh
./run.sh --bg                # 后台启动，日志写入 ./server.log（kill <pid> 停止）
```

脚本会按 `VD_PYTHON` → 当前激活的 conda 环境 → `~/mamba|miniconda3|micromamba/envs/qwen-asr` → PATH 中 `python3` 的顺序定位解释器，并在启动前**预检端口可绑定**（WSL2/Hyper-V 保留端口段会 bind 失败但 `ss` 看不到监听者，见《实现与踩坑》§4.5）；不可绑定时给出换端口/`net stop winnat` 提示并退出。

也可直接手动起：

```bash
/home/guan/mamba/envs/qwen-asr/bin/python -m server.main   # 或激活环境后 python -m server.main
```

启动后：

- 前端页面： <http://localhost:8200/>
- 模型在 Worker 子进程内加载（本机 Qwen 约 14s，其他 backend 以日志为准），日志出现模型加载完成后即可上传。

FireRed 源码需固定到实测 revision：

```bash
pip install -r requirements-firered.txt
git clone https://github.com/FireRedTeam/FireRedASR2S.git /opt/FireRedASR2S
git -C /opt/FireRedASR2S checkout 4e7d9aaf4482a47cec1724807026b9b151926eb5
modelscope download --model xukaituo/FireRedPunc --local_dir /models/FireRedPunc
```

Fun-ASR-Nano 安装与下载：

```bash
pip install -r requirements-funasr.txt
modelscope download --model FunAudioLLM/Fun-ASR-Nano-2512 \
  --local_dir /models/Fun-ASR-Nano-2512
```

## 配置（环境变量前缀 `VD_`，详见设计 §9）

| 变量 | 默认 | 说明 |
|---|---|---|
| `VD_HOST` / `VD_PORT` | `0.0.0.0` / `8200` | 监听 |
| `VD_DATA_DIR` | `./data` | 任务目录根 |
| `VD_ASR_BACKEND` | `qwen` | 启动级切换：`qwen` / `firered` / `funasr` |
| `VD_ASR_MODEL_PATH` / `VD_ALIGNER_MODEL_PATH` | Qwen 仓库名 | 支持本地目录 |
| `VD_ASR_CONTEXT` | 空 | Qwen 自由文本上下文；FunASR 逗号/分号/换行分隔热词；FireRed 返回不支持警告 |
| `VD_FIRERED_MODEL_PATH` | 空 | FireRedASR2-AED checkpoint 目录；选择 `firered` 时必填 |
| `VD_FIRERED_SOURCE_DIR` | 空 | 固定版本且工作区干净的 FireRedASR2S 源码目录；选择 `firered` 时必填 |
| `VD_FIRERED_BEAM_SIZE` / `VD_FIRERED_MIN_CHUNK_SEC` | `1` / `7.5` | FireRed 解码与异常递归下限 |
| `VD_FIRERED_BOUNDARY_SEARCH_SEC` | `5` | 在目标切点前搜索低能量边界，块长不会超过 `VD_CHUNK_SEC` |
| `VD_FIRERED_PUNC_ENABLED` | `false` | FireRed 模式启用官方标点恢复 |
| `VD_FIRERED_PUNC_MODEL_PATH` | 空 | FireRedPunc checkpoint；启用标点时必填 |
| `VD_FIRERED_PUNC_DEVICE` | `cpu` | `cpu` / `cuda`；4GB GPU 推荐 CPU |
| `VD_FUNASR_MODEL_PATH` | `FunAudioLLM/Fun-ASR-Nano-2512` | FunASR 本地目录或 ModelScope ID |
| `VD_FUNASR_BOUNDARY_SEARCH_SEC` | `5` | FunASR 目标切点前的低能量搜索窗口 |
| `VD_ASR_LANGUAGE` | `None`(自动) | Qwen/FunASR 可传语言提示；FireRed 不支持强制语言并会明确报错 |
| `VD_CHUNK_SEC` | `30` | 分块秒数（4GB 卡单块峰值留余量；大显存卡可调高提速） |
| `VD_QUEUE_CAPACITY` | `10` | 容量上限（含上传中、排队中和执行中） |
| `VD_MAX_DURATION_SEC` / `VD_MAX_SIZE_BYTES` | `10800` / `200MB` | 时长/大小上限 |
| `VD_UPLOAD_CHUNK_SIZE_BYTES` | `16777216`（16MiB） | 浏览器分块上传的单请求大小；默认兼容 Cloudflare 代理 |
| `VD_UPLOAD_SESSION_TTL_SEC` | `86400` | 未完成上传的保留时间，过期后自动清理 |
| `VD_TIMEOUT_FACTOR` / `VD_TIMEOUT_BASE_SEC` | `3` / `600` | 超时 = 时长×factor+base |
| `VD_RTF` | `0.65` | 预估实时率（经 /api/config 下发） |
| `VD_FIRERED_RTF` | `0.12` | FireRed 模式预估实时率 |
| `VD_FUNASR_RTF` | `0.13` | FunASR 实验模式预估实时率 |
| `VD_LAYOUT_*` | — | 排版参数（min-dur/max-dur/cps/line-width/merge-below） |

## API（详见设计 §6）

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/uploads` | 初始化分块上传（JSON：`filename` + `size`） |
| PUT | `/api/uploads/{id}` | 上传二进制分块（请求头：`Upload-Offset`） |
| GET / DELETE | `/api/uploads/{id}` | 查询已接收偏移 / 取消并清理上传 |
| POST | `/api/uploads/{id}/complete` | 校验完整音频并创建任务（幂等） |
| GET | `/api/tasks` | 历史任务列表（最新在前，默认不含已删墓碑；`?include_deleted=true` 含之） |
| GET | `/api/tasks/{id}` | 查询状态（含动态 `queue_position`），2s 轮询 |
| DELETE | `/api/tasks/{id}` | 取消排队 / 删除完成任务（留墓碑） |
| GET | `/api/tasks/{id}/result?format=srt\|txt` | 下载产物 |
| GET | `/api/config` | 前端配置下发 |
| GET | `/api/health` | 健康检查 |

## 说明

- `server/pipeline/` 由根目录实验脚本（`qwen3_asr_compare.py` / `build_srt_layout.py`）逐行等价迁移；给定相同 ASR 输出，产物与独立脚本**逐字节一致**。
- 转写阶段在主线程跑第 `i+1` 块 GPU ASR，同时用单个 CPU 线程对齐第 `i` 块；最多一个对齐任务在途，不增加模型或显存占用。120 秒/4 块实测由 119.366s 降至 84.337s（1.415x），正文、语言和 570 个时间戳与 qwen-asr 原串行路径完全一致。
- FireRedPunc 启用后默认在 CPU 常驻，逐主块恢复标点；字符投影守门保证字母、数字、大小写和词内撇号来自原始 ASR，改词或推理失败时仅降级当前块。95/157 分钟历史样本标点 RTF 为 0.00271/0.00251，语义字符保留率 100%，FireRed 部署推荐启用。
- FireRedASR2-AED 不提供当前调用路径可用的语言强制或 LID 输出；自动模式在产物中记为 `Unknown`，避免伪造语言元数据。
- Fun-ASR-Nano 使用原生标点和 CTC 字符时间戳。4GB RTX 3050 的 180 秒实测 RTF 0.122、峰值 3.755GB，显存余量很小，因此仍标记为实验后端；上线前按 20–50 个代表性文件做 CER、OOM 和长视频验收，并审阅其独立的[模型许可](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE)。
- 已有 FireRed `items.json` 可用 `fireredpunc_items.py` 离线重建带标点的 SRT/TXT 和指标，便于回归比较而无需重跑 ASR。
- 本机 GPU 为 4GB RTX 3050，`device_map={"":0}` 不可写成 `"cuda:0"`（触发 bnb SCB bug）。
- 安全：task_id 128bit 随机 + 读取侧正则白名单；下载固定 `application/octet-stream + attachment`；预览用 `textContent`。生产公网暴露须在反向代理层加访问控制。
- Cloudflare Free/Pro 的单请求上传上限为 100MB。页面固定以 16MiB 小请求续传，因此完整音频仍可使用本服务配置的 200MB 上限；不要把 `VD_UPLOAD_CHUNK_SIZE_BYTES` 调到代理上限附近。

单元测试（需 qwen-asr 环境）：

```bash
/home/guan/mamba/envs/qwen-asr/bin/python -m unittest discover -s tests -v
```
