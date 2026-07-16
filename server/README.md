# 在线视频生成字幕系统 — 服务端

需求见 `docs/在线视频生成字幕系统-需求.md`，实现依据 `docs/在线视频生成字幕系统-详细设计.md`。

## 架构一句话

浏览器用 ffmpeg.wasm 无损提取音轨（视频不出本机）→ 仅音频上传 → **API 进程**（FastAPI，不碰 CUDA）校验入队 → **常驻 GPU Worker 子进程**（Qwen3-ASR-1.7B int8 全 GPU + 对齐 0.6B CPU）转写 → 标点断句 + 排版规范化 → SRT/TXT。文件系统即状态，无数据库。

## 运行

需在已装好 Qwen3-ASR 栈的环境（本机为 mamba env `qwen-asr`：torch 2.9.1+cu128、qwen-asr 0.0.6、bitsandbytes、jieba、fastapi、uvicorn、ffmpeg 8.x）：

```bash
# 项目根目录
/home/guan/mamba/envs/qwen-asr/bin/python -m server.main
# 或激活环境后： python -m server.main
```

启动后：

- 前端页面： <http://localhost:8200/>
- 模型在 Worker 子进程内加载（约 14s），日志出现「模型就绪」后即可上传。

## 配置（环境变量前缀 `VD_`，详见设计 §9）

| 变量 | 默认 | 说明 |
|---|---|---|
| `VD_HOST` / `VD_PORT` | `0.0.0.0` / `8200` | 监听 |
| `VD_DATA_DIR` | `./data` | 任务目录根 |
| `VD_ASR_MODEL_PATH` / `VD_ALIGNER_MODEL_PATH` | Qwen 仓库名 | 支持本地目录 |
| `VD_ASR_LANGUAGE` | `None`(自动) | 可强制 `Chinese`/`English` |
| `VD_CHUNK_SEC` | `45` | 分块秒数（4GB 显存实测最优） |
| `VD_QUEUE_CAPACITY` | `10` | 队列长度（含执行中） |
| `VD_MAX_DURATION_SEC` / `VD_MAX_SIZE_BYTES` | `10800` / `200MB` | 时长/大小上限 |
| `VD_TIMEOUT_FACTOR` / `VD_TIMEOUT_BASE_SEC` | `3` / `600` | 超时 = 时长×factor+base |
| `VD_RTF` | `0.65` | 预估实时率（经 /api/config 下发） |
| `VD_LAYOUT_*` | — | 排版参数（min-dur/max-dur/cps/line-width/merge-below） |

## API（详见设计 §6）

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/tasks` | 上传音频建任务（multipart：`file` + `filename`） |
| GET | `/api/tasks/{id}` | 查询状态（含动态 `queue_position`），2s 轮询 |
| DELETE | `/api/tasks/{id}` | 取消排队 / 删除完成任务（留墓碑） |
| GET | `/api/tasks/{id}/result?format=srt\|txt` | 下载产物 |
| GET | `/api/config` | 前端配置下发 |
| GET | `/api/health` | 健康检查 |

## 说明

- `server/pipeline/` 由根目录实验脚本（`qwen3_asr_compare.py` / `build_srt_layout.py`）逐行等价迁移；给定相同 ASR 输出，产物与独立脚本**逐字节一致**。
- 本机 GPU 为 4GB RTX 3050，`device_map={"":0}` 不可写成 `"cuda:0"`（触发 bnb SCB bug）。
- 安全：task_id 128bit 随机 + 读取侧正则白名单；下载固定 `application/octet-stream + attachment`；预览用 `textContent`。生产公网暴露须在反向代理层加访问控制。
