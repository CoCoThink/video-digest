#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可配置项（详细设计 §9）。

全部通过环境变量（前缀 ``VD_``）或 ``.env`` 覆盖，带工程默认值。API 进程与
GPU Worker 子进程共享同一份配置：Worker 由 spawn 启动，父进程把
``settings.model_dump()``（纯 dict，可 pickle）传入子进程。
"""
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VD_", env_file=".env", extra="ignore",
        protected_namespaces=(),  # 允许 asr_model_path 等以 model 无关的字段名
    )

    # 服务
    host: str = "0.0.0.0"
    port: int = 8200
    data_dir: Path = Path("./data")

    # 模型（支持本地目录或 HF/ModelScope 仓库名，从缓存解析）
    asr_backend: Literal["qwen", "firered", "funasr"] = "qwen"
    asr_model_path: str = "Qwen/Qwen3-ASR-1.7B"
    aligner_model_path: str = "Qwen/Qwen3-ForcedAligner-0.6B"
    load_8bit: bool = True
    # Qwen 将其作为自由文本上下文；FunASR 按逗号/分号/换行解析为热词。
    asr_context: str = ""
    # FireRedASR2-AED 快速后端。模型目录包含 model.pth.tar/dict.txt/cmvn.ark；
    # source_dir 必须指向固定 revision 的干净 FireRedASR2S checkout。
    firered_model_path: Path | None = None
    firered_source_dir: Path | None = None
    firered_beam_size: int = 1
    firered_min_chunk_sec: float = 7.5
    firered_boundary_search_sec: float = 5.0
    firered_punc_enabled: bool = False
    firered_punc_model_path: Path | None = None
    firered_punc_device: Literal["cpu", "cuda"] = "cpu"
    # Fun-ASR-Nano 实验后端；固定 FunASR 版本见 requirements-funasr.txt。
    funasr_model_path: str = "FunAudioLLM/Fun-ASR-Nano-2512"
    funasr_boundary_search_sec: float = 5.0
    # 分块秒数：每块 GPU 峰值随块长增长。4GB 卡上常驻 int8 模型基线约占 3.07GB
    # reserved（仅 ~1GB 余量），而峰值是**内容相关**的——语速密集的块生成 token 多、
    # 音频编码器激活大，单块瞬时分配可再高 0.5–1GB。45s 时密集块会越过 4GB，触发
    # 驱动级 OOM（RuntimeError: CUDA driver error: out of memory）并可能损坏 CUDA
    # 上下文致 Worker 硬退出（WORKER_CRASHED，见实现与踩坑）。降到 30s 收窄单块峰值、
    # 留足余量；显存更大的卡可用 VD_CHUNK_SEC 调高。
    chunk_sec: int = 30
    # 识别语言：None=自动判定（支持纯中文/纯英文/中英混合，验收 8）；也可强制
    # "Chinese"/"English"。纯中文内容强制 Chinese 时断句更密、与实测参照物更贴近。
    asr_language: str | None = None

    # 队列与限额
    queue_capacity: int = 10          # 含正在执行的任务
    max_duration_sec: int = 10800     # 音频时长上限（3h）
    max_size_bytes: int = 209_715_200  # 上传大小上限（200MB）
    # 单请求保持远低于 Cloudflare Free/Pro 的 100MB 请求体上限。
    upload_chunk_size_bytes: int = 16_777_216  # 16MiB
    upload_session_ttl_sec: int = 86_400       # 未完成上传保留 24h

    # 超时 = 时长 × factor + base（FR-B2）
    timeout_factor: float = 3.0
    timeout_base_sec: int = 600

    # 前端预估用实时率（经 /api/config 下发）
    rtf: float = 0.65
    firered_rtf: float = 0.12
    funasr_rtf: float = 0.13

    # 排版参数（build_srt_layout 默认值；R5 英文适配时按语言切换参数组）
    layout_min_dur: float = 1.0
    layout_max_dur: float = 7.0
    layout_cps: float = 9.0
    layout_line_width: int = 18
    layout_merge_below: int = 4
    layout_soft_min: int = 12
    layout_hard_max: int = 34


settings = Settings()
