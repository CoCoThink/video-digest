#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ASR + 强制对齐（由 qwen3_asr_compare.py 去 CLI 化重构，详细设计 §4.3）。

Qwen3-ASR-1.7B int8 全 GPU（bitsandbytes），Qwen3-ForcedAligner-0.6B 固定
CPU（float32），4GB 显存可容纳。仅在 GPU Worker 子进程内导入（API 进程不触碰
CUDA），故 torch / qwen_asr 均为函数内延迟导入。
"""
import time


def load_model(cfg: dict):
    """加载常驻模型。cfg 为 settings.model_dump()（可 pickle 跨进程）。"""
    import os
    import torch

    # HuggingFace 镜像兜底：直连 huggingface.co 常超时（ReadTimeoutError）。HF_ENDPOINT
    # 的镜像地址在 import huggingface_hub 时被冻结，故必须在 import qwen_asr 之前设置。
    # setdefault → run.sh 已 export 的值优先，直接跑时才用镜像。
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    # 且让 hf-mirror.com 绕过本机代理（no_proxy 由 requests 在请求时读取，可后设）：
    # 代理走海外出口时，hf-mirror 见境外 IP 会 308 跳回 huggingface.co 再超时/504。
    for _k in ("no_proxy", "NO_PROXY"):
        _v = os.environ.get(_k, "")
        if "hf-mirror.com" not in _v:
            os.environ[_k] = f"{_v},hf-mirror.com" if _v else "hf-mirror.com"

    # 库在 transcribe() 时读取这两个模块级全局；缩小分块窗口让音频编码器卷积
    # 激活值放进小显存（45s 实测最优，见实测报告 §4）。
    import qwen_asr.inference.qwen3_asr as _qa
    _qa.MAX_FORCE_ALIGN_INPUT_SECONDS = cfg["chunk_sec"]
    _qa.MAX_ASR_INPUT_SECONDS = cfg["chunk_sec"]
    from qwen_asr import Qwen3ASRModel

    kwargs = dict(
        forced_aligner=cfg["aligner_model_path"],
        forced_aligner_kwargs={"device_map": "cpu", "dtype": torch.float32},
        max_inference_batch_size=1,
        max_new_tokens=512,
    )
    if cfg.get("load_8bit", True):
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        # device_map={"":0} 必须——"cuda:0" 触发 bnb SCB 分发 bug（实测报告 §4.1）。
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["dtype"] = torch.bfloat16
        kwargs["device_map"] = "cuda:0"

    return Qwen3ASRModel.from_pretrained(cfg["asr_model_path"], **kwargs)


def transcribe(model, audio_path: str, language=None) -> dict:
    """转写 + 对齐，返回 items.json 结构（全文 + 逐 token 时间戳）。

    OOM 自愈（R4）：捕获 CUDA OOM → empty_cache → 原参重试 1 次。
    """
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    def _run():
        return model.transcribe(audio=audio_path, language=language,
                                return_time_stamps=True)

    try:
        results = _run()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        results = _run()  # 重试一次；再失败由调用方标记 GPU_OOM

    r = results[0]
    ts = r.time_stamps
    items = getattr(ts, "items", None) if ts is not None else None
    data = {
        "text": r.text,
        "language": r.language,
        "items": [
            {"t": it.text, "s": float(it.start_time), "e": float(it.end_time)}
            for it in (items or [])
        ],
    }
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        data["_peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    return data
