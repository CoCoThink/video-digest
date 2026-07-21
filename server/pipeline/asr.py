#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ASR + 强制对齐（由 qwen3_asr_compare.py 去 CLI 化重构，详细设计 §4.3）。

Qwen3-ASR-1.7B int8 全 GPU（bitsandbytes），Qwen3-ForcedAligner-0.6B 固定
CPU（float32）。逐块 ASR 与上一块的对齐并行，以同时利用 GPU/CPU 且不增加显存
占用。仅在 GPU Worker 子进程内导入 torch / qwen_asr，API 进程不触碰 CUDA。
"""
from concurrent.futures import ThreadPoolExecutor


def create_backend(cfg: dict):
    """Load the selected resident backend inside the GPU worker process."""
    name = str(cfg.get("asr_backend", "qwen")).strip().lower()
    if name == "qwen":
        from server.pipeline.backends.qwen import QwenBackend
        return QwenBackend(cfg)
    if name == "firered":
        from server.pipeline.backends.firered import FireRedBackend
        return FireRedBackend(cfg)
    if name == "funasr":
        from server.pipeline.backends.funasr import FunAsrBackend
        return FunAsrBackend(cfg)
    raise ValueError(f"unsupported ASR backend: {name!r}")


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
    # 激活值放进小显存。默认 30s：4GB 卡上单块峰值内容相关，45s 会被密集块越界
    # 触发驱动级 OOM 硬退出（见 config.chunk_sec 注释与踩坑 §4.8）。
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


def _is_oom(e: BaseException) -> bool:
    """判定是否为显存不足类错误。

    4GB 卡实测：越界时抛的是**普通 ``RuntimeError``**（"CUDA driver error: out of
    memory" 或 "CUDA out of memory"），并非 ``torch.cuda.OutOfMemoryError``——后者
    是前者子类但此环境几乎不走到。故按消息特征兜底识别，避免漏判成 INTERNAL。
    """
    import torch
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    msg = str(e).lower()
    return isinstance(e, RuntimeError) and ("out of memory" in msg or "cuda driver error" in msg)


class _Progress:
    """按 N 次 ASR + 至多 N 次对齐上报真实完成量。"""

    def __init__(self, cb):
        self.cb = cb
        self.total = 0
        self.done = 0
        self.last = -1

    def reset(self, chunks: int) -> None:
        """OOM 重试会重新推理；重置计数，下一步允许进度回退到真实值。"""
        self.total = 2 * chunks
        self.done = 0
        self.last = -1

    def complete(self) -> None:
        if self.total <= 0:
            return
        self.done += 1
        pct = max(1, min(99, int(self.done / self.total * 100)))
        if pct == self.last:
            return
        self.last = pct
        try:
            self.cb(pct)
        except Exception:
            pass  # 进度落盘失败绝不能拖垮转写


def _align_one(model, chunk, text: str, language: str):
    """在 CPU 线程完成单块对齐并恢复其在原音频内的时间偏移。"""
    result = model.forced_aligner.align(
        audio=[(chunk.wav, chunk.sr)], text=[text], language=[language]
    )[0]
    return model._offset_align_result(result, chunk.offset_sec)


def _transcribe_pipelined(
    model, audio_path: str, language, context: str, progress: _Progress | None,
):
    """逐块 GPU ASR，并在单独 CPU 线程对齐上一块。

    qwen-asr 0.0.6 的原实现先完成全部 ASR 再开始全部对齐。本项目固定 batch=1，
    因此可保持每次推理输入不变，把两段循环改排为容量 1 的两级流水线：主线程跑
    第 i 块 ASR 时，CPU 线程跑第 i-1 块对齐。容量 1 提供自然背压，不会积压音频
    块或在异常时等待一长串无用任务。
    """
    import qwen_asr.inference.qwen3_asr as _qa

    if model.forced_aligner is None:
        raise ValueError("pipelined transcription requires a forced aligner")

    wavs = _qa.normalize_audios(audio_path)
    if len(wavs) != 1:
        raise ValueError(f"expected one audio input, got {len(wavs)}")

    if language is None or str(language).strip() == "":
        language_norm = None
    else:
        language_norm = _qa.normalize_language_name(str(language))
        _qa.validate_language(language_norm)

    chunks = [
        _qa.AudioChunk(orig_index=0, chunk_index=i, wav=wav, sr=_qa.SAMPLE_RATE,
                       offset_sec=offset)
        for i, (wav, offset) in enumerate(_qa.split_audio_into_chunks(
            wav=wavs[0], sr=_qa.SAMPLE_RATE,
            max_chunk_sec=_qa.MAX_FORCE_ALIGN_INPUT_SECONDS,
        ))
    ]
    if progress:
        progress.reset(len(chunks))

    per_chunk_lang = []
    per_chunk_text = []
    per_chunk_align = [None] * len(chunks)
    pending = None
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="forced-align")
    try:
        for idx, chunk in enumerate(chunks):
            raw = model._infer_asr([context], [chunk.wav], [language_norm])[0]
            lang_pred, text = _qa.parse_asr_output(raw, user_language=language_norm)
            per_chunk_lang.append(lang_pred)
            per_chunk_text.append(text)
            if progress:
                progress.complete()

            # ASR 当前块时，pending 正在 CPU 对齐上一块；此处汇合后再投递当前块，
            # 从而始终只有一个对齐任务在途。
            if pending is not None:
                align_idx, future = pending
                per_chunk_align[align_idx] = future.result()
                if progress:
                    progress.complete()
                pending = None

            if text.strip():
                pending = (idx, executor.submit(
                    _align_one, model, chunk, text, lang_pred
                ))

        if pending is not None:
            align_idx, future = pending
            per_chunk_align[align_idx] = future.result()
            if progress:
                progress.complete()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    merged_text = "".join(text for text in per_chunk_text if text is not None)
    merged_language = _qa.merge_languages(per_chunk_lang)
    merged_align = model._merge_align_results(
        [result for result in per_chunk_align if result is not None]
    )
    return _qa.ASRTranscription(
        language=merged_language, text=merged_text, time_stamps=merged_align
    )


def transcribe(
    model, audio_path: str, language=None, context: str = "", progress_cb=None,
) -> dict:
    """转写 + 对齐，返回 items.json 结构（全文 + 逐 token 时间戳）。

    OOM 自愈（R4）：捕获显存不足（含普通 RuntimeError 形态，见 _is_oom）→
    empty_cache → 原参重试 1 次；再失败由调用方标记 GPU_OOM。
    progress_cb(pct:int)：可选，转写过程中按已完成的 ASR/对齐步数回调 1-99。
    """
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    progress = _Progress(progress_cb) if progress_cb else None

    def _run():
        with torch.no_grad():
            return _transcribe_pipelined(
                model, audio_path, language, context.strip(), progress,
            )

    try:
        result = _run()
    except Exception as e:
        if not _is_oom(e):
            raise
        # 显存不足：清缓存后原参重试一次。注意：若驱动级 OOM 已损坏 CUDA 上下文，
        # 同进程重试通常仍失败并再次抛出，交由调用方标记 GPU_OOM（干净失败，
        # 优于 Worker 硬退出的 WORKER_CRASHED）。
        torch.cuda.empty_cache()
        result = _run()

    ts = result.time_stamps
    items = getattr(ts, "items", None) if ts is not None else None
    data = {
        "text": result.text,
        "language": result.language,
        "items": [
            {"t": it.text, "s": float(it.start_time), "e": float(it.end_time)}
            for it in (items or [])
        ],
    }
    from server.pipeline.backends.base import validate_asr_output, wav_duration
    validate_asr_output(
        data, wav_duration(audio_path), backend_name="Qwen3-ASR",
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        data["_peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    return data
