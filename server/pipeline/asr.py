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


_MISSING = object()


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


class _ProgressPatch:
    """把 qwen_asr 逐块推理换算成 0-100 的真实进度（详细设计 §4.3 P1）。

    transcribe() 内部先 split_audio_into_chunks() 把整段音频切成 N 块，随后逐块
    先做一次 ASR 生成、再做一次强制对齐；本项目 batch=1，故 generate/align 各被
    调用一次/块（对齐会跳过静音块，略少）。总步数记为 2N，每完成一步回调百分比。

    实现为「进 transcribe 前打补丁、出来后原样还原」：Worker 单进程串行，同一时刻
    只有一个任务在跑，补丁不会互相污染。补丁点：
      - _qa.split_audio_into_chunks：拿到 N（模块级函数，替换/还原都简单）
      - model.model.generate：每次 ASR 解码 +batch
      - model.forced_aligner.align：每次对齐 +batch
    """

    def __init__(self, model, cb):
        self.model = model
        self.cb = cb
        self.n = 0          # 总块数（split 后确定）
        self.done = 0       # 已完成步数（ASR 生成 + 对齐）
        self.last = -1      # 上次上报的整百分比（去抖，避免每步都落盘）
        self._saved = []    # [(obj, name, orig|_MISSING)] 供还原

    def reset(self):
        """OOM 重试会重新 split+推理，计数需归零以免翻倍。"""
        self.n = self.done = 0
        self.last = -1

    def _emit(self):
        if self.n <= 0:
            return
        pct = max(1, min(99, int(self.done / (2 * self.n) * 100)))
        if pct != self.last:
            self.last = pct
            try:
                self.cb(pct)
            except Exception:
                pass  # 进度落盘失败绝不能拖垮转写

    def _wrap(self, obj, name, make):
        orig = getattr(obj, name)
        had = name in getattr(obj, "__dict__", {})
        setattr(obj, name, make(orig))
        self._saved.append((obj, name, orig if had else _MISSING))

    def install(self):
        import qwen_asr.inference.qwen3_asr as _qa

        def split_maker(orig):
            def wrap(*a, **k):
                parts = orig(*a, **k)
                self.n += len(parts)
                return parts
            return wrap
        self._wrap(_qa, "split_audio_into_chunks", split_maker)

        def gen_maker(orig):
            def wrap(*a, **k):
                out = orig(*a, **k)
                ids = k.get("input_ids")
                self.done += int(ids.shape[0]) if ids is not None else 1
                self._emit()
                return out
            return wrap
        self._wrap(self.model.model, "generate", gen_maker)

        aligner = getattr(self.model, "forced_aligner", None)
        if aligner is not None:
            def align_maker(orig):
                def wrap(*a, **k):
                    out = orig(*a, **k)
                    audio = k.get("audio") if "audio" in k else (a[0] if a else None)
                    self.done += len(audio) if audio is not None else 1
                    self._emit()
                    return out
                return wrap
            self._wrap(aligner, "align", align_maker)
        return self

    def restore(self):
        for obj, name, val in reversed(self._saved):
            try:
                if val is _MISSING:
                    delattr(obj, name)
                else:
                    setattr(obj, name, val)
            except Exception:
                pass
        self._saved.clear()


def transcribe(model, audio_path: str, language=None, progress_cb=None) -> dict:
    """转写 + 对齐，返回 items.json 结构（全文 + 逐 token 时间戳）。

    OOM 自愈（R4）：捕获显存不足（含普通 RuntimeError 形态，见 _is_oom）→
    empty_cache → 原参重试 1 次；再失败由调用方标记 GPU_OOM。
    progress_cb(pct:int)：可选，转写过程中按分块进度回调 1-99 的真实百分比。
    """
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    patch = _ProgressPatch(model, progress_cb).install() if progress_cb else None

    def _run():
        if patch:
            patch.reset()
        return model.transcribe(audio=audio_path, language=language,
                                return_time_stamps=True)

    try:
        try:
            results = _run()
        except Exception as e:
            if not _is_oom(e):
                raise
            # 显存不足：清缓存后原参重试一次。注意：若驱动级 OOM 已损坏 CUDA 上下文，
            # 同进程重试通常仍失败并再次抛出，交由调用方标记 GPU_OOM（干净失败，
            # 优于 Worker 硬退出的 WORKER_CRASHED）。
            torch.cuda.empty_cache()
            results = _run()
    finally:
        if patch:
            patch.restore()

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
