#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU Worker 子进程（详细设计 §4.3）。

spawn 启动 → 加载常驻模型（约 14s，一次性）→ 阻塞读任务队列 → 每个 task_id
独立走完 预处理→转写→后处理→产物 全链，逐阶段写 status.json，结束回报结果队列。

**本模块顶层不导入 torch / qwen_asr**：scheduler 在 API 进程内 import 本模块以
取得 worker_main 入口，若顶层引入 CUDA 会污染无 CUDA 的 API 进程。所有重依赖
延迟到 worker_main 内（此时已在子进程中）。
"""
import logging
import os
import sys
import traceback
from pathlib import Path


def _ensure_path():
    """spawn 子进程需能 import server 包（项目根在 sys.path）。"""
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)


def worker_main(task_q, result_q, cfg: dict, ready_ev=None, startup_q=None):
    """子进程入口。cfg = settings.model_dump()。"""
    _ensure_path()
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from server import store
    from server.pipeline import asr

    log = logging.getLogger("vd.worker")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [worker] %(levelname)s %(message)s")

    data_dir = cfg["data_dir"]
    log.info("加载 ASR backend=%s (chunk=%ss) ...",
             cfg.get("asr_backend", "qwen"), cfg["chunk_sec"])
    try:
        backend = asr.create_backend(cfg)
    except Exception as exc:
        if startup_q is not None:
            startup_q.put({
                "backend": cfg.get("asr_backend", "qwen"),
                "error": f"{type(exc).__name__}: {exc}",
            })
        raise
    log.info("模型就绪：backend=%s model=%s，进入任务循环。",
             backend.name, backend.model_id)
    if ready_ev is not None:
        ready_ev.set()
    if startup_q is not None:
        startup_q.put({"backend": backend.name, "model_id": backend.model_id})

    while True:
        task_id = task_q.get()
        if task_id is None:          # 关停信号
            break
        tdir = store.task_dir(data_dir, task_id)
        try:
            _process_one(backend, cfg, tdir, task_id, log)
            result_q.put((task_id, "ok"))
        except Exception as e:       # 业务失败：落盘 failed + 堆栈，仍清理音频
            code = _classify(e)
            msg = _message_for(code, e)
            log.error("任务 %s 失败 [%s]: %s", task_id, code, e)
            _tasklog(tdir, "ERROR " + traceback.format_exc())
            store.mark_failed(tdir, code, msg)
            store.cleanup_audio(tdir)
            result_q.put((task_id, "failed"))


def _process_one(backend, cfg, tdir, task_id, log):
    from server import store
    from server.pipeline import preprocess, asr, layout, subtitle

    tdir = Path(tdir)
    upload = tdir / "upload.bin"
    wav = tdir / "audio_16k.wav"
    _tasklog(tdir, f"START task={task_id}")

    # 1) 预处理：→ 16k 单声道 WAV
    store.start_stage(tdir, "preprocessing")
    preprocess.to_wav_16k(upload, wav)
    store.end_stage(tdir, "preprocessing")
    _tasklog(tdir, "preprocessing done")

    # 2) 转写 + 对齐（耗时主体，GPU ASR/CPU 对齐逐块重叠，含 OOM 自愈）。逐块
    #    回调真实进度写入 status.json，前端据此渲染（无回调时退回不确定动画）。
    store.start_stage(tdir, "transcribing")
    data = backend.transcribe(
        str(wav), language=cfg.get("asr_language"),
        context=str(cfg.get("asr_context") or ""),
        progress_cb=lambda p: store.update_progress(tdir, p),
    )
    peak = data.pop("_peak_gpu_gb", None)
    punc_sec = data.pop("_punc_sec", None)
    import json as _json
    with open(tdir / "items.json", "w", encoding="utf-8") as f:
        _json.dump(data, f, ensure_ascii=False)
    store.end_stage(tdir, "transcribing")
    _tasklog(tdir, f"transcribing done backend={data.get('backend')} "
                   f"model={data.get('model_id')} lang={data.get('language')} "
                   f"tokens={len(data['items'])} chars={len(data['text'])} "
                   f"punc_sec={punc_sec} peak_gpu={peak}GB")

    # 3) 后处理：断句 + 排版 + 产物
    store.start_stage(tdir, "postprocessing")
    cues, info = layout.build_cues(data, cfg)
    subtitle.write_srt(cues, tdir / "subtitle.srt", cfg["layout_line_width"])
    subtitle.write_txt(data["text"], tdir / "transcript.txt")
    store.end_stage(tdir, "postprocessing")
    _tasklog(tdir, f"postprocessing done cues={len(cues)} {info}")

    # 4) 收尾：删音频 → done
    store.cleanup_audio(tdir)
    store.mark_done(tdir, ["srt", "txt"])
    _tasklog(tdir, "DONE")
    log.info("任务 %s 完成：%d cues", task_id, len(cues))


def _classify(e: Exception) -> str:
    name = type(e).__name__
    msg = str(e).lower()
    # 显存不足在 4GB 卡上多以普通 RuntimeError 形态出现（"CUDA driver error: out of
    # memory" / "CUDA out of memory"），需按消息兜底，避免误判成 INTERNAL。
    if ("OutOfMemory" in name or "out of memory" in msg
            or "cuda driver error" in msg):
        return "GPU_OOM"
    if name == "AsrUnstableError":
        return "ASR_UNSTABLE"
    if name == "AsrOutputError":
        return "ASR_OUTPUT_MISMATCH"
    return "INTERNAL"


def _message_for(code: str, e: Exception) -> str:
    if code == "GPU_OOM":
        return "显存不足，转写失败，请重试或改用更短音频。"
    if code == "ASR_UNSTABLE":
        return "模型在局部音频上输出异常，已缩短分块重试但仍未恢复。"
    if code == "ASR_OUTPUT_MISMATCH":
        return "模型正文与时间戳不一致，已停止生成字幕。"
    return "任务处理失败，请重试或联系管理员。"


def _tasklog(tdir, line: str) -> None:
    """任务级日志（各阶段耗时等，不含音频内容，FR-B7）。"""
    from server.store import now_iso
    try:
        with open(Path(tdir) / "task.log", "a", encoding="utf-8") as f:
            f.write(f"{now_iso()} {line}\n")
    except OSError:
        pass
