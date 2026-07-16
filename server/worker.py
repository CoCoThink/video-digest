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


def worker_main(task_q, result_q, cfg: dict, ready_ev=None):
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
    log.info("加载模型 %s (load_8bit=%s, chunk=%ss) ...",
             cfg["asr_model_path"], cfg["load_8bit"], cfg["chunk_sec"])
    model = asr.load_model(cfg)
    log.info("模型就绪，进入任务循环。")
    if ready_ev is not None:
        ready_ev.set()

    while True:
        task_id = task_q.get()
        if task_id is None:          # 关停信号
            break
        tdir = store.task_dir(data_dir, task_id)
        try:
            _process_one(model, cfg, tdir, task_id, log)
            result_q.put((task_id, "ok"))
        except Exception as e:       # 业务失败：落盘 failed + 堆栈，仍清理音频
            code = _classify(e)
            msg = _message_for(code, e)
            log.error("任务 %s 失败 [%s]: %s", task_id, code, e)
            _tasklog(tdir, "ERROR " + traceback.format_exc())
            store.mark_failed(tdir, code, msg)
            store.cleanup_audio(tdir)
            result_q.put((task_id, "failed"))


def _process_one(model, cfg, tdir, task_id, log):
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

    # 2) 转写 + 对齐（耗时主体，含 OOM 自愈）
    store.start_stage(tdir, "transcribing")
    data = asr.transcribe(model, str(wav), language=cfg.get("asr_language"))  # None=自动判定
    peak = data.pop("_peak_gpu_gb", None)
    import json as _json
    with open(tdir / "items.json", "w", encoding="utf-8") as f:
        _json.dump(data, f, ensure_ascii=False)
    store.end_stage(tdir, "transcribing")
    _tasklog(tdir, f"transcribing done lang={data.get('language')} "
                   f"tokens={len(data['items'])} chars={len(data['text'])} peak_gpu={peak}GB")

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
    if "OutOfMemory" in name or "OutOfMemoryError" in name:
        return "GPU_OOM"
    return "INTERNAL"


def _message_for(code: str, e: Exception) -> str:
    if code == "GPU_OOM":
        return "显存不足，转写失败，请重试或改用更短音频。"
    return "任务处理失败，请重试或联系管理员。"


def _tasklog(tdir, line: str) -> None:
    """任务级日志（各阶段耗时等，不含音频内容，FR-B7）。"""
    from server.store import now_iso
    try:
        with open(Path(tdir) / "task.log", "a", encoding="utf-8") as f:
            f.write(f"{now_iso()} {line}\n")
    except OSError:
        pass
