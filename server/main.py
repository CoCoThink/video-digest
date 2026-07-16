#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FastAPI 应用与路由（详细设计 §6）。

一条 uvicorn 进程：既提供 /api 路由，又用 StaticFiles 托管 web/（同源免 CORS）。
本进程不 import torch、不触碰 CUDA；GPU 全在 Worker 子进程。
"""
import logging
import os
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import store
from .config import settings
from .errors import ApiError, of
from .probe import probe_audio
from .scheduler import Scheduler

log = logging.getLogger("vd.api")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [api] %(levelname)s %(message)s")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
CHUNK = 1 << 20  # 1MB 流式落盘分块

scheduler: Scheduler | None = None


# ---------------------------------------------------------------------------
# 应用生命周期
# ---------------------------------------------------------------------------
async def _lifespan(app: FastAPI):
    global scheduler
    store.ensure_dirs(settings.data_dir)
    n = store.startup_recover(settings.data_dir)
    if n:
        log.info("启动恢复：%d 个中断任务标记为 failed(INTERRUPTED)。", n)
    scheduler = Scheduler(settings)
    scheduler.start()
    yield
    if scheduler:
        scheduler.shutdown()


app = FastAPI(title="在线视频生成字幕系统", version="1.0", lifespan=_lifespan)


@app.exception_handler(ApiError)
async def _api_error_handler(request: Request, exc: ApiError):
    return JSONResponse(status_code=exc.http,
                        content={"error": {"code": exc.code, "message": exc.message}})


@app.middleware("http")
async def _no_cache_static(request: Request, call_next):
    """静态资源一律 no-cache（改动后浏览器立即失效重取，无构建部署必需）。
    大体积 vendor（如 32MB wasm）带 ETag，未变更时仅回 304 不重传，代价极小；
    但能保证已打补丁的 ffmpeg.js 不会被旧缓存挡住。"""
    resp = await call_next(request)
    path = request.url.path
    if not path.startswith("/api/"):
        # HTML 文档用 no-store（绝不复用缓存），确保永远拿到引用最新 ?v= 的页面；
        # 其它静态资源 no-cache（带 ETag，未变更回 304，变更即取新）。
        if path in ("/", "/index.html") or path.endswith(".html"):
            resp.headers["Cache-Control"] = "no-store"
        else:
            resp.headers["Cache-Control"] = "no-cache, must-revalidate"
    return resp


# ---------------------------------------------------------------------------
# API 路由
# ---------------------------------------------------------------------------
@app.post("/api/tasks", status_code=201)
async def create_task(file: UploadFile = File(...), filename: str = Form(...)):
    # 1) 队列容量检查（最廉价，先拒）
    if scheduler.snapshot()["queue_depth"] >= settings.queue_capacity:
        raise of("QUEUE_FULL")

    # 2) 流式落盘 + 边写边计大小
    tid = store.gen_task_id()
    tdir = store.task_dir(settings.data_dir, tid)
    tdir.mkdir(parents=True, exist_ok=True)
    upload = tdir / "upload.bin"
    size = 0
    try:
        with open(upload, "wb") as f:
            while True:
                chunk = await file.read(CHUNK)
                if not chunk:
                    break
                size += len(chunk)
                if size > settings.max_size_bytes:
                    raise of("SIZE_EXCEEDED",
                             f"音频超过 {settings.max_size_bytes // (1 << 20)}MB 上限。")
                f.write(chunk)
        if size == 0:
            raise of("EMPTY_AUDIO")

        # 3) ffprobe 内容嗅探（子进程调用放线程池，避免阻塞事件循环）
        from fastapi.concurrency import run_in_threadpool
        duration = await run_in_threadpool(
            probe_audio, str(upload), settings.max_duration_sec)

        # 4) 建任务 / 入队
        store.create_task(settings.data_dir, tid, filename or file.filename or "subtitle",
                          duration, size)
        if not scheduler.enqueue(tid):
            raise of("QUEUE_FULL")
    except ApiError:
        _rmtree(tdir)
        raise
    except Exception as e:
        _rmtree(tdir)
        log.exception("创建任务失败")
        raise of("INTERNAL") from e

    pos = scheduler.position(tid)
    return {
        "task_id": tid,
        "state": "queued",
        "queue_position": pos,
        "audio_duration_sec": round(duration, 1),
        "estimated_process_sec": round(duration * settings.rtf),
    }


@app.get("/api/tasks")
async def list_tasks(include_deleted: bool = False):
    """历史任务列表（最新在前）。用户重开浏览器后可据此找回已完成/进行中的任务。"""
    items = store.list_tasks(settings.data_dir, include_deleted=include_deleted)
    for it in items:
        it["queue_position"] = (
            scheduler.position(it["task_id"]) if it.get("state") == "queued" else None
        )
    return {"tasks": items}


@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    tdir = _require_task_dir(task_id)
    st = store.read_status(tdir)
    if st is None:
        raise of("TASK_NOT_FOUND")
    st = dict(st)
    st["queue_position"] = scheduler.position(task_id) if st.get("state") == "queued" else None
    return st


@app.delete("/api/tasks/{task_id}")
async def delete_task(task_id: str):
    tdir = _require_task_dir(task_id)
    st = store.read_status(tdir)
    if st is None:
        raise of("TASK_NOT_FOUND")
    state = st.get("state")

    if state == "deleted":
        return {"state": "deleted"}                       # 幂等
    if state == "queued":
        scheduler.cancel(task_id)
        store.mark_cancelled(tdir)
        return {"state": "cancelled"}
    if state in ("preprocessing", "transcribing", "postprocessing") or scheduler.is_running(task_id):
        raise of("TASK_RUNNING")                          # 运行中不支持取消（P1）
    # done / failed / cancelled → 删产物留墓碑
    store.delete_task(tdir)
    return {"state": "deleted"}


@app.get("/api/tasks/{task_id}/result")
async def download_result(task_id: str, format: str = "srt"):
    if format not in ("srt", "txt"):
        raise of("BAD_FORMAT")
    tdir = _require_task_dir(task_id)
    st = store.read_status(tdir)
    if st is None:
        raise of("TASK_NOT_FOUND")
    if st.get("state") == "deleted":
        raise of("TASK_DELETED")
    if st.get("state") != "done":
        raise of("RESULT_NOT_READY")

    fname = "subtitle.srt" if format == "srt" else "transcript.txt"
    path = tdir / fname
    if not path.exists():
        raise of("RESULT_NOT_READY")

    stem = Path(st.get("original_filename") or "subtitle").stem or "subtitle"
    dl_name = f"{stem}.{format}"
    disposition = f"attachment; filename*=UTF-8''{quote(dl_name)}"
    return FileResponse(
        path, media_type="application/octet-stream",
        headers={"Content-Disposition": disposition},
    )


@app.get("/api/config")
async def get_config():
    return {
        "rtf": settings.rtf,
        "max_duration_sec": settings.max_duration_sec,
        "max_size_bytes": settings.max_size_bytes,
        "queue_capacity": settings.queue_capacity,
        "accepted_note": "任意 ffmpeg 可解码音频",
    }


@app.get("/api/health")
async def health():
    snap = scheduler.snapshot() if scheduler else {}
    return {
        "status": "ok",
        "model_loaded": snap.get("worker_alive", False),
        "worker_alive": snap.get("worker_alive", False),
        "queue_depth": snap.get("queue_depth", 0),
        "current_task_id": snap.get("current_task_id"),
    }


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------
def _require_task_dir(task_id: str) -> Path:
    if not store.valid_task_id(task_id):
        raise of("TASK_NOT_FOUND")                        # 非法 id 视为不存在
    return store.task_dir(settings.data_dir, task_id)


def _rmtree(tdir: Path) -> None:
    try:
        for f in tdir.iterdir():
            if f.is_file():
                f.unlink()
        tdir.rmdir()
    except OSError:
        pass


# 静态页托管（放在 API 路由之后；/api/* 已先注册，优先级更高）。
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


def run():
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    run()
