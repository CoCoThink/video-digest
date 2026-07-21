#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FastAPI 应用与路由（详细设计 §6）。

一条 uvicorn 进程：既提供 /api 路由，又用 StaticFiles 托管 web/（同源免 CORS）。
本进程不 import torch、不触碰 CUDA；GPU 全在 Worker 子进程。
"""
import asyncio
import logging
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import store
from .config import settings
from .errors import ApiError, of
from .probe import probe_audio
from .scheduler import Scheduler

log = logging.getLogger("vd.api")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [api] %(levelname)s %(message)s")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

scheduler: Scheduler | None = None
upload_locks: dict[str, asyncio.Lock] = {}


class UploadInit(BaseModel):
    filename: str = "subtitle"
    size: int


# ---------------------------------------------------------------------------
# 应用生命周期
# ---------------------------------------------------------------------------
async def _lifespan(app: FastAPI):
    global scheduler
    store.ensure_dirs(settings.data_dir)
    n = store.startup_recover(settings.data_dir)
    if n:
        log.info("启动恢复：%d 个中断任务标记为 failed(INTERRUPTED)。", n)
    expired = store.cleanup_stale_uploads(
        settings.data_dir, settings.upload_session_ttl_sec)
    if expired:
        log.info("启动清理：删除 %d 个过期的未完成上传。", expired)
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
@app.post("/api/uploads", status_code=201)
async def init_upload(payload: UploadInit):
    """创建分块上传会话，避免大请求被 Cloudflare 在到达源站前拒绝。"""
    if payload.size <= 0:
        raise of("EMPTY_AUDIO")
    if payload.size > settings.max_size_bytes:
        raise of("SIZE_EXCEEDED",
                 f"音频超过 {settings.max_size_bytes // (1 << 20)}MB 上限。")
    store.cleanup_stale_uploads(settings.data_dir, settings.upload_session_ttl_sec)
    if _capacity_in_use() >= settings.queue_capacity:
        raise of("QUEUE_FULL")
    tid = store.gen_task_id()
    filename = (payload.filename or "subtitle").strip()[:1024] or "subtitle"
    try:
        store.create_upload(settings.data_dir, tid, filename, payload.size)
    except Exception as exc:
        log.exception("创建上传会话失败")
        raise of("INTERNAL") from exc
    return {
        "upload_id": tid,
        "chunk_size": settings.upload_chunk_size_bytes,
        "received_size": 0,
    }


@app.get("/api/uploads/{upload_id}")
async def get_upload(upload_id: str):
    tdir, meta = _require_upload(upload_id)
    return {
        "upload_id": upload_id,
        "received_size": _upload_size(tdir),
        "total_size": int(meta["total_size"]),
        "chunk_size": settings.upload_chunk_size_bytes,
    }


@app.put("/api/uploads/{upload_id}")
async def put_upload_chunk(upload_id: str, request: Request):
    """按偏移追加一个原始二进制分块；失败时回滚本次部分写入。"""
    _require_upload(upload_id)  # 不为伪造的随机 id 分配常驻锁对象
    lock = upload_locks.setdefault(upload_id, asyncio.Lock())
    async with lock:
        tdir, meta = _require_upload(upload_id)
        upload = tdir / "upload.bin"
        actual = _upload_size(tdir)
        try:
            offset = int(request.headers.get("Upload-Offset", ""))
        except ValueError:
            raise of("BAD_UPLOAD", "缺少或无法识别 Upload-Offset。")
        if offset != actual:
            raise of("UPLOAD_OFFSET_MISMATCH",
                     f"上传偏移不一致，服务端已接收 {actual} 字节。")

        chunk_limit = max(1, int(settings.upload_chunk_size_bytes))
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > chunk_limit:
                    raise of("UPLOAD_CHUNK_TOO_LARGE",
                             f"单个分块不能超过 {chunk_limit} 字节。")
            except ValueError:
                raise of("BAD_UPLOAD", "Content-Length 无法识别。")

        total = int(meta["total_size"])
        written = 0
        with open(upload, "r+b") as out:
            out.seek(offset)
            try:
                async for part in request.stream():
                    if not part:
                        continue
                    if written + len(part) > chunk_limit:
                        raise of("UPLOAD_CHUNK_TOO_LARGE",
                                 f"单个分块不能超过 {chunk_limit} 字节。")
                    if offset + written + len(part) > total:
                        raise of("SIZE_EXCEEDED", "上传内容超过声明的音频大小。")
                    out.write(part)
                    written += len(part)
            except BaseException:
                out.truncate(offset)
                raise
        if written == 0:
            raise of("BAD_UPLOAD", "上传分块不能为空。")
        store.touch_upload(tdir, meta)
        return {"upload_id": upload_id, "received_size": offset + written}


@app.delete("/api/uploads/{upload_id}")
async def delete_upload(upload_id: str):
    _require_upload(upload_id)
    lock = upload_locks.setdefault(upload_id, asyncio.Lock())
    async with lock:
        tdir, _ = _require_upload(upload_id)
        _rmtree(tdir)
    upload_locks.pop(upload_id, None)
    return {"state": "deleted"}


@app.post("/api/uploads/{upload_id}/complete", status_code=201)
async def complete_upload(upload_id: str):
    """校验完整音频并创建任务；接口幂等，响应丢失后可安全重试。"""
    if not store.valid_task_id(upload_id):
        raise of("UPLOAD_NOT_FOUND")
    tdir = store.task_dir(settings.data_dir, upload_id)
    existing = store.read_status(tdir)
    if existing is not None:
        return _created_task_response(upload_id, existing.get("audio_duration_sec") or 0)

    _require_upload(upload_id)
    lock = upload_locks.setdefault(upload_id, asyncio.Lock())
    try:
        async with lock:
            tdir, meta = _require_upload(upload_id)
            size = _upload_size(tdir)
            if size != int(meta["total_size"]):
                raise of("UPLOAD_INCOMPLETE",
                         f"音频尚未上传完整（{size}/{meta['total_size']} 字节）。")
            if scheduler.snapshot()["queue_depth"] >= settings.queue_capacity:
                raise of("QUEUE_FULL")

            from fastapi.concurrency import run_in_threadpool
            try:
                duration = await run_in_threadpool(
                    probe_audio, str(tdir / "upload.bin"), settings.max_duration_sec)
            except ApiError:
                _rmtree(tdir)
                raise
            except Exception as exc:
                _rmtree(tdir)
                log.exception("完成上传校验失败")
                raise of("INTERNAL") from exc

            store.create_task(
                settings.data_dir, upload_id, meta["filename"], duration, size,
                asr_backend=settings.asr_backend,
                model_id=_selected_model_id(),
            )
            if not scheduler.enqueue(upload_id):
                _rmtree(tdir)
                raise of("QUEUE_FULL")
            store.finish_upload(tdir)
    finally:
        if store.read_upload(tdir) is None:
            upload_locks.pop(upload_id, None)
    return _created_task_response(upload_id, duration)


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
        "rtf": _selected_rtf(),
        "asr_backend": settings.asr_backend,
        "punctuation_enabled": (
            settings.asr_backend == "funasr"
            or (settings.asr_backend == "firered" and settings.firered_punc_enabled)
        ),
        "max_duration_sec": settings.max_duration_sec,
        "max_size_bytes": settings.max_size_bytes,
        "upload_chunk_size_bytes": settings.upload_chunk_size_bytes,
        "queue_capacity": settings.queue_capacity,
        "accepted_note": "任意 ffmpeg 可解码音频",
    }


@app.get("/api/health")
async def health():
    snap = scheduler.snapshot() if scheduler else {}
    loaded = snap.get("model_loaded", False)
    alive = snap.get("worker_alive", False)
    return {
        "status": "ok" if loaded else ("loading" if alive else "error"),
        "backend": settings.asr_backend,
        "model_id": _selected_model_id(),
        "model_loaded": loaded,
        "worker_alive": alive,
        "startup_error": snap.get("startup_error"),
        "queue_depth": snap.get("queue_depth", 0),
        "current_task_id": snap.get("current_task_id"),
    }


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------
def _selected_model_id() -> str:
    if settings.asr_backend == "firered":
        suffix = "+FireRedPunc" if settings.firered_punc_enabled else ""
        return f"FireRedASR2-AED{suffix}"
    if settings.asr_backend == "funasr":
        return settings.funasr_model_path
    return settings.asr_model_path


def _selected_rtf() -> float:
    if settings.asr_backend == "firered":
        return settings.firered_rtf
    if settings.asr_backend == "funasr":
        return settings.funasr_rtf
    return settings.rtf


def _require_task_dir(task_id: str) -> Path:
    if not store.valid_task_id(task_id):
        raise of("TASK_NOT_FOUND")                        # 非法 id 视为不存在
    return store.task_dir(settings.data_dir, task_id)


def _require_upload(upload_id: str) -> tuple[Path, dict]:
    if not store.valid_task_id(upload_id):
        raise of("UPLOAD_NOT_FOUND")
    tdir = store.task_dir(settings.data_dir, upload_id)
    meta = store.read_upload(tdir)
    if meta is None:
        raise of("UPLOAD_NOT_FOUND")
    return tdir, meta


def _upload_size(tdir: Path) -> int:
    try:
        return (tdir / "upload.bin").stat().st_size
    except OSError:
        raise of("UPLOAD_NOT_FOUND")


def _created_task_response(task_id: str, duration: float) -> dict:
    return {
        "task_id": task_id,
        "state": "queued",
        "queue_position": scheduler.position(task_id),
        "audio_duration_sec": round(float(duration), 1),
        "estimated_process_sec": round(float(duration) * _selected_rtf()),
    }


def _capacity_in_use() -> int:
    return (scheduler.snapshot()["queue_depth"]
            + store.count_uploads(settings.data_dir))


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
