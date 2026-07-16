#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务目录与 status.json 读写（详细设计 §5）。

文件系统即状态：每任务一目录，一个 status.json 是唯一状态源。原子写
（临时文件 + os.replace）杜绝轮询读到半截 JSON。API 进程与 Worker 子进程
按 §4.1 的写入者分区共享本模块（同一任务同一时刻只有一个写入者）。
"""
import json
import os
import re
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path

TASK_ID_RE = re.compile(r"^[A-Za-z0-9_-]{16,32}$")
CST = timezone(timedelta(hours=8))
STAGES = ("preprocessing", "transcribing", "postprocessing")
NON_TERMINAL = {"queued", "preprocessing", "transcribing", "postprocessing"}
TERMINAL = {"done", "failed", "cancelled", "deleted"}

# 转写完成/失败后需清理的中间音频（FR-B7：不留音频内容）。
AUDIO_ARTIFACTS = ("upload.bin", "audio_16k.wav")


def now_iso() -> str:
    return datetime.now(CST).isoformat(timespec="seconds")


def gen_task_id() -> str:
    """secrets.token_urlsafe(16) → 128bit 随机、URL 安全、不可猜测（NFR-5）。"""
    return secrets.token_urlsafe(16)


def valid_task_id(tid: str) -> bool:
    return bool(tid) and bool(TASK_ID_RE.match(tid))


def tasks_root(data_dir) -> Path:
    return Path(data_dir) / "tasks"


def task_dir(data_dir, tid: str) -> Path:
    return tasks_root(data_dir) / tid


def ensure_dirs(data_dir) -> None:
    root = tasks_root(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(Path(data_dir), 0o700)  # NFR-5：data/ 权限收紧
    except OSError:
        pass


# ---------------------------------------------------------------------------
# 原子读写
# ---------------------------------------------------------------------------
def _atomic_write(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_status(tdir, status: dict) -> dict:
    status["updated_at"] = now_iso()
    _atomic_write(Path(tdir) / "status.json", status)
    return status


def read_status(tdir) -> dict | None:
    p = Path(tdir) / "status.json"
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# 状态构造与推进
# ---------------------------------------------------------------------------
def new_status(tid: str, filename: str, duration: float, size: int) -> dict:
    t = now_iso()
    return {
        "task_id": tid,
        "state": "queued",
        "original_filename": filename,
        "audio_duration_sec": round(float(duration), 1),
        "audio_size_bytes": int(size),
        "created_at": t,
        "updated_at": t,
        "finished_at": None,
        "progress_percent": None,  # 分块 ASR/对齐的真实完成进度
        "stages": {s: None for s in STAGES},
        "error": None,
        "artifacts": [],
    }


def create_task(data_dir, tid: str, filename: str, duration: float, size: int) -> dict:
    tdir = task_dir(data_dir, tid)
    tdir.mkdir(parents=True, exist_ok=True)
    return write_status(tdir, new_status(tid, filename, duration, size))


def start_stage(tdir, stage: str) -> dict:
    st = read_status(tdir)
    st["state"] = stage
    st["stages"][stage] = {"started_at": now_iso(), "ended_at": None, "elapsed_sec": None}
    return write_status(tdir, st)


def update_progress(tdir, percent) -> None:
    """写入转写阶段的分块级真实进度（0-100，P1）。

    由 Worker 在 transcribing 阶段高频调用（每块 ASR/对齐完成一次），调用方已按
    整百分比去抖，故此处每次落盘的次数≈块数×2，量级很小。读失败（极偶发的半截
    写窗口）直接跳过，不影响主流程。
    """
    st = read_status(tdir)
    if st is None:
        return
    st["progress_percent"] = max(0, min(100, int(percent)))
    write_status(tdir, st)


def end_stage(tdir, stage: str) -> dict:
    st = read_status(tdir)
    s = st["stages"].get(stage)
    if s:
        end = now_iso()
        s["ended_at"] = end
        try:
            started = datetime.fromisoformat(s["started_at"])
            s["elapsed_sec"] = round((datetime.fromisoformat(end) - started).total_seconds(), 2)
        except (ValueError, TypeError):
            s["elapsed_sec"] = None
    return write_status(tdir, st)


def mark_done(tdir, artifacts: list[str]) -> dict:
    st = read_status(tdir)
    st["state"] = "done"
    st["finished_at"] = now_iso()
    st["artifacts"] = artifacts
    st["error"] = None
    return write_status(tdir, st)


def mark_failed(tdir, code: str, message: str) -> dict | None:
    st = read_status(tdir)
    if st is None:
        return None
    st["state"] = "failed"
    st["finished_at"] = now_iso()
    st["error"] = {"code": code, "message": message}
    return write_status(tdir, st)


def mark_cancelled(tdir) -> dict:
    st = read_status(tdir)
    st["state"] = "cancelled"
    st["finished_at"] = now_iso()
    return write_status(tdir, st)


def delete_task(tdir) -> dict:
    """删产物与日志，留墓碑 status.json（state=deleted，D6）。"""
    tdir = Path(tdir)
    st = read_status(tdir) or {}
    tomb = {
        "task_id": st.get("task_id", tdir.name),
        "state": "deleted",
        "original_filename": st.get("original_filename"),
        "created_at": st.get("created_at"),
        "updated_at": now_iso(),
    }
    for f in tdir.iterdir():
        if f.name != "status.json" and f.is_file():
            try:
                f.unlink()
            except OSError:
                pass
    _atomic_write(tdir / "status.json", tomb)
    return tomb


def list_tasks(data_dir, include_deleted: bool = False, limit: int = 200) -> list[dict]:
    """列出所有任务的摘要，按 created_at 倒序（最新在前）。

    仅返回轻量字段供历史列表渲染；默认隐藏 deleted 墓碑（不可恢复）。
    """
    root = tasks_root(data_dir)
    if not root.exists():
        return []
    items = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        st = read_status(d)
        if not st:
            continue
        if not include_deleted and st.get("state") == "deleted":
            continue
        items.append({
            "task_id": st.get("task_id", d.name),
            "state": st.get("state"),
            "original_filename": st.get("original_filename"),
            "audio_duration_sec": st.get("audio_duration_sec"),
            "created_at": st.get("created_at"),
            "updated_at": st.get("updated_at"),
            "finished_at": st.get("finished_at"),
            "error": st.get("error"),
        })
    # created_at 同为 +08:00 ISO，字典序即时间序；缺失回退 updated_at。
    items.sort(key=lambda x: (x.get("created_at") or x.get("updated_at") or ""), reverse=True)
    return items[:limit]


def cleanup_audio(tdir) -> None:
    """删除中间音频文件（FR-B7），done / failed 均调用。"""
    for name in AUDIO_ARTIFACTS:
        p = Path(tdir) / name
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# 启动恢复（详细设计 §4.2）
# ---------------------------------------------------------------------------
def startup_recover(data_dir) -> int:
    """服务启动时把所有非终态任务改写为 failed(INTERRUPTED)；终态原样保留。"""
    root = tasks_root(data_dir)
    if not root.exists():
        return 0
    n = 0
    for d in root.iterdir():
        if not d.is_dir():
            continue
        st = read_status(d)
        if st and st.get("state") in NON_TERMINAL:
            mark_failed(d, "INTERRUPTED", "服务已重启，请重新上传。")
            cleanup_audio(d)  # FR-B7：中断任务同样不留音频
            n += 1
    return n
