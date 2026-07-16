#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""内存队列 + dispatcher 线程 + GPU Worker 生命周期（详细设计 §4.2）。

- 队列：API 进程内 deque（含正在执行的任务算容量），一把 Lock/Condition 保护。
- dispatcher 线程：取队首 → 投 task_id 给 Worker → 带超时等结果 → 超时 kill +
  respawn；Worker 意外死亡则标记 WORKER_CRASHED + respawn。
- 全系统同一时刻至多 1 个任务在执行（FR-B2 单 GPU 串行）。
"""
import logging
import multiprocessing
import queue as pyqueue
import threading
import time
from collections import deque

from . import store
from .worker import worker_main

log = logging.getLogger("vd.scheduler")


class Scheduler:
    def __init__(self, settings):
        self.settings = settings
        self.cfg = settings.model_dump()
        self._deque = deque()          # 等待中的 task_id（不含正在执行）
        self._current = None           # 正在执行的 task_id
        self._cv = threading.Condition()
        self._ctx = multiprocessing.get_context("spawn")  # CUDA 要求 spawn
        self._task_q = None
        self._result_q = None
        self._worker = None
        self._stop = False
        self._thread = None

    # --- 生命周期 --------------------------------------------------------
    def start(self):
        self._spawn_worker(initial=True)
        self._thread = threading.Thread(target=self._dispatch_loop,
                                        name="vd-dispatcher", daemon=True)
        self._thread.start()
        log.info("调度器已启动（队列容量 %d）。", self.settings.queue_capacity)

    def shutdown(self):
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        try:
            if self._worker and self._worker.is_alive():
                self._task_q.put(None)         # 优雅关停信号
                self._worker.join(timeout=5)
                if self._worker.is_alive():
                    self._worker.kill()
        except Exception:
            pass

    def _spawn_worker(self, initial=False):
        # 每次都用全新队列，避免 respawn 后读到上一代残留消息。
        self._task_q = self._ctx.Queue()
        self._result_q = self._ctx.Queue()
        self._worker = self._ctx.Process(
            target=worker_main,
            args=(self._task_q, self._result_q, self.cfg),
            name="vd-gpu-worker", daemon=True,
        )
        self._worker.start()
        log.info("GPU Worker %s (pid=%s)", "启动" if initial else "重启",
                 self._worker.pid)

    # --- 队列操作（对 API 暴露）------------------------------------------
    def enqueue(self, task_id) -> bool:
        with self._cv:
            depth = len(self._deque) + (1 if self._current else 0)
            if depth >= self.settings.queue_capacity:
                return False
            self._deque.append(task_id)
            self._cv.notify()
            return True

    def cancel(self, task_id) -> bool:
        """从等待队列摘除排队中的任务；正在执行的返回 False（由调用方 409）。"""
        with self._cv:
            if task_id in self._deque:
                self._deque.remove(task_id)
                return True
            return False

    def position(self, task_id):
        """queued 任务的 1 基排位（1 = 即将执行）；否则 None。"""
        with self._cv:
            try:
                idx = list(self._deque).index(task_id)
            except ValueError:
                return None
            return (1 if self._current else 0) + idx + 1

    def is_running(self, task_id) -> bool:
        with self._cv:
            return self._current == task_id

    def snapshot(self) -> dict:
        with self._cv:
            return {
                "queue_depth": len(self._deque) + (1 if self._current else 0),
                "current_task_id": self._current,
                "worker_alive": bool(self._worker and self._worker.is_alive()),
            }

    # --- dispatcher 主循环 -----------------------------------------------
    def _dispatch_loop(self):
        while True:
            with self._cv:
                while not self._deque and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                task_id = self._deque.popleft()
                self._current = task_id
            try:
                self._run_task(task_id)
            except Exception:
                log.exception("dispatcher 处理 %s 异常", task_id)
            finally:
                with self._cv:
                    self._current = None

    def _run_task(self, task_id):
        tdir = store.task_dir(self.settings.data_dir, task_id)
        st = store.read_status(tdir)
        if st is None or st.get("state") != "queued":
            return  # 已被取消/删除
        duration = float(st.get("audio_duration_sec") or 0)
        timeout = duration * self.settings.timeout_factor + self.settings.timeout_base_sec

        self._task_q.put(task_id)
        deadline = time.monotonic() + timeout
        while True:
            try:
                self._result_q.get(timeout=1.0)  # 结果 (task_id, ok|failed) 已由 Worker 落盘
                return
            except pyqueue.Empty:
                if not self._worker.is_alive():           # Worker 段错误等意外死亡
                    log.error("Worker 意外死亡，任务 %s 标记 WORKER_CRASHED", task_id)
                    store.mark_failed(tdir, "WORKER_CRASHED", "转写进程异常退出，请重试。")
                    store.cleanup_audio(tdir)
                    self._spawn_worker()
                    return
                if time.monotonic() > deadline:           # 超时：强制终止
                    log.error("任务 %s 超时（%.0fs），强制终止 Worker", task_id, timeout)
                    self._worker.kill()
                    self._worker.join()
                    store.mark_failed(tdir, "TIMEOUT", "任务执行超时，已强制终止。")
                    store.cleanup_audio(tdir)
                    self._spawn_worker()                  # 重载模型（仅异常路径付出 ~14s）
                    return
