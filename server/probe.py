#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ffprobe 上传校验（详细设计 §4.4）。

不信任扩展名 / Content-Type，用 ffprobe 做内容嗅探：确认有音频轨、可解码、
时长在上限内。任一失败抛 ApiError（映射到 §6.7 错误码）。
"""
import json
import subprocess

from .errors import of


def probe_audio(path, max_duration_sec: int) -> float:
    """返回音频时长（秒）。无音轨/不可解码/时长越界时抛 ApiError。"""
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_streams", "-show_format", str(path),
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (subprocess.SubprocessError, OSError) as e:
        raise of("DECODE_FAILED") from e
    if p.returncode != 0:
        raise of("DECODE_FAILED")
    try:
        data = json.loads(p.stdout)
    except json.JSONDecodeError as e:
        raise of("DECODE_FAILED") from e

    streams = data.get("streams", [])
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    if not audio:
        raise of("NO_AUDIO_STREAM")

    # 时长优先取 format.duration，回退到音频流 duration。
    dur = _to_float(data.get("format", {}).get("duration"))
    if dur <= 0:
        for s in audio:
            d = _to_float(s.get("duration"))
            if d > 0:
                dur = d
                break
    if dur <= 0:
        raise of("EMPTY_AUDIO")
    if dur > max_duration_sec:
        raise of("DURATION_EXCEEDED",
                 f"音频时长 {dur/60:.1f} 分钟，超过上限 {max_duration_sec/3600:.1f} 小时。")
    return dur


def _to_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0
