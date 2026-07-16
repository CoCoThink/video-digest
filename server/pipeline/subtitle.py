#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cue → SRT / TXT 序列化（详细设计 §3.1，P1 预留 VTT）。

时间戳格式与折行逻辑与实测参照物一致：SRT 每条文本经 layout.wrap 折成至多两行。
"""
from . import layout


def fmt_ts(sec: float) -> str:
    if sec < 0:
        sec = 0.0
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def render_srt(cues, line_width: int = 18) -> str:
    parts = []
    for i, (s, e, t) in enumerate(cues, 1):
        parts.append(f"{i}\n{fmt_ts(s)} --> {fmt_ts(e)}\n{layout.wrap(t, line_width)}\n")
    return "\n".join(parts) + "\n" if parts else ""


def write_srt(cues, path, line_width: int = 18) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_srt(cues, line_width))


def write_txt(text: str, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")
