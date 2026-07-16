#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""预处理：任意上传音频 → 16kHz 单声道 PCM WAV（详细设计 §4.3 步骤 1）。

全链路唯一一次重采样（FR-B1）。喂给模型的采样率固定 16k，正是「无损上传
到服务端后仍需按模型要求重采样」这一步——所以浏览器侧只做无损解封装即可。
"""
import subprocess
from pathlib import Path


def to_wav_16k(src, dst) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(dst),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or not Path(dst).exists():
        raise RuntimeError(f"ffmpeg 转 16k WAV 失败: {p.stderr[-500:]}")
