#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""错误码目录（详细设计 §6.7）。

``ApiError`` 携带机器码 + HTTP 状态 + 可直接展示给用户的中文 message；
main.py 注册统一异常处理器，输出 ``{"error": {"code","message"}}``。
"""


class ApiError(Exception):
    def __init__(self, code: str, message: str, http: int):
        self.code = code
        self.message = message
        self.http = http
        super().__init__(f"{code}: {message}")


# code -> (http, 默认中文提示)。上传/下载路径直接 raise ApiError.of("CODE")。
CATALOG = {
    "QUEUE_FULL":        (429, "任务队列已满，请稍后重试（无需重新提取音频）。"),
    "SIZE_EXCEEDED":     (413, "音频文件超过大小上限。"),
    "DURATION_EXCEEDED": (422, "音频时长超过上限。"),
    "NO_AUDIO_STREAM":   (422, "文件不含音频轨道。"),
    "DECODE_FAILED":     (422, "文件损坏或无法解码，请在本机执行 ffmpeg -i 视频 -vn -c:a copy 音频.mka 后上传生成的音频。"),
    "EMPTY_AUDIO":       (422, "音频时长为 0。"),
    "TASK_NOT_FOUND":    (404, "任务不存在。"),
    "TASK_RUNNING":      (409, "任务正在执行，暂不支持中途取消。"),
    "RESULT_NOT_READY":  (404, "结果尚未生成。"),
    "TASK_DELETED":      (410, "结果已被删除。"),
    "BAD_FORMAT":        (400, "不支持的下载格式。"),
    "INTERNAL":          (500, "服务器内部错误。"),
}


def of(code: str, message: str | None = None) -> ApiError:
    http, default = CATALOG.get(code, (500, "服务器内部错误。"))
    return ApiError(code, message or default, http)
