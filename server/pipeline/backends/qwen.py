"""Compatibility wrapper around the existing Qwen implementation."""
from __future__ import annotations


class QwenBackend:
    name = "qwen"

    def __init__(self, cfg: dict):
        # Importing this module is cheap. load_model performs all torch/qwen imports.
        from server.pipeline import asr

        self.model_id = cfg["asr_model_path"]
        self._model = asr.load_model(cfg)

    def transcribe(
        self, audio_path: str, *, language=None, context="", progress_cb=None,
    ) -> dict:
        from server.pipeline import asr

        data = asr.transcribe(
            self._model, audio_path, language=language, context=context,
            progress_cb=progress_cb,
        )
        data["backend"] = self.name
        data["model_id"] = self.model_id
        data.setdefault("warnings", [])
        return data
